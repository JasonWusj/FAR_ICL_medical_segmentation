"""Train-only support bank with content-addressed feature caches."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from .config import atomic_save, digest, file_digest, read_tensor
from .data import load_case
from .features import boundary, describe


class CaseBank:
    def __init__(self, cases, cfg, encoder, manifest_hash):
        self.cfg, self.encoder = cfg, encoder
        self.device = torch.device(cfg["device"])
        relevant = {
            k: cfg[k] for k in ("image_size", "encoder", "encoder_repo", "encoder_weights", "mask_values")
        }
        if cfg["encoder_weights"]:
            relevant["weights_hash"] = file_digest(cfg["encoder_weights"])
        # Hash actual encoder parameters so changes in upstream weights invalidate caches.
        import hashlib

        h = hashlib.sha256()
        for name, value in encoder.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.signature = digest(
            dict(manifest=manifest_hash, config=relevant, encoder=h.hexdigest(), schema=1)
        )
        cache_parent = os.environ.get("FARICL_BANK_CACHE_ROOT")
        self.root = (Path(cache_parent) if cache_parent else Path(cfg["output"]) / "cache") / self.signature[:20]
        self.entries = []
        training_cases = [
            case for case in cases
            if case.split == "train" and (not cfg["bank_domains"] or case.domain in cfg["bank_domains"])
        ]
        batch_size = int(os.environ.get("FARICL_BANK_BATCH", "16"))
        workers = int(os.environ.get("FARICL_BANK_WORKERS", "4"))
        if batch_size < 1 or workers < 1:
            raise ValueError("FARICL_BANK_BATCH and FARICL_BANK_WORKERS must be positive")
        cached_count = built_count = 0
        # Concurrent reads/writes hide network-filesystem latency. The encoder
        # sees one batch, while cache entries retain the old per-case format.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            with tqdm(total=len(training_cases), desc="Case bank") as progress:
                for start in range(0, len(training_cases), batch_size):
                    batch = training_cases[start:start + batch_size]
                    paths = [self.root / f"{digest(case.case_id)}.pt" for case in batch]

                    def load_or_prepare(item):
                        case, path = item
                        if path.exists():
                            return read_tensor(path), None
                        return None, load_case(case, cfg)

                    loaded = list(pool.map(load_or_prepare, zip(batch, paths)))
                    entries = [entry for entry, _ in loaded]
                    missing = [index for index, entry in enumerate(entries) if entry is None]
                    cached_count += len(batch) - len(missing)
                    built_count += len(missing)
                    if missing:
                        rgb = torch.stack([loaded[index][1]["rgb"] for index in missing]).to(self.device)
                        z, fmap = encoder(rgb)
                        pending_writes = []
                        for offset, index in enumerate(missing):
                            case, data = batch[index], loaded[index][1]
                            mask = data["mask"].to(self.device)
                            # Known support GT is used only for bank features.
                            features = describe(z[offset], fmap[offset], mask, boundary(mask),
                                                cfg["hard_quantile"])
                            entry = {
                                "case": asdict(case),
                                "image": data["image"],
                                "mask": data["mask"],
                                "features": {key: value.cpu() for key, value in features.items()},
                            }
                            entries[index] = entry
                            pending_writes.append(pool.submit(atomic_save, entry, paths[index]))
                        for future in pending_writes:
                            future.result()
                    self.entries.extend(entries)
                    progress.update(len(batch))
        print(f"Case bank ready: {cached_count} cached, {built_count} newly computed")
        if not self.entries:
            raise ValueError("No training cases available for bank")
        self.vectors = torch.stack([e["features"]["global_feature"] for e in self.entries]).to(self.device)

    def candidates(self, case, vector, n, rng=None):
        allowed = [
            i
            for i, e in enumerate(self.entries)
            if e["case"]["patient_id"] != case.patient_id and e["case"]["task"] == case.task
        ]
        if not allowed:
            raise ValueError(f"No support from another patient for {case.case_id}/{case.task}")
        if rng is not None:
            ordered = list(rng.permutation(allowed))
        else:
            similarity = self.vectors[allowed] @ vector
            ordered = [allowed[j] for j in similarity.argsort(descending=True).tolist()]
        return [self.entries[i] for i in ordered[:n]]

    def features_on_device(self, entries):
        return [{k: v.to(self.device) for k, v in e["features"].items()} for e in entries]
