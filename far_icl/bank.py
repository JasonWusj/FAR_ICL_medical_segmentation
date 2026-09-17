"""Train-only support bank with content-addressed feature caches."""

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
        self.root = Path(cfg["output"]) / "cache" / self.signature[:20]
        self.entries = []
        for case in tqdm(cases, desc="Case bank"):
            if case.split != "train" or (cfg["bank_domains"] and case.domain not in cfg["bank_domains"]):
                continue
            cache = self.root / f"{digest(case.case_id)}.pt"
            if cache.exists():
                entry = read_tensor(cache)
            else:
                data = load_case(case, cfg)
                z, fmap = encoder(data["rgb"][None].to(self.device))
                mask = data["mask"].to(self.device)
                # Support hard features use known GT boundary, never a query GT mask.
                features = describe(z[0], fmap[0], mask, boundary(mask), cfg["hard_quantile"])
                entry = {
                    "case": asdict(case),
                    "image": data["image"],
                    "mask": data["mask"],
                    "features": {k: v.cpu() for k, v in features.items()},
                }
                atomic_save(entry, cache)
            self.entries.append(entry)
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
