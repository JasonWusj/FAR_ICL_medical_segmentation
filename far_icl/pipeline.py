"""Inference and offline supervision share identical label-free retrieval features."""

import hashlib
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .bank import CaseBank
from .config import atomic_save, digest, file_digest, read_tensor, seed_all, write_json
from .data import load_case, load_mask, manifest_signature, read_manifest
from .features import ImageEncoder, describe, pair_features, set_features
from .metrics import dice, diversity, segmentation_metrics
from .retrieval import UtilityRanker, select
from .segmentation import Segmenter, estimate_uncertainty

CORRECTION_METHODS = {"repair", "repair_cover", "repair_adaptive"}
LEARNED = {"utility", "mmr", "set", "adaptive", "adaptive_learned"} | CORRECTION_METHODS


def ranker_kind(method):
    if method in CORRECTION_METHODS:
        return "correction"
    return "marginal" if method in {"set", "adaptive_learned"} else "utility"


class Pipeline:
    def __init__(self, cfg, bank_only=False):
        self.cfg = cfg
        seed_all(cfg["seed"], cfg["deterministic"])
        self.device = torch.device(cfg["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly use --set device=cpu if intended")
        self.cases = read_manifest(cfg["manifest"], cfg["identity_scope"])
        self.manifest_hash = manifest_signature(self.cases)
        self.encoder = ImageEncoder(cfg).to(self.device).eval()
        self.bank = CaseBank(self.cases, cfg, self.encoder, self.manifest_hash)
        if bank_only:
            return
        if cfg["segmenter"] == "tyche":
            from .tyche_adapter import TycheSegmenter

            self.segmenter = TycheSegmenter(cfg)
        else:
            self.segmenter = Segmenter(cfg)
        h = hashlib.sha256()
        for name, value in self.segmenter.model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        keys = (
            "seed",
            "identity_scope",
            "candidate_n",
            "initial_k",
            "uncertainty",
            "uncertainty_samples",
            "hard_quantile",
            "feature_mode",
            "max_k",
            "set_trajectories",
            "bank_domains",
            "query_domains",
            "train_query_domains",
            "segmenter",
            "tyche_samples",
        )
        self.signature = digest(
            dict(
                bank=self.bank.signature, segmenter=h.hexdigest(), config={k: cfg[k] for k in keys}, schema=1
            )
        )
        self.artifacts = Path(cfg["output"]) / "supervision" / self.signature[:20]

    def signature_for(self, kind):
        if kind == "correction":
            return digest(dict(base=self.signature, grid=self.cfg["correction_grid"], correction_schema=1))
        return self.signature

    def artifact_dir(self, kind):
        return Path(self.cfg["output"]) / "supervision" / self.signature_for(kind)[:20] / kind

    def queries(self, split):
        domains = self.cfg["train_query_domains"] if split == "train" else self.cfg["query_domains"]
        cases = [c for c in self.cases if c.split == split and (not domains or c.domain in domains)]
        if not cases:
            raise ValueError(f"No query cases for split={split} / domains={domains}")
        return cases

    def rng_for(self, case):
        seed = int(digest([self.cfg["seed"], case.case_id])[:8], 16)
        # Per-query model RNG makes stochastic Tyche independent of cached/skipped queries.
        torch.manual_seed(seed)
        return np.random.default_rng(seed)

    @torch.no_grad()
    def context(self, case, random_candidates=False, need_failure=True, with_regions=False, data=None):
        rng = self.rng_for(case)
        data = data if data is not None else load_case(case, self.cfg, with_mask=False)
        z, fmap = self.encoder(data["rgb"][None].to(self.device))
        candidates = self.bank.candidates(
            case, z[0], self.cfg["candidate_n"], rng if random_candidates else None
        )
        features = self.bank.features_on_device(candidates)
        if need_failure:
            initial = candidates[: min(self.cfg["initial_k"], len(candidates))]
            pred = self.segmenter(data["image"], initial)
            uncertainty = estimate_uncertainty(
                self.segmenter, data["image"], candidates, initial, self.cfg, rng
            )
        else:
            pred = torch.zeros_like(data["image"], device=self.device)
            uncertainty = torch.zeros_like(pred)
        query = describe(z[0], fmap[0], pred, uncertainty, self.cfg["hard_quantile"])
        if with_regions:
            from .correction import regional_state

            data["regional_state"] = regional_state(fmap[0], pred, uncertainty, self.cfg["correction_grid"])
        return data, query, candidates, features, rng, pred, uncertainty

    @torch.no_grad()
    def generate(self, kind, split, force=False):
        if split not in {"train", "val"}:
            raise ValueError("Test labels must never generate training/validation supervision")
        target = self.artifact_dir(kind) / split
        target.mkdir(parents=True, exist_ok=True)
        for case in tqdm(self.queries(split), desc=f"Generate {kind}/{split}"):
            output = target / f"{digest(case.case_id)}.pt"
            if output.exists() and not force:
                continue
            data, query, candidates, features, rng, initial, _ = self.context(
                case,
                need_failure=kind == "correction" or self.cfg["feature_mode"] != "image",
                with_regions=kind == "correction",
            )
            truth = load_case(case, self.cfg)["mask"].to(self.device)
            groups = []
            if kind == "correction":
                from .correction import correction_targets

                targets = []
                for candidate in candidates:
                    refined = self.segmenter(data["image"], [candidate])
                    targets.append(
                        correction_targets(initial, refined, truth, self.cfg["correction_grid"]).cpu()
                    )
                # Reuse bank features by ID. Never persist N x regions expanded inputs.
                atomic_save(
                    dict(
                        query={key: value.cpu() for key, value in query.items()},
                        regional_state=data["regional_state"].cpu(),
                        candidate_ids=[c["case"]["case_id"] for c in candidates],
                        targets=torch.stack(targets),
                        case_id=case.case_id,
                        patient_id=case.patient_id,
                        split=split,
                        kind=kind,
                        signature=self.signature_for(kind),
                        storage="indexed-correction-v1",
                    ),
                    output,
                )
                continue
            if kind == "utility":
                x = torch.stack([pair_features(query, f, self.cfg["feature_mode"]) for f in features])
                y = torch.tensor([dice(self.segmenter(data["image"], [c]), truth) for c in candidates])
                groups.append({"x": x.cpu(), "y": y})
            elif kind == "marginal":
                for trajectory in range(self.cfg["set_trajectories"]):
                    selected, base = [], 0.0
                    for depth in range(min(self.cfg["max_k"], len(candidates))):
                        available = [j for j in range(len(candidates)) if j not in selected]
                        x = torch.stack(
                            [
                                set_features(
                                    query,
                                    features[j],
                                    [features[s] for s in selected],
                                    self.cfg["feature_mode"],
                                    self.cfg["max_k"],
                                )
                                for j in available
                            ]
                        )
                        utility = [
                            dice(
                                self.segmenter(data["image"], [candidates[s] for s in selected + [j]]), truth
                            )
                            for j in available
                        ]
                        groups.append({"x": x.cpu(), "y": torch.tensor(utility) - base})
                        # Include both greedy and random state trajectories to reduce policy bias.
                        choice = (
                            int(np.argmax(utility)) if trajectory == 0 else int(rng.integers(len(available)))
                        )
                        selected.append(available[choice])
                        base = utility[choice]
            else:
                raise ValueError(kind)
            atomic_save(
                dict(
                    groups=groups,
                    case_id=case.case_id,
                    patient_id=case.patient_id,
                    split=split,
                    kind=kind,
                    signature=self.signature,
                ),
                output,
            )
        write_json(
            dict(
                signature=self.signature_for(kind),
                kind=kind,
                split=split,
                query_count=len(self.queries(split)),
            ),
            target / "metadata.json",
        )
        return target

    def load_ranker(self, path, method):
        if method not in LEARNED:
            return None
        if not path:
            raise ValueError(f"{method} requires --checkpoint")
        checkpoint = read_tensor(path)
        expected = ranker_kind(method)
        if checkpoint["kind"] != expected or checkpoint["signature"] != self.signature_for(expected):
            raise ValueError("Checkpoint type/provenance mismatch; use matching config and bank")
        if expected == "correction":
            from .correction import CorrectionRanker

            model = CorrectionRanker(checkpoint["input_dim"], checkpoint["hidden_dim"]).to(self.device)
        else:
            model = UtilityRanker(checkpoint["input_dim"], checkpoint["hidden_dim"]).to(self.device)
        model.load_state_dict(checkpoint["model"])
        return model.eval()

    @torch.no_grad()
    def evaluate(self, method, split, checkpoint=None, tag=None):
        ranker = self.load_ranker(checkpoint, method)
        name = tag or f"{method}_k{self.cfg['k']}_{split}_seed{self.cfg['seed']}_{digest(self.cfg)[:8]}"
        if Path(name).name != name:
            raise ValueError("tag must be a single directory name")
        result_dir = Path(self.cfg["output"]) / "results" / name
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.cfg, result_dir / "config.json")
        import platform
        import subprocess

        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        write_json(
            dict(
                signature=self.signature_for(ranker_kind(method)),
                manifest_hash=self.manifest_hash,
                identity_scope=self.cfg["identity_scope"],
                git_revision=revision.stdout.strip() or "uncommitted",
                python=platform.python_version(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                checkpoint_sha256=file_digest(checkpoint) if method in LEARNED else None,
            ),
            result_dir / "provenance.json",
        )
        cases = self.queries(split)
        io_workers = int(os.environ.get("FARICL_EVAL_IO_WORKERS", "4"))
        if io_workers < 1:
            raise ValueError("FARICL_EVAL_IO_WORKERS must be positive")

        def prefetched_queries():
            # Only image data is prefetched. Query masks remain unread until
            # support selection and prediction have completed.
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                ahead = min(io_workers * 2, len(cases))
                pending = deque(pool.submit(load_case, case, self.cfg, False) for case in cases[:ahead])
                for index, case in enumerate(cases):
                    future = pending.popleft()
                    if index + ahead < len(cases):
                        pending.append(pool.submit(load_case, cases[index + ahead], self.cfg, False))
                    wait_start = time.perf_counter()
                    data = future.result()
                    yield case, data, time.perf_counter() - wait_start

        rows = []
        for case, query_data, query_io_wait in tqdm(prefetched_queries(), total=len(cases), desc=name):
            self.segmenter.reset_stats()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.segmenter.sync()
            start = time.perf_counter()
            need_failure = method in {"shape", "roles", "adaptive"} | CORRECTION_METHODS or (
                method not in {"random", "knn"} and self.cfg["feature_mode"] != "image"
            )
            data, query, candidates, features, rng, pred0, uncertainty = self.context(
                case,
                random_candidates=method == "random",
                need_failure=need_failure,
                with_regions=method in CORRECTION_METHODS,
                data=query_data,
            )
            gains = []
            regional_predictions = None
            if method in {"random", "knn"}:
                selected = list(range(min(self.cfg["k"], len(candidates))))
            elif method in CORRECTION_METHODS:
                from .correction import correction_inputs, select_corrections

                inputs = correction_inputs(query, features, data["regional_state"], self.cfg["feature_mode"])
                regional_predictions = ranker(inputs)
                # Last five region channels: probability, uncertainty, boundary, y, x.
                region_uncertainty = data["regional_state"][:, -4]
                selected, gains = select_corrections(
                    regional_predictions, region_uncertainty, self.cfg, method
                )
            elif method == "roles":
                # Distinct supports for appearance, morphology and failure roles.
                selected = [0]
                for key in ("shape", "hard"):
                    score = [
                        float(torch.nn.functional.cosine_similarity(query[key], f[key], dim=0))
                        if j not in selected
                        else -float("inf")
                        for j, f in enumerate(features)
                    ]
                    if len(selected) < len(candidates):
                        selected.append(int(np.argmax(score)))
            else:
                selected, gains = select(query, features, self.cfg, method, ranker)
            supports = [candidates[j] for j in selected]
            prediction = self.segmenter(data["image"], supports)
            self.segmenter.sync()
            elapsed = time.perf_counter() - start
            # Query ground truth is first read AFTER selection and prediction.
            scoring_start = time.perf_counter()
            truth = load_mask(case, self.cfg, data["original_size"])
            row = dict(
                case_id=case.case_id,
                patient_id=case.patient_id,
                task=case.task,
                domain=case.domain,
                method=method,
                split=split,
                k=len(selected),
                support_ids=[s["case"]["case_id"] for s in supports],
                support_patient_ids=[s["case"]["patient_id"] for s in supports],
                predicted_gains=gains,
                uncertainty=float(uncertainty.mean()),
                query_io_wait_seconds=query_io_wait,
                elapsed_seconds=elapsed,
                segmentation_seconds=self.segmenter.seconds,
                retrieval_seconds=max(0, elapsed - self.segmenter.seconds),
                segmentation_calls=self.segmenter.calls,
                support_evaluations=self.segmenter.support_evaluations,
                peak_gpu_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2
                if self.device.type == "cuda"
                else 0,
                diversity=diversity(torch.stack([features[j]["global_feature"] for j in selected])),
                **segmentation_metrics(prediction, truth, data["spacing"], self.cfg["surface_tolerance"]),
            )
            if need_failure:
                row["initial_dice"] = dice(pred0.cpu(), truth)
                row["second_pass_gain"] = row["dice"] - row["initial_dice"]
                initial_correct = (pred0.cpu() > 0.5) == (truth > 0.5)
                final_correct = (prediction.cpu() > 0.5) == (truth > 0.5)
                row["repair_fraction"] = float((~initial_correct & final_correct).float().mean())
                row["harm_fraction"] = float((initial_correct & ~final_correct).float().mean())
                row["negative_second_pass"] = float(row["second_pass_gain"] < -1e-8)
            if regional_predictions is not None:
                row["predicted_region_repair"] = regional_predictions[selected, :, 0].cpu().tolist()
                row["predicted_region_harm"] = regional_predictions[selected, :, 1].cpu().tolist()
                row["region_grid"] = self.cfg["correction_grid"]
                row["gain_semantics"] = "correction coverage proxy; not predicted Dice or certified safety"
            row["scoring_seconds"] = time.perf_counter() - scoring_start
            if self.cfg["retrieval_diagnostics"]:
                diagnostic_start = time.perf_counter()
                single_utility = [
                    dice(self.segmenter(data["image"], [support]).cpu(), truth) for support in supports
                ]
                row["utility_at_k"] = float(np.mean(single_utility))
                row["selected_single_utilities"] = single_utility
                row["diagnostic_seconds"] = time.perf_counter() - diagnostic_start
            if self.cfg["save_predictions"]:
                prefix = digest(case.case_id)[:20]
                Image.fromarray((prediction[0].cpu().numpy() > 0.5).astype("uint8") * 255).save(
                    result_dir / f"{prefix}_mask.png"
                )
                if regional_predictions is not None:
                    from .visualization import save_correction_explanation

                    save_correction_explanation(
                        result_dir / f"{prefix}_explanation.png",
                        data,
                        pred0,
                        uncertainty,
                        prediction,
                        supports[0],
                        regional_predictions[selected],
                        self.cfg["correction_grid"],
                    )
                np.savez_compressed(
                    result_dir / f"{prefix}_maps.npz",
                    probability=prediction.cpu().numpy(),
                    uncertainty=uncertainty.cpu().numpy(),
                    initial=pred0.cpu().numpy(),
                )
            row["total_case_seconds"] = query_io_wait + time.perf_counter() - start
            rows.append(row)
        write_json(
            dict(
                signature=self.signature_for(ranker_kind(method)),
                manifest_hash=self.manifest_hash,
                identity_scope=self.cfg["identity_scope"],
                rows=rows,
            ),
            result_dir / "per_case.json",
        )
        summary = summarize(rows, self.cfg["identity_scope"])
        write_json(summary, result_dir / "summary.json")
        print(summary)
        return result_dir

    @torch.no_grad()
    def oracle(self, split):
        rows = []
        for case in tqdm(self.queries(split), desc="Oracle diagnostic (uses GT)"):
            data, query, candidates, features, rng, _, _ = self.context(
                case, random_candidates=True, need_failure=False
            )
            truth = load_case(case, self.cfg)["mask"].to(self.device)
            utilities = [dice(self.segmenter(data["image"], [c]), truth) for c in candidates]
            sims = [float(query["global_feature"] @ f["global_feature"]) for f in features]
            best, worst, knn = int(np.argmax(utilities)), int(np.argmin(utilities)), int(np.argmax(sims))
            random_id = int(rng.integers(len(candidates)))
            rows.append(
                dict(
                    case_id=case.case_id,
                    patient_id=case.patient_id,
                    candidate_count=len(candidates),
                    random=utilities[random_id],
                    random_expected=float(np.mean(utilities)),
                    knn=utilities[knn],
                    oracle=utilities[best],
                    worst=utilities[worst],
                    gap=utilities[best] - utilities[knn],
                    candidates=[c["case"]["case_id"] for c in candidates],
                    utilities=utilities,
                )
            )
        path = Path(self.cfg["output"]) / "results" / f"oracle_{split}_seed{self.cfg['seed']}"
        write_json(
            dict(
                rows=rows,
                signature=self.signature,
                uses_query_ground_truth=True,
                identity_scope=self.cfg["identity_scope"],
                protocol="single-support, shared random candidate pool",
            ),
            path / "oracle.json",
        )
        oracle_summary = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("random", "random_expected", "knn", "oracle", "worst", "gap")
        }
        write_json({"identity_scope": self.cfg["identity_scope"], **oracle_summary}, path / "summary.json")
        return path


def summarize(rows, identity_scope="patient"):
    result = {
        "n_cases": len(rows),
        "identity_scope": identity_scope,
        "hd95_undefined_cases": sum(r["hd95"] is None for r in rows),
    }
    result["n_patients" if identity_scope == "patient" else "n_images"] = len(
        {r["patient_id"] for r in rows}
    )
    for key in (
        "dice",
        "iou",
        "hd95",
        "nsd",
        "k",
        "diversity",
        "elapsed_seconds",
        "query_io_wait_seconds",
        "scoring_seconds",
        "total_case_seconds",
        "segmentation_seconds",
        "retrieval_seconds",
        "segmentation_calls",
        "support_evaluations",
        "peak_gpu_mb",
        "second_pass_gain",
        "utility_at_k",
        "repair_fraction",
        "harm_fraction",
        "negative_second_pass",
    ):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values))
    group_dice = [
        np.mean([r["dice"] for r in rows if r["patient_id"] == group])
        for group in sorted({r["patient_id"] for r in rows})
    ]
    result["patient_macro_dice" if identity_scope == "patient" else "image_macro_dice"] = float(
        np.mean(group_dice)
    )
    return result
