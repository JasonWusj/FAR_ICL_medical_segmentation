"""Inference and offline supervision share identical label-free retrieval features."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .bank import CaseBank
from .config import atomic_save, digest, file_digest, read_tensor, seed_all, write_json
from .data import load_case, manifest_signature, read_manifest
from .features import ImageEncoder, describe, pair_features, set_features
from .metrics import dice, diversity, segmentation_metrics
from .retrieval import UtilityRanker, select
from .segmentation import Segmenter, estimate_uncertainty

LEARNED = {"utility", "mmr", "set", "adaptive", "adaptive_learned"}


class Pipeline:
    def __init__(self, cfg, bank_only=False):
        self.cfg = cfg
        seed_all(cfg["seed"], cfg["deterministic"])
        self.device = torch.device(cfg["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly use --set device=cpu if intended")
        self.cases = read_manifest(cfg["manifest"])
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
    def context(self, case, random_candidates=False, need_failure=True):
        rng = self.rng_for(case)
        data = load_case(case, self.cfg, with_mask=False)
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
        return data, query, candidates, features, rng, pred, uncertainty

    @torch.no_grad()
    def generate(self, kind, split, force=False):
        if split not in {"train", "val"}:
            raise ValueError("Test labels must never generate training/validation supervision")
        target = self.artifacts / kind / split
        target.mkdir(parents=True, exist_ok=True)
        for case in tqdm(self.queries(split), desc=f"Generate {kind}/{split}"):
            output = target / f"{digest(case.case_id)}.pt"
            if output.exists() and not force:
                continue
            data, query, candidates, features, rng, _, _ = self.context(
                case, need_failure=self.cfg["feature_mode"] != "image"
            )
            truth = load_case(case, self.cfg)["mask"].to(self.device)
            groups = []
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
            dict(signature=self.signature, kind=kind, split=split, query_count=len(self.queries(split))),
            target / "metadata.json",
        )
        return target

    def load_ranker(self, path, method):
        if method not in LEARNED:
            return None
        if not path:
            raise ValueError(f"{method} requires --checkpoint")
        checkpoint = read_tensor(path)
        expected = "marginal" if method in {"set", "adaptive_learned"} else "utility"
        if checkpoint["kind"] != expected or checkpoint["signature"] != self.signature:
            raise ValueError("Checkpoint type/provenance mismatch; use matching config and bank")
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
                signature=self.signature,
                manifest_hash=self.manifest_hash,
                git_revision=revision.stdout.strip() or "uncommitted",
                python=platform.python_version(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                checkpoint_sha256=file_digest(checkpoint) if method in LEARNED else None,
            ),
            result_dir / "provenance.json",
        )
        rows = []
        for case in tqdm(self.queries(split), desc=name):
            self.segmenter.reset_stats()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.segmenter.sync()
            start = time.perf_counter()
            need_failure = method in {"shape", "roles", "adaptive"} or (
                method not in {"random", "knn"} and self.cfg["feature_mode"] != "image"
            )
            data, query, candidates, features, rng, pred0, uncertainty = self.context(
                case, random_candidates=method == "random", need_failure=need_failure
            )
            gains = []
            if method in {"random", "knn"}:
                selected = list(range(min(self.cfg["k"], len(candidates))))
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
            truth = load_case(case, self.cfg)["mask"]
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
            if self.cfg["retrieval_diagnostics"]:
                diagnostic_start = time.perf_counter()
                single_utility = [
                    dice(self.segmenter(data["image"], [support]).cpu(), truth) for support in supports
                ]
                row["utility_at_k"] = float(np.mean(single_utility))
                row["selected_single_utilities"] = single_utility
                row["diagnostic_seconds"] = time.perf_counter() - diagnostic_start
            rows.append(row)
            if self.cfg["save_predictions"]:
                prefix = digest(case.case_id)[:20]
                Image.fromarray((prediction[0].cpu().numpy() > 0.5).astype("uint8") * 255).save(
                    result_dir / f"{prefix}_mask.png"
                )
                np.savez_compressed(
                    result_dir / f"{prefix}_maps.npz",
                    probability=prediction.cpu().numpy(),
                    uncertainty=uncertainty.cpu().numpy(),
                    initial=pred0.cpu().numpy(),
                )
        write_json(
            dict(signature=self.signature, manifest_hash=self.manifest_hash, rows=rows),
            result_dir / "per_case.json",
        )
        summary = summarize(rows)
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
                protocol="single-support, shared random candidate pool",
            ),
            path / "oracle.json",
        )
        write_json(
            {
                key: float(np.mean([r[key] for r in rows]))
                for key in ("random", "random_expected", "knn", "oracle", "worst", "gap")
            },
            path / "summary.json",
        )
        return path


def summarize(rows):
    result = {
        "n_cases": len(rows),
        "n_patients": len({r["patient_id"] for r in rows}),
        "hd95_undefined_cases": sum(r["hd95"] is None for r in rows),
    }
    for key in (
        "dice",
        "iou",
        "hd95",
        "nsd",
        "k",
        "diversity",
        "elapsed_seconds",
        "segmentation_seconds",
        "retrieval_seconds",
        "segmentation_calls",
        "support_evaluations",
        "peak_gpu_mb",
        "second_pass_gain",
        "utility_at_k",
    ):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values))
    patient_dice = [
        np.mean([r["dice"] for r in rows if r["patient_id"] == patient])
        for patient in sorted({r["patient_id"] for r in rows})
    ]
    result["patient_macro_dice"] = float(np.mean(patient_dice))
    return result
