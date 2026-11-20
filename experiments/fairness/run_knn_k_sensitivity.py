"""Full-bank KNN K sensitivity. One process runs one K.

Not the Top-12 fairness pool. The ordered support list is the stable cosine
ranking of the full legal training bank, the same domain at every K::

    CaseBank.candidates(case, resnet_vector, K, rng=None, stable=True)

K=2 is therefore the prefix of K=8, 16, and 64. Seed is 42. There is one
deterministic draw per query, so there is no within-query repeat std.
Official K values are 2, 8, 16, and 64.

    python -m experiments.fairness.run_knn_k_sensitivity \\
        --manifest "$MANIFEST" --out-base "$OUT_BASE/k_sensitivity" \\
        --k 8 --seed 42 --repeats 1 --device cuda
    python -m experiments.fairness.run_knn_k_sensitivity --self-test
"""

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.result_io import git_revision
from far_icl.bank import CaseBank
from far_icl.config import digest, file_digest, load_config, write_json

METHOD = "knn_k_sensitivity"
PROTOCOL = "fullbank-k-sensitivity-knn-v1"
ALLOWED_K = (2, 8, 16, 64)
SEED = 42
REPO = Path(__file__).resolve().parents[2]
STD_DEFINITION = {
    "primary_mean": "unweighted mean over validation queries; one deterministic prediction each",
    "sample_std_of_queries": "ddof=1 across queries; this is the query-level std",
    "within_query_repeat_std": "not applicable; KNN has one draw per query",
    "hd95": "finite values only; undefined distances are excluded, not set to zero",
}


def source_revision():
    revision = git_revision()
    if len(revision) != 40:
        raise RuntimeError(f"Full git revision required, got {revision!r}")
    return revision


def knn_ids(bank, case, vector, k):
    """Full-bank stable top-K. The returned ids are a prefix of the full ranking."""
    chosen = bank.candidates(case, vector, k, None, stable=True)
    ids = [entry["case"]["case_id"] for entry in chosen]
    if len(ids) != k or len(set(ids)) != k:
        raise ValueError(f"{case.case_id}: need {k} unique legal supports, found {len(set(ids))}")
    return ids


def full_rank_ids(bank, case, vector):
    n = sum(
        entry["case"]["patient_id"] != case.patient_id and entry["case"]["task"] == case.task
        for entry in bank.entries
    )
    return knn_ids(bank, case, vector, n)


def sample_std(values):
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def mean(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(rows):
    completed = [row for row in rows if row.get("status") == "ok"]
    result = {
        "n_rows": len(rows),
        "n_completed_queries": len(completed),
        "statistics_unit": "query",
        "n_repeats": 1,
        "std_definition": STD_DEFINITION,
    }
    for key in ("dice", "iou", "hd95", "nsd"):
        values = [row[key] for row in completed if row.get(key) is not None]
        result[f"{key}_mean"] = mean(values)
        result[f"{key}_sample_std_of_queries"] = sample_std(values)
        result[f"{key}_n"] = len(values)
    return result


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def make_cfg(manifest, output, device, k, encoder_weights, segmenter_weights):
    values = {
        "manifest": str(Path(manifest).resolve()),
        "output": str(Path(output).resolve()),
        "device": device,
        "identity_scope": "image",
        "k": int(k),
        "max_k": int(k),
        "initial_k": int(k),
        "candidate_n": int(k),
        "seed": SEED,
        "encoder": "resnet50",
        "segmenter": "universeg",
        "encoder_weights": str(Path(encoder_weights).resolve()) if encoder_weights else None,
        "segmenter_weights": str(Path(segmenter_weights).resolve()) if segmenter_weights else None,
        "uncertainty": "none",
        "save_predictions": False,
    }
    return load_config(REPO / "configs/isic.yaml", [f"{key}={json.dumps(value)}" for key, value in values.items()])


def publish(target, writer):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{os.getpid()}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        writer(staging)
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def run(manifest, out_base, k, seed, repeats, device, encoder_weights=None, segmenter_weights=None):
    if int(seed) != SEED or int(repeats) != 1 or int(k) not in ALLOWED_K:
        raise ValueError("KNN sensitivity is seed 42, one draw, K in (2, 8, 16, 64)")
    from far_icl.data import load_case, load_mask
    from far_icl.metrics import segmentation_metrics
    from far_icl.pipeline import Pipeline

    cfg = make_cfg(manifest, Path(out_base) / "_work", device, k, encoder_weights, segmenter_weights)
    pipe = Pipeline(cfg)
    queries = pipe.queries("val")
    signature = digest(
        {
            "protocol": PROTOCOL,
            "k": int(k),
            "seed": SEED,
            "manifest_hash": pipe.manifest_hash,
            "encoder": pipe.bank.encoder_sha256,
            "segmenter": pipe.segmenter_sha256,
            "revision": source_revision(),
        }
    )
    target = Path(out_base) / "k_sensitivity_fullbank" / METHOD / f"{METHOD}_k{int(k)}_val_seed{SEED}_{signature[:12]}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    rows = []
    stop = None
    with torch.inference_mode():
        for case in queries:
            if stop is not None:
                rows.append({"case_id": case.case_id, "status": "not_attempted_after_cuda_oom", "k": int(k)})
                continue
            data = load_case(case, cfg, with_mask=False)
            vector = pipe.encoder(data["rgb"][None].to(pipe.device))[0][0]
            try:
                support_ids = knn_ids(pipe.bank, case, vector, int(k))
                by_id = {entry["case"]["case_id"]: entry for entry in pipe.bank.entries}
                prediction = pipe.segmenter(data["image"], [by_id[item] for item in support_ids])
            except Exception as exc:
                if not is_cuda_oom(exc):
                    raise
                if pipe.device.type == "cuda":
                    torch.cuda.empty_cache()
                stop = str(exc)
                rows.append({"case_id": case.case_id, "status": "cuda_oom", "k": int(k), "error": stop,
                             "dice": None, "iou": None, "hd95": None})
                continue
            truth = load_mask(case, cfg, data["original_size"])
            metrics = segmentation_metrics(prediction, truth, data["spacing"], cfg["surface_tolerance"])
            rows.append(
                {
                    "case_id": case.case_id,
                    "patient_id": case.patient_id,
                    "status": "ok",
                    "method": METHOD,
                    "split": "val",
                    "k": int(k),
                    "seed": SEED,
                    "pool_domain": "full_legal_train_bank",
                    "tie_break": "stable argsort descending",
                    "support_ids": support_ids,
                    **metrics,
                }
            )
    status = "complete" if stop is None else "incomplete_cuda_oom"
    summary = aggregate(rows)
    summary.update({"status": status, "requested_k": int(k), "protocol": PROTOCOL,
                    "distinct_from": "resnet50-legal-train-top12-v1", "oom_error": stop})
    extra = {
        "protocol": PROTOCOL,
        "distinct_from": "Top-12 fairness fixes N=12. This ranking uses the full legal train bank.",
        "retrieval_domain": "full_legal_train_bank",
        "shortlist_size": None,
        "tie_break": "CaseBank.candidates stable=True",
        "manifest_hash": pipe.manifest_hash,
        "manifest_path": cfg["manifest"],
        "query_order": [case.case_id for case in queries],
        "train_bank_ids": [entry["case"]["case_id"] for entry in pipe.bank.entries],
        "encoder_parameter_sha256": pipe.bank.encoder_sha256,
        "encoder_weights_sha256": file_digest(cfg["encoder_weights"]) if cfg["encoder_weights"] else None,
        "segmenter_parameter_sha256": pipe.segmenter_sha256,
        "segmenter_weights_sha256": file_digest(cfg["segmenter_weights"]) if cfg["segmenter_weights"] else None,
        "requested_k": int(k),
        "seed": SEED,
        "repeats": 1,
        "status": status,
        "std_definition": STD_DEFINITION,
    }

    def writer(staging):
        write_json(
            {"method": METHOD, "k": int(k), "seed": SEED, "repeats": 1, "protocol": PROTOCOL,
             "candidate_n_is_not_a_shortlist": True, "manifest": cfg["manifest"], "device": device},
            staging / "config.json",
        )
        write_json(
            {"method": METHOD, "protocol": PROTOCOL, "manifest_hash": pipe.manifest_hash,
             "identity_scope": "image", "git_revision": source_revision(),
             "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
             "extra": extra},
            staging / "provenance.json",
        )
        write_json({"statistics_unit": "query", "rows": rows}, staging / "per_case.json")
        write_json(summary, staging / "summary.json")
        (staging / "log.txt").write_text(f"method={METHOD}\nk={k}\nseed={SEED}\nstatus={status}\n")

    publish(target, writer)
    print(summary)
    return target, status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-base", type=Path)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--segmenter-weights")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.manifest is None or args.out_base is None or args.k is None:
        parser.error("--manifest, --out-base, and --k are required")
    if args.k not in ALLOWED_K or args.seed != SEED or args.repeats != 1:
        parser.error(f"protocol is seed {SEED}, repeats 1, K in {ALLOWED_K}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    _target, status = run(
        args.manifest, args.out_base, args.k, args.seed, args.repeats, args.device,
        args.encoder_weights, args.segmenter_weights,
    )
    return 0 if status == "complete" else 3


def self_test():
    import torch.nn.functional as F

    query = type("Case", (), {"case_id": "q", "patient_id": "patient-q", "task": "lesion"})()
    bank = object.__new__(CaseBank)
    bank.entries = [
        {"case": {"case_id": "q", "patient_id": "patient-q", "task": "lesion"}},
        {"case": {"case_id": "same", "patient_id": "patient-q", "task": "lesion"}},
        {"case": {"case_id": "other-task", "patient_id": "p", "task": "disc"}},
        *[{"case": {"case_id": str(i), "patient_id": str(i), "task": "lesion"}} for i in range(6)],
    ]
    bank.vectors = F.normalize(torch.tensor([[1.0, 0.0]] * 3 + [[float(i), 1.0] for i in range(6)], dtype=torch.float32))
    vector = torch.tensor([1.0, 0.0])
    full = full_rank_ids(bank, query, vector)
    assert knn_ids(bank, query, vector, 2) == full[:2]
    assert knn_ids(bank, query, vector, 4) == full[:4]
    assert "q" not in full and "same" not in full and "other-task" not in full
    values = aggregate([
        {"status": "ok", "dice": 0.2, "iou": 0.1, "hd95": None, "nsd": 0.0},
        {"status": "ok", "dice": 0.6, "iou": 0.4, "hd95": 2.0, "nsd": 1.0},
    ])
    assert values["dice_n"] == 2 and values["hd95_n"] == 1
    assert values["n_repeats"] == 1
    try:
        parse_args(["--k", "32", "--seed", "43", "--manifest", "m", "--out-base", "o"])
    except SystemExit:
        pass
    else:
        raise AssertionError("non-protocol KNN settings were accepted")
    print("PASS knn-k-sensitivity self-test")


if __name__ == "__main__":
    raise SystemExit(main())
