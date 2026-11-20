"""Full-bank MMR K sensitivity. One process runs one K.

Not the Top-12 fairness pool and not a candidate_n shortlist. MMR lambda is
0.5 on every legal training case, using far_icl.baselines.mmr_indices:
score = 0.5 * query cosine - 0.5 * max similarity to an already chosen case.
The greedy list is prefix-stable, so K=2 is the prefix of K=8 and K=16.
K=64 is intentionally rejected. Seed is 42. One deterministic draw per query.

    python -m experiments.fairness.run_mmr_k_sensitivity \\
        --manifest "$MANIFEST" --out-base "$OUT_BASE/k_sensitivity" \\
        --k 2 --seed 42 --repeats 1 --device cuda
    python -m experiments.fairness.run_mmr_k_sensitivity --self-test
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
from far_icl.baselines import mmr_indices
from far_icl.config import digest, file_digest, load_config, write_json

METHOD = "mmr_k_sensitivity"
PROTOCOL = "fullbank-k-sensitivity-mmr-v1"
ALLOWED_K = (2, 8, 16)
LAMBDA = 0.5
SEED = 42
REPO = Path(__file__).resolve().parents[2]
STD_DEFINITION = {
    "primary_mean": "unweighted mean over validation queries; one deterministic prediction each",
    "sample_std_of_queries": "ddof=1 across queries; this is the query-level std",
    "within_query_repeat_std": "not applicable; MMR has one draw per query",
    "hd95": "finite values only; undefined distances are excluded, not set to zero",
    "lambda": "0.5, far_icl.baselines.mmr_indices relevance_weight",
    "domain": "full legal train bank; a Top-12 or other shortlist is a different experiment",
}


def source_revision():
    revision = git_revision()
    if len(revision) != 40:
        raise RuntimeError(f"Full git revision required, got {revision!r}")
    return revision


def legal_indices(bank, case):
    return [
        i
        for i, entry in enumerate(bank.entries)
        if entry["case"]["patient_id"] != case.patient_id and entry["case"]["task"] == case.task
    ]


def mmr_support_ids(ids, vectors, query, k, relevance_weight=LAMBDA):
    if vectors.shape[0] != len(ids):
        raise ValueError("MMR must score the full legal bank, not a shortlist")
    if k > len(ids):
        raise ValueError(f"K={k} exceeds {len(ids)} legal supports; refusing to lower K")
    local = mmr_indices(query, vectors, k, relevance_weight)
    chosen = [ids[i] for i in local]
    if len(chosen) != k or len(set(chosen)) != k:
        raise ValueError("MMR returned duplicate or short supports")
    return chosen


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
        "mmr_lambda": LAMBDA,
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
        raise ValueError("MMR sensitivity is seed 42, one draw, K in (2, 8, 16); K=64 is omitted")
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
            "lambda": LAMBDA,
            "seed": SEED,
            "manifest_hash": pipe.manifest_hash,
            "encoder": pipe.bank.encoder_sha256,
            "segmenter": pipe.segmenter_sha256,
            "revision": source_revision(),
        }
    )
    target = (
        Path(out_base) / "k_sensitivity_fullbank" / METHOD
        / f"{METHOD}_k{int(k)}_val_seed{SEED}_lambda{LAMBDA:g}_{signature[:12]}"
    )
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    rows = []
    stop = None
    with torch.inference_mode():
        for case in queries:
            if stop is not None:
                rows.append({"case_id": case.case_id, "status": "not_attempted_after_cuda_oom", "k": int(k)})
                continue
            indices = legal_indices(pipe.bank, case)
            ids = [pipe.bank.entries[i]["case"]["case_id"] for i in indices]
            data = load_case(case, cfg, with_mask=False)
            vector = pipe.encoder(data["rgb"][None].to(pipe.device))[0][0]
            vectors = pipe.bank.vectors[indices]
            try:
                support_ids = mmr_support_ids(ids, vectors, vector, int(k), LAMBDA)
                by_id = {pipe.bank.entries[i]["case"]["case_id"]: pipe.bank.entries[i] for i in indices}
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
                    "mmr_lambda": LAMBDA,
                    "pool_domain": "full_legal_train_bank",
                    "support_ids": support_ids,
                    **metrics,
                }
            )
    status = "complete" if stop is None else "incomplete_cuda_oom"
    summary = aggregate(rows)
    summary.update({"status": status, "requested_k": int(k), "protocol": PROTOCOL, "mmr_lambda": LAMBDA,
                    "distinct_from": "top12-mmr-and-shortlist-baseline", "oom_error": stop})
    extra = {
        "protocol": PROTOCOL,
        "distinct_from": "Historical Top-12 or candidate_n-limited MMR is a different domain. This MMR sees the full legal train bank.",
        "retrieval_domain": "full_legal_train_bank",
        "shortlist_size": None,
        "mmr_lambda": LAMBDA,
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
            {"method": METHOD, "k": int(k), "seed": SEED, "repeats": 1, "mmr_lambda": LAMBDA,
             "protocol": PROTOCOL, "candidate_n_is_not_a_shortlist": True, "manifest": cfg["manifest"],
             "device": device},
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
        (staging / "log.txt").write_text(
            f"method={METHOD}\nk={k}\nlambda={LAMBDA}\nseed={SEED}\nstatus={status}\n"
        )

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
        parser.error("protocol is seed 42, repeats 1, K in (2, 8, 16); K=64 is omitted")
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
    query = torch.tensor([0.8, 0.6])
    vectors = torch.tensor(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    ids = ["near-a", "near-b", "diverse", "opposite"]
    assert mmr_support_ids(ids, vectors, query, 2) == ["near-a", "diverse"]
    assert mmr_support_ids(ids, vectors, query, 2) == mmr_support_ids(ids, vectors, query, 3)[:2]
    shortlist = mmr_indices(query, vectors[:2], 2, LAMBDA)
    assert [ids[i] for i in shortlist] == ["near-a", "near-b"]
    summary = aggregate([
        {"status": "ok", "dice": 0.5, "iou": 0.4, "hd95": 3.0, "nsd": 0.2},
        {"status": "cuda_oom", "dice": None, "iou": None, "hd95": None, "nsd": None},
    ])
    assert summary["n_completed_queries"] == 1
    assert summary["dice_n"] == 1
    assert summary["dice_sample_std_of_queries"] is None
    try:
        parse_args(["--k", "64", "--seed", "42", "--manifest", "m", "--out-base", "o"])
    except SystemExit:
        pass
    else:
        raise AssertionError("MMR K=64 was accepted")
    print("PASS mmr-k-sensitivity self-test")


if __name__ == "__main__":
    raise SystemExit(main())
