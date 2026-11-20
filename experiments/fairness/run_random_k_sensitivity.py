"""Full-bank Random K sensitivity. One process runs one K.

This is not the Top-12 fairness experiment. Each validation query draws from
its full legal training bank, the same 2,594-case domain at every K. There is
no candidate shortlist. For one query and repeat, one permutation of that bank
is drawn; K supports are its prefix, so K=2 is the prefix of K=8, K=16, and
K=64. The stream does not depend on K::

    SeedSequence(int(sha256(["fullbank-k-sensitivity-random-v1", 42, case_id, repeat])))
    Generator.permutation(n_legal)[:K]

Seed is 42. Repeats are 10. Official K values are 2, 8, 16, and 64.
Primary Dice/IoU/HD95 means are means of per-query means (n = n_queries).
The query-level std is the sample std (ddof=1) of those query means.
Within-query std is the sample std of the 10 draws and is averaged separately.

    python -m experiments.fairness.run_random_k_sensitivity \\
        --manifest "$MANIFEST" --out-base "$OUT_BASE/k_sensitivity" \\
        --k 2 --seed 42 --repeats 10 --device cuda
    python -m experiments.fairness.run_random_k_sensitivity --self-test
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
from far_icl.config import digest, file_digest, load_config, write_json

METHOD = "random_k_sensitivity"
PROTOCOL = "fullbank-k-sensitivity-random-v1"
ALLOWED_K = (2, 8, 16, 64)
SEED = 42
REPEATS = 10
REPO = Path(__file__).resolve().parents[2]
STD_DEFINITION = {
    "primary_mean": "unweighted mean of per-query means; n = n_queries, not n_queries*n_repeats",
    "sample_std_of_query_means": "ddof=1 across queries of each query's mean over its repeats",
    "within_query_sample_std": "ddof=1 across the 10 draws of one query",
    "mean_within_query_sample_std": "unweighted mean of the within-query sample stds; repeat variability, separate from the query-level std",
    "hd95": "means and stds use finite HD95 values only; an all-undefined query is excluded, not treated as zero",
}


def source_revision():
    revision = git_revision()
    if len(revision) != 40:
        raise RuntimeError(f"Full git revision required, got {revision!r}")
    return revision


def legal_indices(bank, case):
    """Same legal set as CaseBank.candidates, in bank order, with no shortlist."""
    return [
        i
        for i, entry in enumerate(bank.entries)
        if entry["case"]["patient_id"] != case.patient_id and entry["case"]["task"] == case.task
    ]


def random_permutation(eligible_ids, seed, case_id, repeat):
    key = [PROTOCOL, int(seed), str(case_id), int(repeat)]
    rng = np.random.default_rng(np.random.SeedSequence(int(digest(key), 16)))
    order = rng.permutation(len(eligible_ids))
    return [eligible_ids[int(index)] for index in order]


def prefix_supports(permutation, k):
    if not 1 <= k <= len(permutation):
        raise ValueError(f"K={k} does not fit the legal bank ({len(permutation)})")
    chosen = list(permutation[:k])
    if len(set(chosen)) != k:
        raise ValueError("random draw produced duplicate supports")
    return chosen


def sample_std(values):
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def mean(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def finite(rows, key):
    return [row[key] for row in rows if row.get(key) is not None]


def aggregate_random(rows):
    """Query rows already store the 10-draw mean and within-query sample std."""
    completed = [row for row in rows if row.get("status") == "ok"]
    result = {
        "n_rows": len(rows),
        "n_completed_queries": len(completed),
        "statistics_unit": "query",
        "n_repeats": REPEATS,
        "std_definition": STD_DEFINITION,
    }
    for key in ("dice", "iou", "hd95", "nsd"):
        values = finite(completed, key)
        result[f"{key}_mean_of_query_means"] = mean(values)
        result[f"{key}_sample_std_of_query_means"] = sample_std(values)
        result[f"{key}_n_query_means"] = len(values)
        result[f"{key}_mean_within_query_sample_std"] = mean(finite(completed, f"{key}_repeat_std"))
    return result


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in text


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
    if int(seed) != SEED:
        raise ValueError("Random K sensitivity is seed 42 only")
    if int(k) not in ALLOWED_K:
        raise ValueError(f"Random K must be one of {ALLOWED_K}")
    if int(repeats) != REPEATS:
        raise ValueError("Random K sensitivity uses 10 repeats")
    from far_icl.data import load_case, load_mask
    from far_icl.metrics import segmentation_metrics
    from far_icl.pipeline import Pipeline

    cfg = make_cfg(manifest, Path(out_base) / "_work", device, k, encoder_weights, segmenter_weights)
    pipe = Pipeline(cfg)
    queries = pipe.queries("val")
    bank_ids = [entry["case"]["case_id"] for entry in pipe.bank.entries]
    signature = digest(
        {
            "protocol": PROTOCOL,
            "k": int(k),
            "seed": SEED,
            "repeats": REPEATS,
            "manifest_hash": pipe.manifest_hash,
            "encoder": pipe.bank.encoder_sha256,
            "segmenter": pipe.segmenter_sha256,
            "revision": source_revision(),
            "numpy": np.__version__,
        }
    )
    target = (
        Path(out_base)
        / "k_sensitivity_fullbank"
        / METHOD
        / f"{METHOD}_k{int(k)}_val_seed{SEED}_r{REPEATS}_{signature[:12]}"
    )
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    rows = []
    stop = None
    eligible_counts = {}
    with torch.inference_mode():
        for case in queries:
            indices = legal_indices(pipe.bank, case)
            eligible_ids = [bank_ids[i] for i in indices]
            eligible_counts[case.case_id] = len(eligible_ids)
            if stop is not None:
                rows.append({"case_id": case.case_id, "status": "not_attempted_after_cuda_oom", "k": int(k)})
                continue
            if len(eligible_ids) < int(k):
                raise ValueError(f"{case.case_id}: {len(eligible_ids)} legal supports, refusing to lower K={k}")
            data = load_case(case, cfg, with_mask=False)
            repeats_out = []
            try:
                predictions = []
                draws = []
                for repeat in range(REPEATS):
                    chosen = prefix_supports(random_permutation(eligible_ids, SEED, case.case_id, repeat), int(k))
                    draws.append(chosen)
                    by_id = {bank_ids[i]: pipe.bank.entries[i] for i in indices}
                    predictions.append(pipe.segmenter(data["image"], [by_id[item] for item in chosen]))
                truth = load_mask(case, cfg, data["original_size"])
                for repeat, (chosen, prediction) in enumerate(zip(draws, predictions)):
                    metrics = segmentation_metrics(prediction, truth, data["spacing"], cfg["surface_tolerance"])
                    repeats_out.append({"repeat": repeat, "support_ids": chosen, "k": int(k), **metrics})
            except Exception as exc:
                if not is_cuda_oom(exc):
                    raise
                if pipe.device.type == "cuda":
                    torch.cuda.empty_cache()
                stop = str(exc)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "status": "cuda_oom",
                        "k": int(k),
                        "error": stop,
                        "dice": None,
                        "iou": None,
                        "hd95": None,
                    }
                )
                continue
            row = {
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "status": "ok",
                "method": METHOD,
                "split": "val",
                "k": int(k),
                "seed": SEED,
                "n_repeats": REPEATS,
                "pool_domain": "full_legal_train_bank",
                "repeats": repeats_out,
            }
            for key in ("dice", "iou", "hd95", "nsd"):
                values = finite(repeats_out, key)
                row[key] = mean(values)
                row[f"{key}_repeat_std"] = sample_std(values)
                row[f"{key}_finite_repeats"] = len(values)
            rows.append(row)
    status = "complete" if stop is None else "incomplete_cuda_oom"
    summary = aggregate_random(rows)
    summary.update(
        {
            "status": status,
            "requested_k": int(k),
            "protocol": PROTOCOL,
            "distinct_from": "resnet50-legal-train-top12-v1",
            "oom_error": stop,
        }
    )
    extra = {
        "protocol": PROTOCOL,
        "distinct_from": "Top-12 fairness uses a frozen 12-candidate pool. This run uses the full legal train bank at every K.",
        "retrieval_domain": "full_legal_train_bank",
        "shortlist_size": None,
        "nested_prefix": "K supports are the prefix of one K-independent permutation",
        "rng": PROTOCOL,
        "manifest_hash": pipe.manifest_hash,
        "manifest_path": cfg["manifest"],
        "query_order": [case.case_id for case in queries],
        "train_bank_ids": bank_ids,
        "eligible_counts": eligible_counts,
        "eligibility": {
            "split": "train",
            "same_task": True,
            "exclude_same_patient_id": True,
            "identity_scope": "image",
            "patient_id_rule": "image:{case_id}",
        },
        "encoder_parameter_sha256": pipe.bank.encoder_sha256,
        "encoder_weights_sha256": file_digest(cfg["encoder_weights"]) if cfg["encoder_weights"] else None,
        "segmenter_parameter_sha256": pipe.segmenter_sha256,
        "segmenter_weights_sha256": file_digest(cfg["segmenter_weights"]) if cfg["segmenter_weights"] else None,
        "requested_k": int(k),
        "seed": SEED,
        "repeats": REPEATS,
        "status": status,
        "std_definition": STD_DEFINITION,
    }
    run_cfg = {
        "method": METHOD,
        "k": int(k),
        "seed": SEED,
        "repeats": REPEATS,
        "device": device,
        "manifest": cfg["manifest"],
        "protocol": PROTOCOL,
        "candidate_n_is_not_a_shortlist": True,
    }

    def writer(staging):
        write_json(run_cfg, staging / "config.json")
        write_json(
            {
                "method": METHOD,
                "protocol": PROTOCOL,
                "manifest_hash": pipe.manifest_hash,
                "identity_scope": "image",
                "git_revision": source_revision(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "extra": extra,
            },
            staging / "provenance.json",
        )
        write_json({"statistics_unit": "query", "rows": rows}, staging / "per_case.json")
        write_json(summary, staging / "summary.json")
        (staging / "log.txt").write_text(
            f"method={METHOD}\nk={k}\nseed={SEED}\nrepeats={REPEATS}\nstatus={status}\n"
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
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--segmenter-weights")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.manifest is None or args.out_base is None or args.k is None:
        parser.error("--manifest, --out-base, and --k are required")
    if args.k not in ALLOWED_K or args.seed != SEED or args.repeats != REPEATS:
        parser.error(f"protocol is seed {SEED}, repeats {REPEATS}, K in {ALLOWED_K}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        _target, status = run(
            args.manifest,
            args.out_base,
            args.k,
            args.seed,
            args.repeats,
            args.device,
            args.encoder_weights,
            args.segmenter_weights,
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(f"cuda_oom before per-query loop k={args.k}: {exc}")
        return 3
    return 0 if status == "complete" else 3


def self_test():
    ids = [f"s{i:02d}" for i in range(80)]
    order = random_permutation(ids, 42, "q", 0)
    assert order == random_permutation(ids, 42, "q", 0)
    assert order != random_permutation(ids, 42, "q", 1)
    assert prefix_supports(order, 2) == order[:2]
    assert prefix_supports(order, 8)[:2] == prefix_supports(order, 2)
    assert prefix_supports(order, 16)[:8] == prefix_supports(order, 8)
    assert prefix_supports(order, 64)[:16] == prefix_supports(order, 16)
    assert len(set(prefix_supports(order, 64))) == 64
    repeats = [{"dice": 0.2, "iou": 0.1, "hd95": 1.0}, {"dice": 0.4, "iou": 0.3, "hd95": None}]
    assert abs(mean(finite(repeats, "dice")) - 0.3) < 1e-12
    assert sample_std(finite(repeats, "dice")) == float(np.std([0.2, 0.4], ddof=1))
    assert mean(finite(repeats, "hd95")) == 1.0
    rows = [
        {"status": "ok", "dice": 0.2, "iou": 0.1, "hd95": 1.0, "nsd": 1.0,
         "dice_repeat_std": 0.05, "iou_repeat_std": 0.02, "hd95_repeat_std": 0.1, "nsd_repeat_std": 0.0},
        {"status": "ok", "dice": 0.4, "iou": 0.3, "hd95": None, "nsd": 0.0,
         "dice_repeat_std": 0.07, "iou_repeat_std": 0.01, "hd95_repeat_std": None, "nsd_repeat_std": 0.0},
    ]
    summary = aggregate_random(rows)
    assert summary["dice_n_query_means"] == 2
    assert summary["hd95_n_query_means"] == 1
    assert abs(summary["dice_mean_of_query_means"] - 0.3) < 1e-12
    assert abs(summary["dice_sample_std_of_query_means"] - float(np.std([0.2, 0.4], ddof=1))) < 1e-12
    try:
        parse_args(["--k", "32", "--seed", "42", "--manifest", "m", "--out-base", "o"])
    except SystemExit:
        pass
    else:
        raise AssertionError("K=32 was accepted")
    assert is_cuda_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    print("PASS random-k-sensitivity self-test")


if __name__ == "__main__":
    raise SystemExit(main())
