"""Evaluate frozen UniverSeg retrieval baselines with a shared training bank."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

METHODS = ("random", "knn", "mmr", "dual_similarity_adapted")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-base", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--identity-scope", choices=("image", "patient"), default="image")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--candidate-n", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--initial-k", type=int, default=2, help="KNN supports for the dual mask prior")
    parser.add_argument("--segmenter-weights")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.candidate_n is None:
        args.candidate_n = max(64, args.k)
    if not 1 <= args.k <= args.candidate_n:
        parser.error("require 1 <= k <= candidate-n")
    if min(args.random_repeats, args.initial_k) < 1 or not 0 <= args.mmr_lambda <= 1:
        parser.error("repeats and initial-k must be positive; mmr-lambda must be in [0,1]")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32-1")
    return args


def configuration(args):
    from far_icl.config import load_config

    values = dict(
        manifest=str(args.manifest.resolve()),
        output=str(args.out_base.resolve() / "baselines"),
        device=args.device,
        identity_scope=args.identity_scope,
        k=args.k,
        max_k=args.k,
        initial_k=min(args.initial_k, args.candidate_n),
        candidate_n=args.candidate_n,
        seed=args.seed,
        encoder_weights=args.encoder_weights,
        segmenter_weights=args.segmenter_weights,
        uncertainty="none",
        save_predictions=False,
    )
    return load_config(
        Path(__file__).resolve().parents[2] / "configs/isic.yaml",
        [f"{key}={json.dumps(value)}" for key, value in values.items()],
    )


def evaluate(pipe, cases, method, repeat, args):
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    from far_icl.baselines import dual_similarity_indices, mmr_indices
    from far_icl.config import digest
    from far_icl.data import load_case, load_mask
    from far_icl.features import shape_descriptor
    from far_icl.metrics import diversity, segmentation_metrics

    rows = []
    with torch.inference_mode():
        for case in tqdm(cases, desc=f"{method}/{args.split}"):
            pipe.segmenter.reset_stats()
            if pipe.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(pipe.device)
            pipe.segmenter.sync()
            start = time.perf_counter()
            data = load_case(case, pipe.cfg, with_mask=False)
            # Repeat zero preserves the original seed/query stream for uniform random retrieval.
            rng_key = [args.seed, case.case_id] + ([repeat] if repeat else [])
            rng = np.random.default_rng(int(digest(rng_key)[:8], 16)) if method == "random" else None
            vector = None
            if method != "random":
                vector = pipe.encoder(data["rgb"][None].to(pipe.device))[0][0]
            pool = pipe.bank.candidates(case, vector, args.candidate_n, rng, stable=True)
            if len(pool) < args.k:
                raise ValueError(f"{case.case_id}: requested K={args.k}, found {len(pool)} eligible supports")
            chosen = list(range(args.k))
            detail = {}
            if method in ("mmr", "dual_similarity_adapted"):
                vectors = torch.stack([entry["features"]["global_feature"] for entry in pool]).to(pipe.device)
                if method == "mmr":
                    chosen = mmr_indices(vector, vectors, args.k, args.mmr_lambda)
                if method == "dual_similarity_adapted":
                    initial = pool[: pipe.cfg["initial_k"]]
                    prior = shape_descriptor(pipe.segmenter(data["image"], initial))
                    shapes = torch.stack([entry["features"]["shape"] for entry in pool]).to(pipe.device)
                    chosen = dual_similarity_indices(vector, vectors, prior, shapes, args.k)
                    detail["initial_support_ids"] = [entry["case"]["case_id"] for entry in initial]

            supports = [pool[index] for index in chosen]
            prediction = pipe.segmenter(data["image"], supports)
            pipe.segmenter.sync()
            elapsed = time.perf_counter() - start
            truth = load_mask(case, pipe.cfg, data["original_size"])
            rows.append(
                dict(
                    case_id=case.case_id,
                    patient_id=case.patient_id,
                    task=case.task,
                    domain=case.domain,
                    method=method,
                    split=args.split,
                    k=len(supports),
                    pool_size=len(pool),
                    support_ids=[entry["case"]["case_id"] for entry in supports],
                    support_patient_ids=[entry["case"]["patient_id"] for entry in supports],
                    diversity=diversity(
                        torch.stack([entry["features"]["global_feature"] for entry in supports])
                    ),
                    elapsed_seconds=elapsed,
                    segmentation_calls=pipe.segmenter.calls,
                    segmentation_seconds=pipe.segmenter.seconds,
                    support_evaluations=pipe.segmenter.support_evaluations,
                    peak_gpu_mb=torch.cuda.max_memory_allocated(pipe.device) / 1024**2
                    if pipe.device.type == "cuda"
                    else 0,
                    **detail,
                    **segmentation_metrics(prediction, truth, data["spacing"], pipe.cfg["surface_tolerance"]),
                )
            )
    return rows


def run(args):
    from experiments.result_io import git_revision, save_results
    from far_icl.config import digest
    from far_icl.pipeline import Pipeline

    cfg = configuration(args)
    pipe = Pipeline(cfg)
    cases = pipe.queries(args.split)
    root = Path(cfg["output"])
    result_root = root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    revision = git_revision()
    provenance = dict(
        bank_signature=pipe.bank.signature,
        bank_cache=str(pipe.bank.root),
        encoder_sha256=pipe.bank.encoder_sha256,
        segmenter_sha256=pipe.segmenter_sha256,
        query_ids=[case.case_id for case in cases],
    )
    targets = []
    for method in METHODS if args.method == "all" else (args.method,):
        for repeat in range(args.random_repeats if method == "random" else 1):
            settings = cfg | dict(
                method=method,
                split=args.split,
                random_repeat=repeat if method == "random" else None,
                mmr_lambda=args.mmr_lambda if method == "mmr" else None,
            )
            suffix = f"_r{repeat}" if method == "random" else ""
            if method == "mmr":
                suffix = f"_lambda{args.mmr_lambda:g}"
            label = f"{method}_k{args.k}_n{args.candidate_n}_{args.split}_seed{args.seed}{suffix}"
            signature = digest([pipe.signature, settings, revision])
            target = result_root / f"{label}_{signature[:12]}"
            targets.append(target)
            if target.exists():
                names = ["config.json", "provenance.json", "per_case.json", "summary.json"]
                if all((target / name).is_file() for name in names):
                    saved_config, metadata, predictions, summary = [
                        json.loads((target / name).read_text()) for name in names
                    ]
                    if (
                        saved_config == settings
                        and metadata["extra"].get("run_signature") == signature
                        and metadata["git_revision"] == revision
                        and summary["n_cases"] == len(cases)
                        and [r["case_id"] for r in predictions["rows"]] == provenance["query_ids"]
                    ):
                        print(f"Reuse completed run: {target}")
                        continue
                raise FileExistsError(f"Incomplete or mismatched run: {target}. Choose another OUT_BASE.")

            rows = evaluate(pipe, cases, method, repeat, args)
            with TemporaryDirectory(dir=root) as staging:
                save_results(
                    settings,
                    pipe.manifest_hash,
                    method,
                    args.split,
                    rows,
                    extra=provenance | {"run_signature": signature},
                    target=Path(staging),
                    revision=revision,
                    protocol="Frozen UniverSeg same-backbone retrieval baselines",
                )
                Path(staging).rename(target)
            print(target)
    return targets


if __name__ == "__main__":
    run(parse_args())
