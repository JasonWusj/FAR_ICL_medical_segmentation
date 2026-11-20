"""IRES-S3-Top12 Adapted on the frozen ResNet50 Top-12 pool.

Both encoder rankings are restricted to the 12 ids saved by
``run_random_top12 prepare`` before each encoder's stable top-2 is taken.
The S3 comparison is the repository function ``stability_s3`` and the same
tie rule (ResNet50 wins ties). There is no IRES/P2R training. This file never
calls ``CaseBank.candidates`` and never rebuilds the pool.

DINOv2 pin, checked when the CLI runs::

    source 7764ea0f912e53c92e82eb78a2a1631e92725fc8
    weights b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9

Commands::

    python -m experiments.fairness.run_ires_s3_top12 evaluate \\
        --pool "$OUT/top12/shared_pool/resnet50_legal_train_top12_val.json" \\
        --out-base "$OUT/top12" --seed 42 --device cuda \\
        --dinov2-repo "$DINOV2_REPO" --dinov2-weights "$DINOV2_WEIGHTS"
    python -m experiments.fairness.run_ires_s3_top12 evaluate \\
        --pool "$OUT/top12/shared_pool/resnet50_legal_train_top12_val.json" \\
        --out-base "$OUT/top12" --seed 43 --device cuda \\
        --dinov2-repo "$DINOV2_REPO" --dinov2-weights "$DINOV2_WEIGHTS"
    python -m experiments.fairness.run_ires_s3_top12 self-test
"""

import argparse
import subprocess
from pathlib import Path

import torch

from experiments.fairness.run_random_top12 import (
    K,
    SEEDS,
    SPLIT,
    bind_pool,
    digest,
    load_pool,
    make_cfg,
    open_pipeline,
    result_dir,
    source_revision,
    write_outputs,
    complete_match,
)
from experiments.frozen_adaptations import stability_s3
from far_icl.config import file_digest

OFFICIAL_DINOV2_SOURCE = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
OFFICIAL_DINOV2_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
METHOD = "ires_s3_top12_adapted"
PROTOCOL = (
    "IRES S3 Top-12 Adapted: stable top-2 of frozen ResNet50 and DINOv2 inside the "
    "saved ResNet50 legal-train pool, then the repository S3 rule. No P2R training."
)


def check_dinov2_pin(weights, repo, enforce):
    sha = file_digest(weights)
    revision = None
    if Path(repo).joinpath(".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        revision = proc.stdout.strip() or None
    if enforce and sha != OFFICIAL_DINOV2_SHA:
        raise ValueError(f"DINOv2 weights sha {sha} != official {OFFICIAL_DINOV2_SHA}")
    if enforce and revision != OFFICIAL_DINOV2_SOURCE:
        raise ValueError(f"DINOv2 source {revision!r} != official {OFFICIAL_DINOV2_SOURCE}")
    return sha, revision


def stable_rank(ids, vectors, query):
    """Stable cosine order. The returned ids are a permutation of ``ids`` only."""
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError("DINO/ResNet scores must align with the saved Top-12")
    scores = vectors.to(query.device) @ query
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = [ids[int(index)] for index in order.tolist()]
    if len(ranked) != len(set(ranked)) or set(ranked) != set(ids):
        raise RuntimeError("ranking left the saved Top-12")
    return ranked


def choose_encoder(resnet_score, dino_score):
    """Exact frozen_adaptations tie: ResNet50 wins when the S3 scores are equal."""
    return "resnet50" if resnet_score >= dino_score else "dinov2"


def evaluate_ires(
    pool_path,
    out_base,
    seed,
    device,
    dinov2_repo,
    dinov2_weights,
    encoder_weights=None,
    segmenter_weights=None,
    enforce_pin=True,
):
    if int(seed) not in SEEDS:
        raise ValueError("IRES-S3-Top12 official seeds are only 42 and 43")
    weight_sha, source_sha = check_dinov2_pin(dinov2_weights, dinov2_repo, enforce_pin)
    doc = load_pool(pool_path)
    stored = doc["retrieval_config"]
    cfg = make_cfg(
        stored["manifest"],
        Path(out_base) / "_work",
        device,
        encoder_weights if encoder_weights is not None else stored["encoder_weights"],
        segmenter_weights if segmenter_weights is not None else stored["segmenter_weights"],
        seed,
    )
    pipe = open_pipeline(cfg)
    bind_pool(doc, pipe, cfg)
    from far_icl.bank import CaseBank
    from far_icl.features import ImageEncoder

    alternate = dict(pipe.cfg)
    alternate["encoder"] = "dinov2"
    alternate["encoder_repo"] = str(Path(dinov2_repo).resolve())
    alternate["encoder_weights"] = str(Path(dinov2_weights).resolve())
    dino_encoder = ImageEncoder(alternate).to(pipe.device).eval()
    dino_bank = CaseBank(pipe.cases, alternate, dino_encoder, pipe.manifest_hash)
    resnet_by = {entry["case"]["case_id"]: entry for entry in pipe.bank.entries}
    dino_by = {entry["case"]["case_id"]: entry for entry in dino_bank.entries}
    signature = digest(
        {
            "method": METHOD,
            "seed": int(seed),
            "pool_hash": doc["pool_hash"],
            "retrieval_config_hash": doc["retrieval_config_hash"],
            "dinov2_weights_sha256": weight_sha,
            "dinov2_source": source_sha,
            "revision": source_revision(),
        }
    )
    tag = f"{METHOD}_k{K}_n12_{SPLIT}_seed{int(seed)}"
    target = result_dir(out_base, METHOD, int(seed), signature, tag)
    if complete_match(target, signature, doc["query_order"]):
        print(f"Reuse completed run: {target}")
        return target
    from far_icl.data import load_case, load_mask
    from far_icl.metrics import diversity, segmentation_metrics

    cases = {case.case_id: case for case in pipe.queries(SPLIT)}
    rows = []
    with torch.inference_mode():
        for query_id in doc["query_order"]:
            case = cases[query_id]
            saved = list(doc["pools"][query_id])
            missing = [item for item in saved if item not in resnet_by or item not in dino_by]
            if missing:
                raise RuntimeError(f"{query_id}: saved ids missing from the frozen banks: {missing}")
            pipe.segmenter.reset_stats()
            if pipe.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(pipe.device)
            pipe.segmenter.sync()
            start = __import__("time").perf_counter()
            data = load_case(case, cfg, with_mask=False)
            query_resnet = pipe.encoder(data["rgb"][None].to(pipe.device))[0][0]
            query_dino = dino_encoder(data["rgb"][None].to(pipe.device))[0][0]
            resnet_vectors = torch.stack([resnet_by[item]["features"]["global_feature"] for item in saved])
            dino_vectors = torch.stack([dino_by[item]["features"]["global_feature"] for item in saved])
            ranked_resnet = stable_rank(saved, resnet_vectors, query_resnet)
            if ranked_resnet != saved:
                raise RuntimeError("Saved Top-12 is not the stable ResNet order; refusing to regenerate")
            ranked_dino = stable_rank(saved, dino_vectors, query_dino)
            resnet_ids = ranked_resnet[:K]
            dino_ids = ranked_dino[:K]
            if not set(resnet_ids) <= set(saved) or not set(dino_ids) <= set(saved):
                raise RuntimeError("top-2 left the saved pool")
            rng = pipe.rng_for(case)
            torch.manual_seed(int(rng.integers(0, 2**31)))
            resnet_prediction = pipe.segmenter(data["image"], [resnet_by[item] for item in resnet_ids])
            torch.manual_seed(int(rng.integers(0, 2**31)))
            dino_prediction = pipe.segmenter(data["image"], [resnet_by[item] for item in dino_ids])
            resnet_s3 = stability_s3(resnet_prediction)
            dino_s3 = stability_s3(dino_prediction)
            chosen_encoder = choose_encoder(resnet_s3, dino_s3)
            if chosen_encoder == "resnet50":
                chosen_ids, prediction = resnet_ids, resnet_prediction
                feature_rows = resnet_by
            else:
                chosen_ids, prediction = dino_ids, dino_prediction
                feature_rows = dino_by
            pipe.segmenter.sync()
            elapsed = __import__("time").perf_counter() - start
            truth = load_mask(case, cfg, data["original_size"])
            metrics = segmentation_metrics(prediction, truth, data["spacing"], cfg["surface_tolerance"])
            features = torch.stack([feature_rows[item]["features"]["global_feature"] for item in chosen_ids])
            rows.append(
                {
                    "case_id": query_id,
                    "patient_id": case.patient_id,
                    "task": case.task,
                    "domain": case.domain,
                    "method": METHOD,
                    "adapted": True,
                    "split": SPLIT,
                    "k": K,
                    "pool_size": len(saved),
                    "seed": int(seed),
                    "pool_hash": doc["pool_hash"],
                    "support_ids": chosen_ids,
                    "support_patient_ids": [resnet_by[item]["case"]["patient_id"] for item in chosen_ids],
                    "resnet_support_ids": resnet_ids,
                    "dino_support_ids": dino_ids,
                    "s3_resnet50": resnet_s3,
                    "s3_dinov2": dino_s3,
                    "chosen_encoder": chosen_encoder,
                    "diversity": diversity(features),
                    "elapsed_seconds": elapsed,
                    "segmentation_calls": pipe.segmenter.calls,
                    "segmentation_seconds": pipe.segmenter.seconds,
                    "support_evaluations": pipe.segmenter.support_evaluations,
                    "peak_gpu_mb": torch.cuda.max_memory_allocated(pipe.device) / 1024**2
                    if pipe.device.type == "cuda"
                    else 0.0,
                    **metrics,
                }
            )
    extra = {
        "pool_path": str(Path(pool_path).resolve()),
        "pool_hash": doc["pool_hash"],
        "tie_break": doc["tie_break"],
        "query_ids": doc["query_order"],
        "train_ids": doc["train_ids"],
        "train_filter": doc["train_filter"],
        "manifest_path": doc["manifest_path"],
        "manifest_hash": doc["manifest_hash"],
        "encoder_parameter_sha256": doc["encoder_parameter_sha256"],
        "encoder_weights_sha256": doc["encoder_weights_sha256"],
        "segmenter_parameter_sha256": doc["segmenter_parameter_sha256"],
        "segmenter_weights_sha256": doc["segmenter_weights_sha256"],
        "dinov2_parameter_sha256": dino_bank.encoder_sha256,
        "dinov2_weights_sha256": weight_sha,
        "dinov2_source_revision": source_sha,
        "official_dinov2_source": OFFICIAL_DINOV2_SOURCE,
        "official_dinov2_weights_sha256": OFFICIAL_DINOV2_SHA,
        "retrieval_config": doc["retrieval_config"],
        "retrieval_config_hash": doc["retrieval_config_hash"],
        "source_revision": doc["source_revision"],
        "run_signature": signature,
        "adapted": True,
        "n_repeats": 1,
        "statistics_unit": "query",
        "bank_cache": str(pipe.bank.root),
        "dino_bank_cache": str(dino_bank.root),
    }
    log_lines = [
        f"method={METHOD}",
        "adapted=true",
        f"seed={int(seed)}",
        f"pool_hash={doc['pool_hash']}",
        f"queries={len(rows)}",
        f"dinov2_weights_sha256={weight_sha}",
        f"dinov2_source={source_sha}",
        "selection=stable-top2-inside-saved-resnet-top12",
    ]
    return write_outputs(target, cfg, pipe.manifest_hash, METHOD, rows, extra, PROTOCOL, log_lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="Run IRES-S3 inside the saved ResNet50 Top-12 pool")
    evaluate.add_argument("--pool", type=Path, required=True)
    evaluate.add_argument("--out-base", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dinov2-repo", type=Path, required=True)
    evaluate.add_argument("--dinov2-weights", type=Path, required=True)
    evaluate.add_argument("--encoder-weights")
    evaluate.add_argument("--segmenter-weights")
    sub.add_parser("self-test", help="Run the synthetic checks in this file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "self-test":
        self_test()
        return
    evaluate_ires(
        args.pool,
        args.out_base,
        args.seed,
        args.device,
        args.dinov2_repo,
        args.dinov2_weights,
        args.encoder_weights,
        args.segmenter_weights,
        enforce_pin=True,
    )


def test_s3_tie_and_pin(tmp=None):
    import tempfile

    assert choose_encoder(0.2, 0.2) == "resnet50"
    assert choose_encoder(0.1, 0.4) == "dinov2"
    assert stability_s3(torch.zeros(1, 8, 8)) == 0.0
    assert stability_s3(torch.ones(1, 8, 8)) == 0.0
    root = Path(tmp) if tmp is not None else Path(tempfile.mkdtemp())
    weights = root / "weights.bin"
    weights.write_bytes(b"not-the-official-checkpoint")
    try:
        check_dinov2_pin(weights, root, enforce=True)
    except ValueError as exc:
        assert "DINOv2" in str(exc)
    else:
        raise AssertionError("unpinned DINOv2 file was accepted")
    ids = [f"s{i}" for i in range(4)]
    vectors = torch.eye(4)
    assert stable_rank(ids, vectors, vectors[2])[0] == "s2"
    assert set(stable_rank(ids, vectors, vectors[1])) == set(ids)


def test_synthetic_ires_stays_inside_pool():
    import os
    import tempfile

    import far_icl.features as features_mod
    import far_icl.pipeline as pipeline_mod
    from far_icl.bank import CaseBank

    from experiments.fairness.run_random_top12 import _fixture, _install_fakes, export_pool

    tmp = Path(tempfile.mkdtemp())
    previous_cache = os.environ.get("FARICL_BANK_CACHE_ROOT")
    previous_feature = features_mod.ImageEncoder
    previous_pipeline = pipeline_mod.ImageEncoder
    data_mod, original_mask = _install_fakes()
    features_mod.ImageEncoder = pipeline_mod.ImageEncoder
    original_candidates = CaseBank.candidates
    os.environ["FARICL_BANK_CACHE_ROOT"] = str(tmp / "cache")
    try:
        manifest = _fixture(tmp)
        out = tmp / "out"
        pool_path = export_pool(manifest, out, "cpu")
        doc = load_pool(pool_path)

        def forbidden(self, *args, **kwargs):
            raise AssertionError("IRES evaluation regenerated a candidate pool")

        CaseBank.candidates = forbidden
        weights = tmp / "dino.bin"
        weights.write_bytes(b"synthetic-dino")
        _install_fakes.trace.clear()
        result = evaluate_ires(
            pool_path, out, 42, "cpu", tmp, weights, enforce_pin=False
        )
        suffix = _install_fakes.trace[_install_fakes.trace.index("segment") :]
        assert suffix == (["segment", "segment", "mask"]) * 2
        rows = __import__("json").loads((result / "per_case.json").read_text())["rows"]
        summary = __import__("json").loads((result / "summary.json").read_text())
        assert summary["n_cases"] == len(rows) == 2
        assert summary["statistics_unit"] == "query"
        for row in rows:
            saved = doc["pools"][row["case_id"]]
            assert row["resnet_support_ids"] == saved[:K], row
            assert row["dino_support_ids"] == [saved[11], saved[10]], (row["case_id"], saved, row["dino_support_ids"])
            assert "t15" not in row["dino_support_ids"]
            assert set(row["dino_support_ids"]) <= set(saved)
            assert set(row["support_ids"]) <= set(saved)
            assert row["chosen_encoder"] == "resnet50"
            assert row["segmentation_calls"] == 2
            assert len(set(row["support_ids"])) == K
        provenance = __import__("json").loads((result / "provenance.json").read_text())
        assert provenance["extra"]["pool_hash"] == doc["pool_hash"]
        assert provenance["extra"]["adapted"] is True
        stamp = (result / "summary.json").stat().st_mtime_ns
        assert evaluate_ires(pool_path, out, 42, "cpu", tmp, weights, enforce_pin=False) == result
        assert (result / "summary.json").stat().st_mtime_ns == stamp
        try:
            parse_args(["evaluate", "--pool", "p", "--out-base", "o", "--seed", "9",
                        "--dinov2-repo", "r", "--dinov2-weights", "w"])
        except SystemExit:
            pass
        else:
            raise AssertionError("seed 9 was accepted")
    finally:
        CaseBank.candidates = original_candidates
        data_mod.load_mask = original_mask
        features_mod.ImageEncoder = previous_feature
        pipeline_mod.ImageEncoder = previous_pipeline
        if previous_cache is None:
            os.environ.pop("FARICL_BANK_CACHE_ROOT", None)
        else:
            os.environ["FARICL_BANK_CACHE_ROOT"] = previous_cache


def self_test():
    test_s3_tie_and_pin()
    test_synthetic_ires_stays_inside_pool()
    print("PASS ires-s3-top12 self-test")


if __name__ == "__main__":
    main()
