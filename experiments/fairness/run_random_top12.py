"""Frozen ResNet50 Top-12 pool, plus uniform Random-Top12 draws.

The pool is exported once by ``prepare`` and then only read. Evaluation never
calls ``CaseBank.candidates`` and never rewrites a complete pool or result.

Pool tie break, matching the fixed baseline runner::

    CaseBank.candidates(query, resnet_vector, 12, rng=None, stable=True)

Random draws are independent across queries and repeats. The stream is
``numpy.random.SeedSequence(int(sha256(["fair-random-top12-v1", seed, case_id, repeat]), 16))``
and ``Generator.choice(12, size=2, replace=False)``. A later repeat may draw
the same pair. Official runs use seeds 42 and 43, ten repeats, and average
Dice, IoU, and finite HD95 inside each query before any cross-query mean.

Commands::

    python -m experiments.fairness.run_random_top12 prepare \\
        --manifest "$MANIFEST" --out-base "$OUT/top12" --device cuda
    python -m experiments.fairness.run_random_top12 evaluate \\
        --pool "$OUT/top12/shared_pool/resnet50_legal_train_top12_val.json" \\
        --out-base "$OUT/top12" --seed 42 --device cuda
    python -m experiments.fairness.run_random_top12 evaluate \\
        --pool "$OUT/top12/shared_pool/resnet50_legal_train_top12_val.json" \\
        --out-base "$OUT/top12" --seed 43 --device cuda
    python -m experiments.fairness.run_random_top12 audit \\
        --pool "$OUT/top12/shared_pool/resnet50_legal_train_top12_val.json" \\
        --out "$OUT/top12/fairness_audit.json" \\
        --evidence knn=path/to/knn_result_dir
    python -m experiments.fairness.run_random_top12 self-test
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
from far_icl.pipeline import summarize

SCHEMA = "resnet50-legal-train-top12-v1"
TIE_BREAK = "CaseBank.candidates(vector, n=12, rng=None, stable=True)"
K = 2
N = 12
REPEATS = 10
SEEDS = (42, 43)
SPLIT = "val"
IDENTITY = "image"
RNG_NAME = "fair-random-top12-v1"
METHOD = "random_top12"
EXPECTED_AUDIT = ("knn", "mmr", "dual", "swap_dice", "far", "repair")
RETRIEVAL_KEYS = (
    "manifest",
    "identity_scope",
    "image_size",
    "encoder",
    "encoder_repo",
    "encoder_weights",
    "segmenter",
    "segmenter_weights",
    "mask_values",
    "candidate_n",
    "k",
    "bank_domains",
    "query_domains",
    "train_query_domains",
    "hard_quantile",
)
HASH_KEYS = (
    "schema",
    "k",
    "candidate_n",
    "split",
    "identity_scope",
    "tie_break",
    "query_order",
    "train_ids",
    "train_filter",
    "pools",
    "cases",
    "manifest_path",
    "manifest_hash",
    "encoder_parameter_sha256",
    "encoder_weights_sha256",
    "segmenter_parameter_sha256",
    "segmenter_weights_sha256",
    "retrieval_config",
    "retrieval_config_hash",
    "actual_config",
    "source_revision",
)
POOL_KEYS = set(HASH_KEYS) | {"pool_hash"}
CASE_FIELDS = ("patient_id", "split", "task", "domain")
REPO = Path(__file__).resolve().parents[2]


def source_revision():
    revision = git_revision()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise RuntimeError(f"Full git revision required, got {revision!r}")
    return revision


def json_roundtrip(value):
    return json.loads(json.dumps(value))


def make_cfg(manifest, output, device, encoder_weights, segmenter_weights, seed):
    values = {
        "manifest": str(Path(manifest).resolve()),
        "output": str(Path(output).resolve()),
        "device": device,
        "identity_scope": IDENTITY,
        "k": K,
        "max_k": K,
        "initial_k": K,
        "candidate_n": N,
        "seed": int(seed),
        "encoder": "resnet50",
        "segmenter": "universeg",
        "encoder_weights": str(Path(encoder_weights).resolve()) if encoder_weights else None,
        "segmenter_weights": str(Path(segmenter_weights).resolve()) if segmenter_weights else None,
        "uncertainty": "none",
        "save_predictions": False,
    }
    return load_config(
        REPO / "configs/isic.yaml",
        [f"{key}={json.dumps(value)}" for key, value in values.items()],
    )


def locked_retrieval_config(cfg):
    locked = {}
    for key in RETRIEVAL_KEYS:
        value = cfg[key]
        if key in {"manifest", "encoder_weights", "segmenter_weights", "encoder_repo"} and value:
            value = str(Path(value).resolve())
        locked[key] = value
    if locked["identity_scope"] != IDENTITY or locked["encoder"] != "resnet50":
        raise ValueError("Top-12 pool requires image scope and a ResNet50 encoder")
    if locked["segmenter"] != "universeg" or locked["image_size"] != 128:
        raise ValueError("Top-12 pool requires frozen UniverSeg at 128 pixels")
    if locked["k"] != K or locked["candidate_n"] != N:
        raise ValueError("Top-12 pool requires K=2 and candidate_n=12")
    return json_roundtrip(locked)


def pool_hash(doc):
    return digest({key: doc[key] for key in HASH_KEYS})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pool_document(doc):
    """Reject a pool that is incomplete, hashed differently, or illegally composed."""
    _require(isinstance(doc, dict) and set(doc) == POOL_KEYS, "pool schema keys mismatch")
    _require(doc["schema"] == SCHEMA, "pool schema mismatch; refusing to regenerate")
    _require(doc["k"] == K and doc["candidate_n"] == N, "pool must be K=2 N=12")
    _require(doc["split"] == SPLIT and doc["identity_scope"] == IDENTITY, "pool must be val/image")
    _require(doc["tie_break"] == TIE_BREAK, "pool tie break mismatch; refusing to regenerate")
    _require(doc["pool_hash"] == pool_hash(doc), "pool_hash mismatch; refusing to regenerate")
    _require(doc["retrieval_config_hash"] == digest(doc["retrieval_config"]), "retrieval hash mismatch")
    _require(doc["source_revision"] and len(doc["source_revision"]) == 40, "source revision missing")
    queries = doc["query_order"]
    trains = doc["train_ids"]
    _require(isinstance(queries, list) and queries and len(queries) == len(set(queries)), "bad queries")
    _require(isinstance(trains, list) and trains and len(trains) == len(set(trains)), "bad train ids")
    _require(not set(queries) & set(trains), "query id appears in the training ids")
    _require(set(doc["pools"]) == set(queries), "pool queries do not match query_order")
    cases = doc["cases"]
    _require(set(cases) >= set(queries) | set(trains), "case table is missing ids")
    filt = doc["train_filter"]
    _require(filt["split"] == "train" and filt["identity_scope"] == IDENTITY, "train filter mismatch")
    _require(filt["same_task"] is True and filt["exclude_same_patient_id"] is True, "filter mismatch")
    _require(filt["patient_id_rule"] == "image:{case_id}", "identity rule mismatch")
    for case_id, row in cases.items():
        _require(set(row) == set(CASE_FIELDS), f"case fields mismatch: {case_id}")
        _require(row["patient_id"] == f"image:{case_id}", f"identity scope mismatch: {case_id}")
    for case_id in trains:
        _require(cases[case_id]["split"] == "train", f"{case_id} is not a training case")
    for query_id in queries:
        query = cases[query_id]
        _require(query["split"] == SPLIT, f"{query_id} is not a validation query")
        supports = doc["pools"][query_id]
        _require(isinstance(supports, list) and len(supports) == N, f"{query_id} pool length")
        _require(len(set(supports)) == N and set(supports) <= set(trains), f"{query_id} illegal supports")
        _require(query_id not in supports, f"{query_id} includes itself")
        for support_id in supports:
            support = cases[support_id]
            _require(support["task"] == query["task"], f"{query_id} task mismatch")
            _require(support["patient_id"] != query["patient_id"], f"{query_id} identity leak")
    return doc


def load_pool(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen pool is missing: {path}")
    return validate_pool_document(json.loads(path.read_text()))


def bind_pool(doc, pipe, cfg):
    """Match the frozen file to the live manifest, bank, and weights. Never rebuild it."""
    _require(doc["source_revision"] == source_revision(), "source revision mismatch; refusing to regenerate")
    live_retrieval = locked_retrieval_config(cfg)
    _require(live_retrieval == doc["retrieval_config"], "retrieval config mismatch; refusing to regenerate")
    _require(pipe.manifest_hash == doc["manifest_hash"], "manifest hash mismatch; refusing to regenerate")
    _require(pipe.bank.encoder_sha256 == doc["encoder_parameter_sha256"], "encoder hash mismatch")
    _require(pipe.segmenter_sha256 == doc["segmenter_parameter_sha256"], "UniverSeg hash mismatch")
    for label, recorded, current in (
        ("encoder", doc["encoder_weights_sha256"], cfg["encoder_weights"]),
        ("segmenter", doc["segmenter_weights_sha256"], cfg["segmenter_weights"]),
    ):
        actual = file_digest(current) if current else None
        _require(actual == recorded, f"{label} weight-file hash mismatch")
    queries = [case.case_id for case in pipe.queries(SPLIT)]
    trains = [entry["case"]["case_id"] for entry in pipe.bank.entries]
    _require(queries == doc["query_order"], "query order mismatch; refusing to regenerate")
    _require(trains == doc["train_ids"], "training-bank order mismatch; refusing to regenerate")
    by_id = {case.case_id: case for case in pipe.cases}
    for query_id, supports in doc["pools"].items():
        query = by_id[query_id]
        _require(query.patient_id == f"image:{query_id}" and query.split == SPLIT, "live query identity")
        for support_id in supports:
            support = by_id[support_id]
            legal = (
                support.split == "train"
                and support.task == query.task
                and support.patient_id != query.patient_id
                and support.patient_id == f"image:{support_id}"
            )
            _require(legal, f"live support {support_id} is not legal for {query_id}")
    return doc


def open_pipeline(cfg):
    from far_icl.pipeline import Pipeline

    return Pipeline(cfg)


def case_row(case):
    return {key: getattr(case, key) for key in CASE_FIELDS}


def export_pool(manifest, out_base, device, encoder_weights=None, segmenter_weights=None):
    """Write the ordered ResNet50 Top-12 pool. This is the only retrieval call."""
    out_base = Path(out_base)
    pool_path = out_base / "shared_pool" / "resnet50_legal_train_top12_val.json"
    work = out_base / "_work"
    cfg = make_cfg(manifest, work, device, encoder_weights, segmenter_weights, seed=0)
    if pool_path.exists():
        doc = load_pool(pool_path)
        bind_pool(doc, open_pipeline(cfg), cfg)
        print(f"Reuse frozen pool {pool_path}")
        return pool_path
    pipe = open_pipeline(cfg)
    from far_icl.data import load_case

    pools = {}
    for case in pipe.queries(SPLIT):
        data = load_case(case, cfg, with_mask=False)
        vector = pipe.encoder(data["rgb"][None].to(pipe.device))[0][0]
        # Exact fixed-baseline tie break. Evaluation must not repeat this call.
        chosen = pipe.bank.candidates(case, vector, N, None, stable=True)
        ids = [entry["case"]["case_id"] for entry in chosen]
        if len(ids) != N or len(set(ids)) != N:
            raise ValueError(f"{case.case_id}: need {N} unique legal supports, found {len(set(ids))}")
        pools[case.case_id] = ids
    retrieval = locked_retrieval_config(cfg)
    records = {case.case_id: case_row(case) for case in pipe.cases if case.split == "train"}
    for case in pipe.queries(SPLIT):
        records[case.case_id] = case_row(case)
    doc = {
        "schema": SCHEMA,
        "k": K,
        "candidate_n": N,
        "split": SPLIT,
        "identity_scope": IDENTITY,
        "tie_break": TIE_BREAK,
        "query_order": [case.case_id for case in pipe.queries(SPLIT)],
        "train_ids": [entry["case"]["case_id"] for entry in pipe.bank.entries],
        "train_filter": {
            "split": "train",
            "same_task": True,
            "exclude_same_patient_id": True,
            "identity_scope": IDENTITY,
            "patient_id_rule": "image:{case_id}",
            "bank_domains": cfg["bank_domains"],
        },
        "pools": pools,
        "cases": records,
        "manifest_path": cfg["manifest"],
        "manifest_hash": pipe.manifest_hash,
        "encoder_parameter_sha256": pipe.bank.encoder_sha256,
        "encoder_weights_sha256": file_digest(cfg["encoder_weights"]) if cfg["encoder_weights"] else None,
        "segmenter_parameter_sha256": pipe.segmenter_sha256,
        "segmenter_weights_sha256": file_digest(cfg["segmenter_weights"]) if cfg["segmenter_weights"] else None,
        "retrieval_config": retrieval,
        "retrieval_config_hash": digest(retrieval),
        "actual_config": json_roundtrip(cfg),
        "source_revision": source_revision(),
    }
    doc = json_roundtrip(doc)
    doc["pool_hash"] = pool_hash(doc)
    validate_pool_document(doc)
    if pool_path.exists():
        raise FileExistsError(f"Refusing to overwrite pool {pool_path}")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pool_path.with_name(f".{pool_path.name}.{os.getpid()}.partial")
    try:
        write_json(doc, temporary)
        os.rename(temporary, pool_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(pool_path)
    return pool_path


def sample_pair(pool_ids, seed, case_id, repeat):
    """Draw 2 distinct supports. Repeats do not consume a shared generator."""
    if len(pool_ids) != N or len(set(pool_ids)) != N:
        raise ValueError("Random-Top12 samples from exactly 12 unique saved ids")
    key = [RNG_NAME, int(seed), str(case_id), int(repeat)]
    rng = np.random.default_rng(np.random.SeedSequence(int(digest(key), 16)))
    picked = rng.choice(N, size=K, replace=False)
    return [pool_ids[int(index)] for index in picked]


def mean_defined(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def publish_directory(target, populate):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{os.getpid()}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        populate(staging)
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def result_dir(out_base, method, seed, signature, tag):
    return Path(out_base) / method / f"seed{seed}" / f"{tag}_{signature[:12]}"


def complete_match(target, signature, query_ids):
    names = ("config.json", "provenance.json", "per_case.json", "summary.json", "log.txt")
    if not target.exists():
        return False
    if not all((target / name).is_file() for name in names):
        raise FileExistsError(f"Incomplete output, refusing to overwrite: {target}")
    saved = json.loads((target / "provenance.json").read_text())
    rows = json.loads((target / "per_case.json").read_text())["rows"]
    matched = (
        saved["extra"].get("run_signature") == signature
        and saved["git_revision"] == source_revision()
        and [row["case_id"] for row in rows] == query_ids
    )
    if matched:
        return True
    raise FileExistsError(f"Complete output does not match this run, refusing to overwrite: {target}")


def write_outputs(target, cfg, manifest_hash, method, rows, extra, protocol, log_lines):
    def populate(staging):
        write_json(cfg, staging / "config.json")
        write_json(
            {
                "method": method,
                "protocol": protocol,
                "manifest_hash": manifest_hash,
                "identity_scope": IDENTITY,
                "git_revision": source_revision(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "extra": extra,
            },
            staging / "provenance.json",
        )
        write_json(
            {
                "signature": digest([manifest_hash, method, cfg]),
                "manifest_hash": manifest_hash,
                "identity_scope": IDENTITY,
                "statistics_unit": "query",
                "rows": rows,
            },
            staging / "per_case.json",
        )
        summary = summarize(rows, IDENTITY)
        summary["statistics_unit"] = "query"
        summary["n_repeats"] = extra.get("n_repeats", 1)
        summary["n_queries"] = len(rows)
        write_json(summary, staging / "summary.json")
        (staging / "log.txt").write_text("\n".join(log_lines) + "\n")
        print(summary)

    return publish_directory(target, populate)


def evaluate_random(pool_path, out_base, seed, device, encoder_weights=None, segmenter_weights=None):
    if int(seed) not in SEEDS:
        raise ValueError("Random-Top12 official seeds are only 42 and 43")
    doc = load_pool(pool_path)
    work = Path(out_base) / "_work"
    weights = doc["retrieval_config"]
    cfg = make_cfg(
        weights["manifest"],
        work,
        device,
        encoder_weights if encoder_weights is not None else weights["encoder_weights"],
        segmenter_weights if segmenter_weights is not None else weights["segmenter_weights"],
        seed,
    )
    pipe = open_pipeline(cfg)
    bind_pool(doc, pipe, cfg)
    signature = digest(
        {
            "method": METHOD,
            "seed": int(seed),
            "pool_hash": doc["pool_hash"],
            "retrieval_config_hash": doc["retrieval_config_hash"],
            "repeats": REPEATS,
            "rng": RNG_NAME,
            "revision": source_revision(),
            "numpy": np.__version__,
        }
    )
    tag = f"{METHOD}_k{K}_n{N}_{SPLIT}_seed{int(seed)}_r{REPEATS}"
    target = result_dir(out_base, METHOD, int(seed), signature, tag)
    if complete_match(target, signature, doc["query_order"]):
        print(f"Reuse completed run: {target}")
        return target
    from far_icl.data import load_case, load_mask
    from far_icl.metrics import diversity, segmentation_metrics

    by_id = {entry["case"]["case_id"]: entry for entry in pipe.bank.entries}
    cases = {case.case_id: case for case in pipe.queries(SPLIT)}
    rows = []
    with torch.inference_mode():
        for query_id in doc["query_order"]:
            case = cases[query_id]
            pipe.segmenter.reset_stats()
            if pipe.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(pipe.device)
            pipe.segmenter.sync()
            start = __import__("time").perf_counter()
            data = load_case(case, cfg, with_mask=False)
            draws = [sample_pair(doc["pools"][query_id], seed, query_id, repeat) for repeat in range(REPEATS)]
            predictions = []
            for support_ids in draws:
                supports = [by_id[support_id] for support_id in support_ids]
                predictions.append(pipe.segmenter(data["image"], supports))
            pipe.segmenter.sync()
            elapsed = __import__("time").perf_counter() - start
            truth = load_mask(case, cfg, data["original_size"])
            repeats = []
            for repeat, (support_ids, prediction) in enumerate(zip(draws, predictions)):
                metrics = segmentation_metrics(prediction, truth, data["spacing"], cfg["surface_tolerance"])
                features = torch.stack([by_id[support_id]["features"]["global_feature"] for support_id in support_ids])
                repeats.append(
                    {
                        "repeat": repeat,
                        "support_ids": support_ids,
                        "support_patient_ids": [by_id[support_id]["case"]["patient_id"] for support_id in support_ids],
                        "diversity": diversity(features),
                        **metrics,
                    }
                )
            rows.append(
                {
                    "case_id": query_id,
                    "patient_id": case.patient_id,
                    "task": case.task,
                    "domain": case.domain,
                    "method": METHOD,
                    "split": SPLIT,
                    "k": K,
                    "pool_size": N,
                    "n_repeats": REPEATS,
                    "seed": int(seed),
                    "pool_hash": doc["pool_hash"],
                    "repeat_support_ids": draws,
                    "dice": mean_defined(repeats, "dice"),
                    "iou": mean_defined(repeats, "iou"),
                    "hd95": mean_defined(repeats, "hd95"),
                    "nsd": mean_defined(repeats, "nsd"),
                    "hd95_finite_repeats": sum(row["hd95"] is not None for row in repeats),
                    "hd95_undefined_repeats": sum(row["hd95"] is None for row in repeats),
                    "diversity": mean_defined(repeats, "diversity"),
                    "elapsed_seconds": elapsed,
                    "segmentation_calls": pipe.segmenter.calls,
                    "segmentation_seconds": pipe.segmenter.seconds,
                    "support_evaluations": pipe.segmenter.support_evaluations,
                    "peak_gpu_mb": torch.cuda.max_memory_allocated(pipe.device) / 1024**2
                    if pipe.device.type == "cuda"
                    else 0.0,
                    "repeats": repeats,
                }
            )
    extra = {
        "pool_path": str(Path(pool_path).resolve()),
        "pool_hash": doc["pool_hash"],
        "pool_schema": SCHEMA,
        "tie_break": TIE_BREAK,
        "query_ids": doc["query_order"],
        "train_ids": doc["train_ids"],
        "train_filter": doc["train_filter"],
        "manifest_path": doc["manifest_path"],
        "manifest_hash": doc["manifest_hash"],
        "encoder_parameter_sha256": doc["encoder_parameter_sha256"],
        "encoder_weights_sha256": doc["encoder_weights_sha256"],
        "segmenter_parameter_sha256": doc["segmenter_parameter_sha256"],
        "segmenter_weights_sha256": doc["segmenter_weights_sha256"],
        "retrieval_config": doc["retrieval_config"],
        "retrieval_config_hash": doc["retrieval_config_hash"],
        "source_revision": doc["source_revision"],
        "run_signature": signature,
        "rng": RNG_NAME,
        "n_repeats": REPEATS,
        "statistics_unit": "query",
        "bank_cache": str(pipe.bank.root),
        "bank_signature": pipe.bank.signature,
    }
    log_lines = [
        f"method={METHOD}",
        f"seed={int(seed)}",
        f"repeats={REPEATS}",
        f"pool_hash={doc['pool_hash']}",
        f"queries={len(rows)}",
        f"rng={RNG_NAME}",
        "statistics_unit=query",
    ]
    run_cfg = json_roundtrip(cfg) | {"method": METHOD, "random_repeats": REPEATS, "pool_hash": doc["pool_hash"]}
    return write_outputs(
        target,
        run_cfg,
        pipe.manifest_hash,
        METHOD,
        rows,
        extra,
        "Uniform Random-Top12 inside one frozen ResNet50 legal-train pool",
        log_lines,
    )


def _status(name, status, detail):
    if status not in {"PASS", "UNKNOWN", "MISMATCH"}:
        raise ValueError(status)
    return {"name": name, "status": status, "detail": detail}


def _overall(checks):
    statuses = [item["status"] for item in checks]
    if "MISMATCH" in statuses:
        return "MISMATCH"
    if "UNKNOWN" in statuses or not statuses:
        return "UNKNOWN"
    return "PASS"


def _load_optional(path):
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}


def audit_method(doc, name, evidence):
    if evidence is None:
        return {"status": "UNKNOWN", "checks": [_status("evidence", "UNKNOWN", "no result path provided")]}
    path = Path(evidence)
    if path.is_file() and path.name == "per_case.json":
        directory = path.parent
    elif path.is_dir():
        directory = path
    else:
        return {"status": "UNKNOWN", "checks": [_status("evidence", "UNKNOWN", f"missing {path}")]}
    per_case = _load_optional(directory / "per_case.json")
    provenance = _load_optional(directory / "provenance.json") or {}
    config = _load_optional(directory / "config.json") or {}
    checks = []
    if not isinstance(per_case, dict) or "rows" not in per_case:
        checks.append(_status("per_case", "UNKNOWN", "per_case.json rows are absent"))
        return {"status": _overall(checks), "checks": checks, "evidence": str(directory)}
    rows = per_case["rows"]
    if not isinstance(rows, list) or not rows:
        checks.append(_status("per_case", "UNKNOWN", "no rows"))
        return {"status": _overall(checks), "checks": checks, "evidence": str(directory)}
    checks.append(_status("per_case", "PASS", f"{len(rows)} rows"))
    found = {row.get("case_id") for row in rows}
    pool_queries = set(doc["query_order"])
    if found - pool_queries:
        checks.append(_status("queries", "MISMATCH", f"outside frozen queries: {sorted(found - pool_queries)}"))
    elif pool_queries - found:
        checks.append(_status("queries", "UNKNOWN", f"missing queries: {sorted(pool_queries - found)}"))
    else:
        checks.append(_status("queries", "PASS", "same validation query ids"))
    k_values = {row.get("k") for row in rows}
    if any(row.get("k") is None for row in rows):
        checks.append(_status("k", "UNKNOWN", "at least one row has no k"))
    elif k_values != {K}:
        checks.append(_status("k", "MISMATCH", f"k values {sorted(k_values)}"))
    else:
        checks.append(_status("k", "PASS", "K=2"))
    support_rows = [row for row in rows if isinstance(row.get("support_ids"), list)]
    if len(support_rows) != len(rows):
        checks.append(_status("support_ids", "UNKNOWN", "support_ids missing on at least one row"))
    elif any(len(row["support_ids"]) != K or len(set(row["support_ids"])) != K for row in rows):
        checks.append(_status("support_ids", "MISMATCH", "supports are not two unique ids"))
    else:
        checks.append(_status("support_ids", "PASS", "two unique supports"))
    if len(support_rows) != len(rows):
        checks.append(_status("within_top12", "UNKNOWN", "cannot test membership without support_ids"))
    else:
        outside = []
        for row in rows:
            allowed = doc["pools"].get(row.get("case_id"))
            if allowed is None:
                continue
            if any(support not in allowed for support in row["support_ids"]):
                outside.append(row["case_id"])
        if outside:
            checks.append(_status("within_top12", "MISMATCH", f"supports outside Top-12: {outside}"))
        elif found >= pool_queries:
            checks.append(_status("within_top12", "PASS", "every recorded support is inside the saved Top-12"))
        else:
            checks.append(_status("within_top12", "UNKNOWN", "membership holds only for rows that were present"))
    listed = []
    missing_list = False
    for row in rows:
        candidates = row.get("candidate_ids", row.get("pool_ids"))
        if not isinstance(candidates, list):
            missing_list = True
            continue
        saved = doc["pools"].get(row.get("case_id"))
        listed.append(saved == candidates)
    if not listed:
        checks.append(_status("ordered_top12", "UNKNOWN", "candidate_ids/pool_ids were not recorded"))
    elif missing_list or len(listed) != len(rows):
        checks.append(_status("ordered_top12", "UNKNOWN", "candidate list is only present on some rows"))
    elif all(listed):
        checks.append(_status("ordered_top12", "PASS", "ordered Top-12 matches the frozen pool"))
    else:
        checks.append(_status("ordered_top12", "MISMATCH", "ordered candidate list differs from the frozen pool"))
    manifest = provenance.get("manifest_hash")
    if not isinstance(manifest, str):
        checks.append(_status("manifest_hash", "UNKNOWN", "provenance has no manifest_hash"))
    elif manifest == doc["manifest_hash"]:
        checks.append(_status("manifest_hash", "PASS", manifest))
    else:
        checks.append(_status("manifest_hash", "MISMATCH", manifest))
    scope = provenance.get("identity_scope", config.get("identity_scope"))
    if scope is None:
        checks.append(_status("identity_scope", "UNKNOWN", "identity scope was not recorded"))
    elif scope == IDENTITY:
        checks.append(_status("identity_scope", "PASS", scope))
    else:
        checks.append(_status("identity_scope", "MISMATCH", str(scope)))
    segmenter_hash = provenance.get("segmenter_sha256") or (provenance.get("extra") or {}).get(
        "segmenter_parameter_sha256"
    )
    if not isinstance(segmenter_hash, str):
        checks.append(
            _status(
                "universeg_hash",
                "UNKNOWN",
                "no UniverSeg parameter hash; a null ranker checkpoint_sha256 is not that hash",
            )
        )
    elif segmenter_hash == doc["segmenter_parameter_sha256"]:
        checks.append(_status("universeg_hash", "PASS", segmenter_hash))
    else:
        checks.append(_status("universeg_hash", "MISMATCH", segmenter_hash))
    if not config or "_read_error" in config:
        checks.append(_status("preprocess", "UNKNOWN", "config.json absent"))
    else:
        observed = {
            "image_size": config.get("image_size"),
            "encoder": config.get("encoder"),
            "segmenter": config.get("segmenter"),
            "candidate_n": config.get("candidate_n"),
        }
        if any(value is None for value in observed.values()):
            checks.append(_status("preprocess", "UNKNOWN", f"incomplete config {observed}"))
        elif observed == {"image_size": 128, "encoder": "resnet50", "segmenter": "universeg", "candidate_n": N}:
            checks.append(_status("preprocess", "PASS", "128 / resnet50 / universeg / N=12"))
        else:
            checks.append(_status("preprocess", "MISMATCH", str(observed)))
    return {"status": _overall(checks), "checks": checks, "evidence": str(directory), "method_name": name}


def audit_pool(pool_path, out_path, evidence):
    doc = load_pool(pool_path)
    out_path = Path(out_path)
    supplied = {name: path for name, path in evidence}
    names = list(EXPECTED_AUDIT)
    for name in supplied:
        if name not in names:
            names.append(name)
    resolved = []
    for name in names:
        raw = supplied.get(name)
        if raw is not None and Path(raw).exists():
            resolved.append(Path(raw).resolve())
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit {out_path}")
    if any(out_path.resolve() == item or out_path.resolve() in item.parents for item in resolved):
        raise ValueError("audit output must not overwrite evidence")
    before = {item: item.read_bytes() if item.is_file() else None for item in resolved}
    report = {
        "protocol": "read-only Top-12 fairness audit",
        "pool_hash": doc["pool_hash"],
        "pool_path": str(Path(pool_path).resolve()),
        "source_revision": source_revision(),
        "note": "UNKNOWN is missing evidence, not a pass. No historical result is edited or rerun.",
        "methods": {name: audit_method(doc, name, supplied.get(name)) for name in names},
    }
    for item, payload in before.items():
        if item.is_file() and item.read_bytes() != payload:
            raise RuntimeError(f"audit mutated evidence {item}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.{os.getpid()}.partial")
    try:
        write_json(report, temporary)
        os.rename(temporary, out_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(out_path)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Export one frozen ResNet50 Top-12 pool")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--out-base", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--encoder-weights")
    prepare.add_argument("--segmenter-weights")
    evaluate = sub.add_parser("evaluate", help="Draw 10 Random-Top12 repeats from the saved pool")
    evaluate.add_argument("--pool", type=Path, required=True)
    evaluate.add_argument("--out-base", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--encoder-weights")
    evaluate.add_argument("--segmenter-weights")
    review = sub.add_parser("audit", help="Read existing results; missing evidence stays UNKNOWN")
    review.add_argument("--pool", type=Path, required=True)
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--evidence", action="append", default=[], metavar="NAME=PATH")
    sub.add_parser("self-test", help="Run the synthetic checks in this file")
    args = parser.parse_args(argv)
    if args.command == "audit":
        parsed = []
        for item in args.evidence:
            name, sep, raw = item.partition("=")
            if not sep or not name:
                parser.error(f"evidence must be NAME=PATH, got {item}")
            parsed.append((name, raw))
        args.evidence = parsed
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        export_pool(args.manifest, args.out_base, args.device, args.encoder_weights, args.segmenter_weights)
    elif args.command == "evaluate":
        evaluate_random(
            args.pool, args.out_base, args.seed, args.device, args.encoder_weights, args.segmenter_weights
        )
    elif args.command == "audit":
        audit_pool(args.pool, args.out, args.evidence)
    else:
        self_test()


def _fixture(tmp):
    import csv

    from PIL import Image

    mask = tmp / "mask.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask)
    manifest = tmp / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "patient_id", "task", "image_path", "mask_path", "split"])
        for index in range(16):
            case_id = f"t{index:02d}"
            image = tmp / f"{case_id}.png"
            pixels = np.zeros((8, 8, 3), dtype=np.uint8)
            pixels[..., 0] = 255
            pixels[..., 1] = index
            pixels[..., 2] = index
            Image.fromarray(pixels).save(image)
            writer.writerow([case_id, f"image:{case_id}", "lesion", image.name, mask.name, "train"])
        for index in range(2):
            case_id = f"v{index:02d}"
            image = tmp / f"{case_id}.png"
            pixels = np.zeros((8, 8, 3), dtype=np.uint8)
            pixels[..., 0] = 255
            pixels[..., 2] = 255 - index
            Image.fromarray(pixels).save(image)
            writer.writerow([case_id, f"image:{case_id}", "lesion", image.name, mask.name, "val"])
    return manifest


def _install_fakes():
    import sys
    from types import SimpleNamespace

    import far_icl.pipeline as pipeline_mod

    class FakeEncoder(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.kind = cfg["encoder"]

        def forward(self, rgb):
            green = rgb[:, 1].mean((-2, -1))
            blue = rgb[:, 2].mean((-2, -1))
            raw = torch.stack((torch.ones_like(green), -green if self.kind != "dinov2" else blue), dim=-1)
            vector = torch.nn.functional.normalize(raw, dim=-1)
            return vector, vector[:, :, None, None]

    class ZeroLogits(torch.nn.Module):
        def forward(self, query, images, masks):
            _install_fakes.trace.append("segment")
            return torch.zeros_like(query)

    _install_fakes.trace = []
    pipeline_mod.ImageEncoder = FakeEncoder
    sys.modules["universeg"] = SimpleNamespace(universeg=lambda pretrained=False: ZeroLogits())
    import far_icl.data as data_mod

    original = data_mod.load_mask

    def wrapped(case, cfg, original_size=None):
        _install_fakes.trace.append("mask")
        return original(case, cfg, original_size)

    data_mod.load_mask = wrapped
    return data_mod, original


def test_sample_pair_is_stable_and_can_recur():
    ids = [f"s{i:02d}" for i in range(N)]
    first = [sample_pair(ids, 42, "query", repeat) for repeat in range(REPEATS)]
    assert first == [sample_pair(ids, 42, "query", repeat) for repeat in range(REPEATS)]
    assert first != [sample_pair(ids, 43, "query", repeat) for repeat in range(REPEATS)]
    assert first != [sample_pair(ids, 42, "other", repeat) for repeat in range(REPEATS)]
    for draw in first:
        assert len(draw) == len(set(draw)) == K
        assert set(draw) <= set(ids)
    many = [tuple(sample_pair(ids, 42, "query", repeat)) for repeat in range(400)]
    assert len(set(many)) < len(many)


def test_pool_document_rejects_illegal_edits():
    ids = [f"t{i:02d}" for i in range(N)]
    query = "v00"
    cases = {case_id: {"patient_id": f"image:{case_id}", "split": "train", "task": "lesion", "domain": "unknown"} for case_id in ids}
    cases[query] = {"patient_id": f"image:{query}", "split": "val", "task": "lesion", "domain": "unknown"}
    retrieval = {"manifest": "/m.csv", "identity_scope": IDENTITY, "k": K, "candidate_n": N}
    doc = {key: None for key in HASH_KEYS}
    doc.update(
        {
            "schema": SCHEMA,
            "k": K,
            "candidate_n": N,
            "split": SPLIT,
            "identity_scope": IDENTITY,
            "tie_break": TIE_BREAK,
            "query_order": [query],
            "train_ids": ids,
            "train_filter": {
                "split": "train",
                "same_task": True,
                "exclude_same_patient_id": True,
                "identity_scope": IDENTITY,
                "patient_id_rule": "image:{case_id}",
                "bank_domains": [],
            },
            "pools": {query: ids},
            "cases": cases,
            "manifest_path": "/m.csv",
            "manifest_hash": "a" * 64,
            "encoder_parameter_sha256": "b" * 64,
            "encoder_weights_sha256": None,
            "segmenter_parameter_sha256": "c" * 64,
            "segmenter_weights_sha256": None,
            "retrieval_config": retrieval,
            "retrieval_config_hash": digest(retrieval),
            "actual_config": {"seed": 0},
            "source_revision": "d" * 40,
        }
    )
    doc["pool_hash"] = pool_hash(doc)
    validate_pool_document(doc)
    broken = dict(doc)
    broken["pool_hash"] = "e" * 64
    try:
        validate_pool_document(broken)
    except ValueError as exc:
        assert "pool_hash" in str(exc)
    else:
        raise AssertionError("tampered hash was accepted")
    duplicated = json.loads(json.dumps(doc))
    duplicated["pools"][query] = [ids[0], *ids[:-1]]
    duplicated["pool_hash"] = pool_hash(duplicated)
    try:
        validate_pool_document(duplicated)
    except ValueError as exc:
        assert "unique" in str(exc) or "illegal" in str(exc)
    else:
        raise AssertionError("duplicate support was accepted")


def test_synthetic_pool_random_and_audit(tmp_path=None):
    import tempfile

    tmp = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    from far_icl.bank import CaseBank

    previous = os.environ.get("FARICL_BANK_CACHE_ROOT")
    data_mod, original_mask = _install_fakes()
    original_candidates = CaseBank.candidates
    os.environ["FARICL_BANK_CACHE_ROOT"] = str(tmp / "cache")
    try:
        manifest = _fixture(tmp)
        out = tmp / "out"
        pool_path = export_pool(manifest, out, "cpu")
        doc = load_pool(pool_path)
        assert doc["query_order"] == ["v00", "v01"]
        assert doc["pools"]["v00"] == [f"t{i:02d}" for i in range(N)]
        cfg = make_cfg(manifest, out / "_work", "cpu", None, None, 0)
        pipe = open_pipeline(cfg)
        from far_icl.data import load_case

        for case in pipe.queries(SPLIT):
            data = load_case(case, cfg, with_mask=False)
            vector = pipe.encoder(data["rgb"][None].to(pipe.device))[0][0]
            fresh = [entry["case"]["case_id"] for entry in pipe.bank.candidates(case, vector, N, None, stable=True)]
            assert fresh == doc["pools"][case.case_id]
        original_candidates = CaseBank.candidates

        def forbidden(self, *args, **kwargs):
            raise AssertionError("evaluation regenerated a candidate pool")

        CaseBank.candidates = forbidden
        _install_fakes.trace.clear()
        first = evaluate_random(pool_path, out, 42, "cpu")
        assert _install_fakes.trace[ _install_fakes.trace.index("segment") :] == (["segment"] * REPEATS + ["mask"]) * 2
        stamp = (first / "summary.json").stat().st_mtime_ns
        assert evaluate_random(pool_path, out, 42, "cpu") == first
        assert (first / "summary.json").stat().st_mtime_ns == stamp
        rows = json.loads((first / "per_case.json").read_text())["rows"]
        summary = json.loads((first / "summary.json").read_text())
        assert summary["n_cases"] == summary["dice"]["n"] == 2
        assert summary["statistics_unit"] == "query"
        assert len(rows) == 2
        for row in rows:
            assert row["segmentation_calls"] == REPEATS
            assert len(row["repeats"]) == REPEATS
            assert row["dice"] == sum(item["dice"] for item in row["repeats"]) / REPEATS
            for draw in row["repeat_support_ids"]:
                assert len(set(draw)) == K and set(draw) <= set(doc["pools"][row["case_id"]])
        second = evaluate_random(pool_path, out, 43, "cpu")
        other = json.loads((second / "per_case.json").read_text())["rows"]
        assert [row["repeat_support_ids"] for row in rows] != [row["repeat_support_ids"] for row in other]
        saved = (first / "per_case.json").read_bytes()
        (first / "summary.json").unlink()
        try:
            evaluate_random(pool_path, out, 42, "cpu")
        except FileExistsError:
            pass
        else:
            raise AssertionError("partial output was overwritten")
        assert not (first / "summary.json").exists()
        assert (first / "per_case.json").read_bytes() == saved
        tampered = tmp / "tampered.json"
        payload = json.loads(pool_path.read_text())
        payload["pools"]["v00"] = list(reversed(payload["pools"]["v00"]))
        tampered.write_text(json.dumps(payload))
        try:
            evaluate_random(tampered, tmp / "unused", 42, "cpu")
        except ValueError as exc:
            assert "pool_hash" in str(exc)
        else:
            raise AssertionError("tampered pool was evaluated")
        evidence = tmp / "historical"
        evidence.mkdir()
        historical = {
            "rows": [
                {
                    "case_id": query_id,
                    "k": K,
                    "support_ids": doc["pools"][query_id][:K],
                    "candidate_ids": doc["pools"][query_id],
                }
                for query_id in doc["query_order"]
            ]
        }
        write_json(historical, evidence / "per_case.json")
        write_json(
            {
                "manifest_hash": doc["manifest_hash"],
                "identity_scope": IDENTITY,
                "segmenter_sha256": doc["segmenter_parameter_sha256"],
            },
            evidence / "provenance.json",
        )
        write_json(
            {"image_size": 128, "encoder": "resnet50", "segmenter": "universeg", "candidate_n": N, "k": K},
            evidence / "config.json",
        )
        before = {path: path.read_bytes() for path in evidence.iterdir()}
        report = audit_pool(pool_path, tmp / "audit.json", [("knn", str(evidence))])
        assert report["methods"]["knn"]["status"] == "PASS"
        assert report["methods"]["repair"]["status"] == "UNKNOWN"
        assert {path: path.read_bytes() for path in evidence.iterdir()} == before
        mismatched = tmp / "mismatch"
        mismatched.mkdir()
        write_json(
            {"rows": [{"case_id": "v00", "k": K, "support_ids": ["t14", "t15"]}]},
            mismatched / "per_case.json",
        )
        mismatch_report = audit_pool(pool_path, tmp / "audit_mismatch.json", [("knn", str(mismatched))])
        assert mismatch_report["methods"]["knn"]["status"] == "MISMATCH"
    finally:
        CaseBank.candidates = original_candidates
        data_mod.load_mask = original_mask
        if previous is None:
            os.environ.pop("FARICL_BANK_CACHE_ROOT", None)
        else:
            os.environ["FARICL_BANK_CACHE_ROOT"] = previous


def self_test():
    test_sample_pair_is_stable_and_can_recur()
    test_pool_document_rejects_illegal_edits()
    test_synthetic_pool_random_and_audit()
    try:
        parse_args(["evaluate", "--pool", "pool.json", "--out-base", "out", "--seed", "7"])
    except SystemExit:
        pass
    else:
        raise AssertionError("seed 7 was accepted")
    print("PASS random-top12 self-test")


if __name__ == "__main__":
    main()
