"""Shared output format for experiment comparisons."""

import platform
import subprocess
from pathlib import Path

import torch

from far_icl.config import digest, write_json
from far_icl.pipeline import summarize


def git_revision():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True
    )
    return revision.stdout.strip() or "unknown"


def save_results(
    cfg, manifest_hash, method, split, rows, extra=None, *, target=None, protocol=None, revision=None
):
    support_k = rows[0].get("k")
    k_label = "unknown" if support_k is None else str(support_k)
    if target is None:
        target = Path(cfg["output"]) / "results" / f"{method}_k{k_label}_{split}_seed{cfg['seed']}"
    target.mkdir(parents=True, exist_ok=True)
    write_json(cfg, target / "config.json")
    write_json(
        {
            "method": method,
            "protocol": protocol or "ISIC2018 exploratory image-level comparison",
            "manifest_hash": manifest_hash,
            "identity_scope": cfg["identity_scope"],
            "git_revision": revision or git_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "extra": extra or {},
        },
        target / "provenance.json",
    )
    write_json(
        {
            "signature": digest([manifest_hash, method, cfg]),
            "manifest_hash": manifest_hash,
            "identity_scope": cfg["identity_scope"],
            "rows": rows,
        },
        target / "per_case.json",
    )
    summary = summarize(rows, cfg["identity_scope"])
    summary["isic_thresholded_jaccard_0_65"] = sum(
        row["iou"] if row["iou"] >= 0.65 else 0.0 for row in rows
    ) / len(rows)
    write_json(summary, target / "summary.json")
    print(summary)
    return target
