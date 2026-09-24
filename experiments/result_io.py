"""Shared output format for paper-comparison adapters."""

import platform
import subprocess
from pathlib import Path

import torch

from far_icl.config import digest, write_json
from far_icl.pipeline import summarize


def save_results(cfg, manifest_hash, method, split, rows, extra=None):
    target = Path(cfg["output"]) / "results" / f"{method}_k{cfg['k']}_{split}_seed{cfg['seed']}"
    target.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    write_json(cfg, target / "config.json")
    write_json(
        {
            "method": method,
            "protocol": "ISIC2018 exploratory image-level comparison",
            "manifest_hash": manifest_hash,
            "identity_scope": cfg["identity_scope"],
            "git_revision": revision.stdout.strip() or "unknown",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "extra": extra or {},
        },
        target / "provenance.json",
    )
    write_json(
        {"signature": digest([manifest_hash, method, cfg]), "manifest_hash": manifest_hash,
         "identity_scope": cfg["identity_scope"], "rows": rows},
        target / "per_case.json",
    )
    summary = summarize(rows, cfg["identity_scope"])
    summary["isic_thresholded_jaccard_0_65"] = sum(
        row["iou"] if row["iou"] >= 0.65 else 0.0 for row in rows
    ) / len(rows)
    write_json(summary, target / "summary.json")
    print(summary)
    return target
