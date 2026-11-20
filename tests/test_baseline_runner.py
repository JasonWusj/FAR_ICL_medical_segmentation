"""Synthetic integration checks of the frozen-baseline CLI, bank and output protocol."""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.baselines.run import parse_args, run


def test_baseline_cli_defaults_and_invalid_settings():
    base = ["--method", "all", "--manifest", "samples.csv", "--out-base", "run"]
    assert parse_args(base).candidate_n == 64
    assert parse_args([*base, "--k", "72"]).candidate_n == 72
    for options in (
        ["--k", "0"],
        ["--k", "8", "--candidate-n", "4"],
        ["--mmr-lambda", "1.1"],
        ["--random-repeats", "0"],
        ["--split", "train"],
    ):
        with pytest.raises(SystemExit):
            parse_args([*base, *options])


def test_synthetic_baseline_suite(tmp_path, monkeypatch):
    # Substitute only the two pretrained networks. Exercise the actual bank, runner,
    # frozen Segmenter wrapper, data loading, selection, metrics and result writer.
    class Encoder(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()

        def forward(self, x):
            fmap = torch.nn.functional.avg_pool2d(x, 32)
            return torch.nn.functional.normalize(fmap.mean((-2, -1))), fmap

    class Logits(torch.nn.Module):
        def forward(self, query, images, masks):
            assert images.shape == masks.shape
            assert images.shape[:1] + images.shape[2:] == query.shape == (1, 1, 128, 128)
            return torch.full_like(query, -5.0)

    monkeypatch.setattr("far_icl.pipeline.ImageEncoder", Encoder)
    monkeypatch.setitem(sys.modules, "universeg", SimpleNamespace(universeg=lambda pretrained: Logits()))
    monkeypatch.delenv("FARICL_BANK_CACHE_ROOT", raising=False)
    Image.fromarray(np.zeros((8, 8), dtype="uint8")).save(tmp_path / "mask.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "patient_id", "task", "image_path", "mask_path", "split"])
        for i in range(9):
            pixels = np.zeros((8, 8, 3), dtype="uint8")
            pixels.flat[i] = 255
            Image.fromarray(pixels).save(tmp_path / f"image{i}.png")
            writer.writerow(
                [str(i), f"image:{i}", "synthetic", f"image{i}.png", "mask.png", "train" if i < 8 else "val"]
            )

    options = [
        "--out-base",
        str(tmp_path / "out"),
        "--manifest",
        str(manifest),
        "--device",
        "cpu",
        "--candidate-n",
        "8",
        "--random-repeats",
        "2",
    ]
    first = run(parse_args([*options, "--random-repeats", "1", "--method", "random"]))[0]
    first_summary = (first / "summary.json").stat().st_mtime_ns
    dirs = run(parse_args([*options, "--method", "all"]))
    assert dirs[0] == first and (first / "summary.json").stat().st_mtime_ns == first_summary
    assert len(set(dirs)) == 5
    prior_contents = {}
    cache_roots = set()
    for path in dirs:
        rows = json.loads((path / "per_case.json").read_text())["rows"]
        summary = json.loads((path / "summary.json").read_text())
        metadata = json.loads((path / "provenance.json").read_text())["extra"]
        assert metadata["query_ids"] == [r["case_id"] for r in rows] == ["8"]
        assert len(metadata["segmenter_sha256"]) == len(metadata["encoder_sha256"]) == 64
        cache_roots.add(metadata["bank_cache"])
        row = rows[0]
        assert len(set(row["support_ids"])) == row["k"] == 2
        assert set(row["support_ids"]) <= {str(i) for i in range(8)}
        assert row["segmentation_calls"] == (2 if row["method"] == "dual_similarity_adapted" else 1)
        assert summary["dice"]["mean"] == summary["iou"]["mean"] == 1
        assert summary["hd95"]["mean"] == 0
        prior_contents[path] = (path / "per_case.json").read_bytes()
    assert len(cache_roots) == 1
    cache = Path(cache_roots.pop())
    timestamps = {p: p.stat().st_mtime_ns for p in cache.glob("*.pt")}
    assert len(timestamps) == 8
    new = run(parse_args([*options, "--method", "knn", "--k", "8"]))
    assert json.loads((new[0] / "per_case.json").read_text())["rows"][0]["k"] == 8
    assert {p: p.stat().st_mtime_ns for p in cache.glob("*.pt")} == timestamps
    assert run(parse_args([*options, "--method", "all"])) == dirs
    (first / "summary.json").unlink()
    with pytest.raises(FileExistsError, match="Incomplete or mismatched"):
        run(parse_args([*options, "--method", "all"]))
    assert not (first / "summary.json").exists()
    for path in dirs:
        assert (path / "per_case.json").read_bytes() == prior_contents[path]
    with pytest.raises(ValueError, match="requested K=9"):
        run(parse_args([*options, "--method", "random", "--k", "9", "--candidate-n", "9"]))
