"""Score externally produced binary mask PNGs on the common 128x128 evaluation grid."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from far_icl.config import load_config
from far_icl.data import load_case, manifest_signature, read_manifest
from far_icl.metrics import segmentation_metrics

from .result_io import save_results


def score(cfg, pred_dir, method, split):
    pred_dir = Path(pred_dir)
    cases = read_manifest(cfg["manifest"], cfg["identity_scope"])
    manifest_hash = manifest_signature(cases)
    selected = [c for c in cases if c.split == split]
    if not selected:
        raise ValueError(f"No {split} cases")
    expected = {f"{c.case_id}.png" for c in selected}
    found = {p.name for p in pred_dir.glob("*.png")}
    missing, extra = expected - found, found - expected
    if missing or extra:
        raise ValueError(f"Prediction IDs mismatch: {len(missing)} missing, {len(extra)} extra; "
                         f"examples={sorted(missing)[:3]}, {sorted(extra)[:3]}")
    rows = []
    for case in tqdm(selected, desc=f"Score {method}/{split}"):
        data = load_case(case, cfg)
        with Image.open(pred_dir / f"{case.case_id}.png") as im:
            mask = np.asarray(im.convert("L"))
        if mask.shape != (128, 128):
            raise ValueError(f"Expected 128x128 prediction: {case.case_id}, got {mask.shape}")
        pred = torch.from_numpy((mask > 0).astype(np.float32))[None]
        rows.append(dict(case_id=case.case_id, patient_id=case.patient_id, task=case.task,
                         domain=case.domain, method=method, split=split, k=0,
                         **segmentation_metrics(pred, data["mask"], data["spacing"], cfg["surface_tolerance"])))
    return save_results(cfg, manifest_hash, method, split, rows, extra={"pred_dir": str(pred_dir)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isic.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    score(load_config(args.config, args.set), args.pred_dir, args.method, args.split)


if __name__ == "__main__":
    main()
