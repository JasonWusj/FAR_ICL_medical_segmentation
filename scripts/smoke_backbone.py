#!/usr/bin/env python3
"""Optional Linux smoke inference on explicitly chosen real cases; never auto-run."""

import argparse

import torch

from far_icl.config import load_config, seed_all
from far_icl.data import load_case, read_manifest
from far_icl.metrics import segmentation_metrics
from far_icl.segmentation import Segmenter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/isic.yaml")
    p.add_argument("--query", required=True, help="Query case_id in manifest")
    p.add_argument(
        "--support", action="append", required=True, help="Support case_id; repeat for multiple cases"
    )
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    seed_all(cfg["seed"])
    cases = {c.case_id: c for c in read_manifest(cfg["manifest"], cfg["identity_scope"])}
    q, support_cases = cases[args.query], [cases[s] for s in args.support]
    if len(set(args.support)) != len(args.support):
        p.error("Duplicate supports")
    if any(s.patient_id == q.patient_id or s.task != q.task or s.split != "train" for s in support_cases):
        p.error("Supports must be training cases of the same task, from different patients than query")
    if cfg["segmenter"] == "tyche":
        from far_icl.tyche_adapter import TycheSegmenter

        model = TycheSegmenter(cfg)
    else:
        model = Segmenter(cfg)
    data = load_case(q, cfg)
    with torch.no_grad():
        pred = model(data["image"], [load_case(s, cfg) for s in support_cases])
    if pred.shape != (1, 128, 128) or not torch.isfinite(pred).all():
        raise RuntimeError("Backbone returned invalid probabilities")
    print(segmentation_metrics(pred, data["mask"], data["spacing"], cfg["surface_tolerance"]))


if __name__ == "__main__":
    main()
