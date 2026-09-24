"""Prepare an explicit ISIC train/val fold for official nnU-Net v2."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from far_icl.config import load_config, write_json
from far_icl.data import load_case, read_manifest


def prepare(cfg, dataset_id, split_only=False):
    out = Path(cfg["output"])
    dataset = f"Dataset{dataset_id}_ISIC2018"
    raw = out / "nnUNet_raw" / dataset
    preprocessed = out / "nnUNet_preprocessed" / dataset
    cases = read_manifest(cfg["manifest"], cfg["identity_scope"])
    train = [c.case_id for c in cases if c.split == "train"]
    val = [c.case_id for c in cases if c.split == "val"]
    test = [c.case_id for c in cases if c.split == "test"]
    if not train or not val or not test:
        raise ValueError("Expected nonempty train, val and test splits")
    if split_only:
        if not preprocessed.is_dir():
            raise FileNotFoundError(f"Run nnUNetv2_plan_and_preprocess first: {preprocessed}")
        write_json([{"train": train, "val": val}], preprocessed / "splits_final.json")
        return
    for name in ("imagesTr", "labelsTr", "imagesTs"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    for case in tqdm(cases, desc="Prepare nnU-Net PNGs"):
        image_dir = raw / ("imagesTs" if case.split == "test" else "imagesTr")
        image_path = image_dir / f"{case.case_id}_0000.png"
        label_path = raw / "labelsTr" / f"{case.case_id}.png"
        if image_path.exists() and (case.split == "test" or label_path.exists()):
            continue
        data = load_case(case, cfg)
        rgb = (data["rgb"].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(image_path)
        if case.split != "test":
            mask = (data["mask"][0].numpy() > 0.5).astype(np.uint8)
            Image.fromarray(mask, mode="L").save(label_path)
    write_json({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(train) + len(val),
        "file_ending": ".png",
        "name": "ISIC2018_Task1_exploratory_128px",
        "reference": "https://challenge.isic-archive.com/data/",
        "description": "Original train+validation with explicit fold; test has no training labels",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }, raw / "dataset.json")
    write_json({"train": train, "val": val, "test": test}, out / "split_ids.json")
    print(f"Prepared {dataset}: {len(train)} train, {len(val)} val, {len(test)} test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isic.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-id", type=int, default=901)
    parser.add_argument("--write-split-only", action="store_true")
    args = parser.parse_args()
    prepare(load_config(args.config, args.set), args.dataset_id, args.write_split_only)


if __name__ == "__main__":
    main()
