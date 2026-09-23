#!/usr/bin/env python3
"""Build an explicitly image-level ISIC 2018 Task 1 exploratory manifest.

The official Task 1 archives do not provide the patient mapping required by the
patient-level protocol. This manifest must be run with identity_scope=image.
"""

import argparse
import csv
import random
from pathlib import Path


EXPECTED = {"Training": 2594, "Validation": 100, "Test": 1000}
FIELDS = (
    "case_id",
    "patient_id",
    "image_path",
    "mask_path",
    "split",
    "task",
    "domain",
    "spacing_y",
    "spacing_x",
)


def paired_cases(root, source, split):
    image_dir = root / f"ISIC2018_Task1-2_{source}_Input"
    mask_dir = root / f"ISIC2018_Task1_{source}_GroundTruth"
    images = sorted(image_dir.glob("*.jpg"))
    masks = sorted(mask_dir.glob("*_segmentation.png"))
    if len(images) != EXPECTED[source] or len(masks) != EXPECTED[source]:
        raise ValueError(
            f"{source}: expected {EXPECTED[source]} images and masks; "
            f"found {len(images)} images and {len(masks)} masks"
        )
    image_ids = {p.stem for p in images}
    mask_ids = {p.name.removesuffix("_segmentation.png") for p in masks}
    if image_ids != mask_ids:
        raise ValueError(f"{source}: image/mask IDs do not match")
    return [
        dict(
            case_id=image.stem,
            patient_id=f"image:{image.stem}",
            image_path=str(image.resolve()),
            mask_path=str((mask_dir / f"{image.stem}_segmentation.png").resolve()),
            split=split,
            task="lesion",
            domain="isic2018",
            spacing_y=1,
            spacing_x=1,
        )
        for image in images
    ]


def select(rows, limit, seed):
    if limit is None:
        return rows
    if limit < 1 or limit > len(rows):
        raise ValueError(f"Limit must be 1..{len(rows)}")
    return sorted(random.Random(seed).sample(rows, limit), key=lambda row: row["case_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument(
        "--include-test", action="store_true", help="Include the official 1000-image Task 1 test set"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    train = select(paired_cases(root, "Training", "train"), args.train_limit, args.seed)
    val = select(paired_cases(root, "Validation", "val"), args.val_limit, args.seed)
    test = paired_cases(root, "Test", "test") if args.include_test else []
    if len(train) < 2:
        parser.error("At least two training images are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(train + val + test)
    print(f"Wrote {output}: {len(train)} train, {len(val)} val, {len(test)} test images")
    print("Exploratory IMAGE-level identity only. Do not report patient-level validation.")


if __name__ == "__main__":
    main()
