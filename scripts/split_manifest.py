#!/usr/bin/env python3
"""Split an explicit metadata CSV by patient; no medical identity inference."""

import argparse
import csv
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "input", help="CSV with case_id,patient_id,image_path,mask_path,task[,domain,spacing_y,spacing_x]"
    )
    p.add_argument("output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    args = p.parse_args()
    if not 0 < args.val < 1 or not 0 < args.test < 1 or args.val + args.test >= 1:
        p.error("val and test fractions must be positive with sum < 1")
    source, dest = Path(args.input).resolve(), Path(args.output).resolve()
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if not rows or not all(r.get("patient_id") for r in rows):
        p.error("Every case needs an explicit globally unique patient_id")
    patients = sorted({r["patient_id"] for r in rows})
    if len(patients) < 3:
        p.error("At least three patients are required")
    random.Random(args.seed).shuffle(patients)
    nv, nt = max(1, round(len(patients) * args.val)), max(1, round(len(patients) * args.test))
    if nv + nt >= len(patients):
        p.error("Fractions leave no training patients")
    val, test = set(patients[:nv]), set(patients[nv : nv + nt])
    for row in rows:
        row["split"] = "val" if row["patient_id"] in val else "test" if row["patient_id"] in test else "train"
        for key in ("image_path", "mask_path"):
            # Preserve paths when output manifest is in another directory.
            row[key] = str((source.parent / row[key]).resolve())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields if "split" in fields else fields + ["split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases / {len(patients)} patients to {dest}")


if __name__ == "__main__":
    main()
