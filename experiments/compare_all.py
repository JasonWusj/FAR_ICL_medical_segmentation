"""Collect completed test runs and compare each with the shared KNN baseline."""

import argparse
import csv
import json
from pathlib import Path

from far_icl.config import write_json
from far_icl.report import paired_identity_bootstrap


RUNS = {
    "universeg_knn": "same_backbone/universeg_knn",
    "far_icl_repair": "same_backbone/far_icl_repair",
    "dual_similarity_adapted": "same_backbone/dual_similarity_adapted",
    "tyche": "published_icl/tyche",
    "ires_s3_adapted": "published_icl/ires_s3_adapted",
    "iris_external": "published_icl/iris_external",
    "unet": "supervised/unet",
    "nnunet": "supervised/nnunet",
}


def collect(root, seed, samples):
    data = {}
    missing = []
    for label, subdir in RUNS.items():
        files = sorted((root / subdir / "results").glob("*_test_*/per_case.json"))
        if len(files) != 1:
            if not files:
                missing.append(label)
                continue
            raise ValueError(f"Multiple test results for {label}; select one explicitly: {files}")
        data[label] = (files[0], json.loads(files[0].read_text()))
    if "universeg_knn" not in data:
        raise ValueError("Run same_backbone/universeg_knn/run.sh first")
    baseline = data["universeg_knn"][1]
    rows = []
    paired = {}
    for label, (path, result) in data.items():
        if result["manifest_hash"] != baseline["manifest_hash"]:
            raise ValueError(f"Manifest mismatch: {label}")
        if result["identity_scope"] != baseline["identity_scope"]:
            raise ValueError(f"Identity scope mismatch: {label}")
        cases = result["rows"]
        metric = lambda key: sum(float(r[key]) for r in cases) / len(cases)
        row = {
            "method": label,
            "n": len(cases),
            "dice": metric("dice"),
            "iou": metric("iou"),
            "isic_thresholded_jaccard_0_65": sum(
                r["iou"] if r["iou"] >= 0.65 else 0 for r in cases
            ) / len(cases),
            "source": str(path),
        }
        if label != "universeg_knn":
            comparison = paired_identity_bootstrap(
                baseline["rows"], cases, result["identity_scope"], seed, samples
            )
            row["delta_dice_vs_knn"] = comparison[
                "image_macro_delta" if result["identity_scope"] == "image" else "patient_macro_delta"
            ]
            row["ci95_low"], row["ci95_high"] = comparison["ci95"]
            paired[label] = comparison
        rows.append(row)
    target = root / "comparison"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "test_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "n", "dice", "iou",
                                                 "isic_thresholded_jaccard_0_65",
                                                 "delta_dice_vs_knn", "ci95_low", "ci95_high", "source"])
        writer.writeheader()
        writer.writerows(rows)
    write_json({"paired_vs_knn": paired, "missing": missing,
                "note": "Exploratory image-level comparison; test split has already been inspected."},
               target / "paired.json")
    print(target / "test_table.csv")
    print("Missing:", ", ".join(missing) if missing else "none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-base", default="/home/featurize/work/far_icl_paper_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    collect(Path(args.out_base), args.seed, args.samples)


if __name__ == "__main__":
    main()
