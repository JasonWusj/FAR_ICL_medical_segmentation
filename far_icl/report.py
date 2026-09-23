"""Export observed results; never populate unexecuted experiments with invented values."""

import csv
import json
from pathlib import Path

import numpy as np


def report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output)
    rows = []
    for file in sorted((root / "results").glob("*/summary.json")):
        summary = json.loads(file.read_text())
        row = {"run": file.parent.name}
        for key, value in summary.items():
            if isinstance(value, dict):
                row[key] = value["mean"]
            else:
                row[key] = value
        rows.append(row)
    if not rows:
        raise ValueError(f"No experiment results in {root}/results")
    keys = sorted({k for row in rows for k in row})
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if "dice" in r]
    if valid:
        fig, ax = plt.subplots(figsize=(max(8, len(valid) * 0.65), 5))
        ax.bar([r["run"] for r in valid], [r["dice"] for r in valid])
        ax.set_ylabel("Mean case Dice")
        ax.tick_params(axis="x", rotation=65)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(root / "dice_comparison.png", dpi=160)
        plt.close(fig)
    for file in (root / "results").glob("*/oracle.json"):
        oracle = json.loads(file.read_text())["rows"]
        labels = ["random_expected", "knn", "oracle", "worst"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, [np.mean([r[k] for r in oracle]) for k in labels])
        ax.set(ylabel="Single-support Dice", ylim=(0, 1), title="Same-pool oracle diagnostic")
        fig.tight_layout()
        fig.savefig(file.parent / "oracle_gap.png", dpi=160)
        plt.close(fig)
    print(root / "results.csv")


def paired_identity_bootstrap(rows_a, rows_b, identity_scope="patient", seed=42, samples=2000):
    """Case-aligned CI; cluster by patient or by image for exploratory runs."""
    if identity_scope not in {"patient", "image"}:
        raise ValueError("identity_scope must be patient or image")
    a, b = {r["case_id"]: r for r in rows_a}, {r["case_id"]: r for r in rows_b}
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical query cases")
    groups = {}
    for case_id, row in a.items():
        if row["patient_id"] != b[case_id]["patient_id"]:
            raise ValueError("Identity metadata mismatch")
        groups.setdefault(row["patient_id"], []).append(b[case_id]["dice"] - row["dice"])
    values = np.array([np.mean(v) for v in groups.values()])
    if not len(values):
        raise ValueError("Empty comparison")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    label = "patient" if identity_scope == "patient" else "image"
    return {
        "identity_scope": identity_scope,
        f"{label}_macro_delta": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        f"n_{label}s": len(values),
    }


def paired_patient_bootstrap(rows_a, rows_b, seed=42, samples=2000):
    """Backward-compatible patient-clustered comparison."""
    return paired_identity_bootstrap(rows_a, rows_b, "patient", seed, samples)
