#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from far_icl.report import paired_patient_bootstrap

p = argparse.ArgumentParser(description="Patient-clustered paired Dice comparison")
p.add_argument("a", help="First per_case.json")
p.add_argument("b", help="Second per_case.json")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--samples", type=int, default=2000)
args = p.parse_args()
a, b = [json.loads(Path(path).read_text()) for path in (args.a, args.b)]
if a["manifest_hash"] != b["manifest_hash"]:
    p.error("Results use different manifests")
print(json.dumps(paired_patient_bootstrap(a["rows"], b["rows"], args.seed, args.samples), indent=2))
