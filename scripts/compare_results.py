#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from far_icl.report import paired_identity_bootstrap

p = argparse.ArgumentParser(description="Identity-aware paired Dice comparison")
p.add_argument("a", help="First per_case.json")
p.add_argument("b", help="Second per_case.json")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--samples", type=int, default=2000)
args = p.parse_args()
a, b = [json.loads(Path(path).read_text()) for path in (args.a, args.b)]
if a["manifest_hash"] != b["manifest_hash"]:
    p.error("Results use different manifests")
scope_a, scope_b = a.get("identity_scope", "patient"), b.get("identity_scope", "patient")
if scope_a != scope_b:
    p.error("Results use different identity scopes")
print(json.dumps(paired_identity_bootstrap(a["rows"], b["rows"], scope_a, args.seed, args.samples), indent=2))
