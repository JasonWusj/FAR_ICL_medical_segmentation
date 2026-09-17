"""CLI imports GPU dependencies only after parsing arguments."""

import argparse

METHODS = [
    "random",
    "knn",
    "shape",
    "failure",
    "utility",
    "mmr",
    "set",
    "adaptive",
    "adaptive_learned",
    "roles",
]


def main():
    parser = argparse.ArgumentParser(description="FAR-ICL medical segmentation")
    parser.add_argument(
        "command", choices=["check-data", "bank", "generate", "train", "evaluate", "oracle", "run", "report"]
    )
    parser.add_argument("--config", default="configs/isic.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--kind", choices=["utility", "marginal"], default="utility")
    parser.add_argument("--method", choices=METHODS, default="knn")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tag")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate cached supervision")
    args = parser.parse_args()
    from .config import load_config

    cfg = load_config(args.config, args.set)
    if args.command == "check-data":
        from .data import load_case, read_manifest

        cases = read_manifest(cfg["manifest"])
        for case in cases:
            load_case(case, cfg)
        print({s: sum(c.split == s for c in cases) for s in ("train", "val", "test")})
        return
    if args.command == "report":
        from .report import report

        report(cfg["output"])
        return
    from .pipeline import LEARNED, Pipeline
    from .training import checkpoint_path, train

    pipeline = Pipeline(cfg, bank_only=args.command == "bank")
    if args.command == "bank":
        print(pipeline.bank.root)
    elif args.command == "generate":
        pipeline.generate(args.kind, args.split, args.force)
    elif args.command == "train":
        print(train(pipeline, args.kind, args.resume))
    elif args.command == "oracle":
        pipeline.oracle(args.split)
    elif args.command == "evaluate":
        kind = "marginal" if args.method in {"set", "adaptive_learned"} else "utility"
        checkpoint = args.checkpoint or checkpoint_path(pipeline, kind)
        pipeline.evaluate(args.method, args.split, checkpoint, args.tag)
    elif args.command == "run":
        checkpoint = args.checkpoint
        if args.method in LEARNED and not checkpoint:
            kind = "marginal" if args.method in {"set", "adaptive_learned"} else "utility"
            pipeline.generate(kind, "train", args.force)
            pipeline.generate(kind, "val", args.force)
            checkpoint = train(pipeline, kind, args.resume)
        pipeline.evaluate(args.method, args.split, checkpoint, args.tag)


if __name__ == "__main__":
    main()
