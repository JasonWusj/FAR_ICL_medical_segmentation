"""Frozen backbone interface. All public predictions are probabilities."""

import time

import torch


class Segmenter:
    def __init__(self, cfg):
        self.device = torch.device(cfg["device"])
        self.kind = cfg["segmenter"]
        if self.kind != "universeg":
            raise ValueError("Use universeg, or the documented Tyche adapter configuration")
        from universeg import universeg

        self.model = universeg(pretrained=not bool(cfg["segmenter_weights"]))
        if cfg["segmenter_weights"]:
            self.model.load_state_dict(
                torch.load(cfg["segmenter_weights"], map_location="cpu", weights_only=True)
            )
        self.model.to(self.device).eval().requires_grad_(False)
        self.reset_stats()

    def reset_stats(self):
        self.calls, self.seconds, self.support_evaluations = 0, 0.0, 0

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def __call__(self, query, supports):
        if not supports:
            raise ValueError("UniverSeg requires at least one support")
        self.sync()
        start = time.perf_counter()
        images = torch.stack([s["image"] for s in supports])[None].to(self.device)
        masks = torch.stack([s["mask"] for s in supports])[None].to(self.device)
        result = self.model(query[None].to(self.device), images, masks).sigmoid()[0]
        self.sync()
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.support_evaluations += len(supports)
        return result


def estimate_uncertainty(segmenter, query, candidates, initial, cfg, rng):
    mode = cfg["uncertainty"]
    if mode == "none":
        return torch.zeros_like(query, device=segmenter.device)
    predictions = []
    if mode == "context":
        k = min(cfg["initial_k"], len(candidates))
        for _ in range(cfg["uncertainty_samples"]):
            ids = rng.choice(len(candidates), k, replace=False)
            predictions.append(segmenter(query, [candidates[i] for i in ids]))
    elif mode == "tta":
        # Exact inverses; query and supports undergo the same spatial transform.
        for dims in ((), (-1,), (-2,), (-2, -1)):
            q = query.flip(dims) if dims else query
            support = [
                {
                    "image": s["image"].flip(dims) if dims else s["image"],
                    "mask": s["mask"].flip(dims) if dims else s["mask"],
                }
                for s in initial
            ]
            pred = segmenter(q, support)
            predictions.append(pred.flip(dims) if dims else pred)
    elif mode == "tyche":
        if not hasattr(segmenter, "sample"):
            raise ValueError("Tyche uncertainty needs segmenter=tyche with official code and weights")
        predictions = list(segmenter.sample(query, initial, cfg["tyche_samples"]))
    else:
        raise ValueError(mode)
    return torch.stack(predictions).var(0, unbiased=False)
