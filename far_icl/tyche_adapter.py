"""Tyche-TS adapter for the official mariannerakic/Tyche implementation."""

import time

import torch

from .segmentation import Segmenter


class TycheSegmenter(Segmenter):
    def __init__(self, cfg):
        from tyche import tychets

        self.device = torch.device(cfg["device"])
        self.kind, self.samples = "tyche", cfg["tyche_samples"]
        if self.samples < 2:
            raise ValueError("tyche_samples must be >=2")
        self.model = tychets(pretrained=not bool(cfg["segmenter_weights"]))
        if cfg["segmenter_weights"]:
            state = torch.load(cfg["segmenter_weights"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.model.to(self.device).eval().requires_grad_(False)
        self.reset_stats()

    @torch.no_grad()
    def sample(self, query, supports, count):
        if not supports:
            raise ValueError("Tyche requires support cases")
        self.sync()
        start = time.perf_counter()
        sx = torch.stack([s["image"] for s in supports])[None].to(self.device)
        sy = torch.stack([s["mask"] for s in supports])[None].to(self.device)
        target = query.to(self.device)[None, None].expand(1, count, -1, -1, -1)
        # Explicit official shapes: B,S,1,H,W and B,K,1,H,W.
        logits = self.model(
            support_images=sx, support_labels=sy, target_image=target, noise_image=torch.randn_like(target)
        )
        self.sync()
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.support_evaluations += len(supports)
        return logits.sigmoid()[0]

    def __call__(self, query, supports):
        return self.sample(query, supports, self.samples).mean(0)
