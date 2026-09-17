"""Heuristic, learned utility, MMR and conditional marginal selectors."""

import torch
import torch.nn.functional as F
from torch import nn

from .features import pair_features, set_features


class UtilityRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def ranking_loss(scores, targets, kind="pairwise", regression_weight=1.0):
    regression = F.mse_loss(scores, targets)
    if kind == "regression":
        return regression
    if kind == "listwise":
        ranking = -(F.softmax(targets / 0.1, dim=0) * F.log_softmax(scores / 0.1, dim=0)).sum()
    elif kind == "pairwise":
        # Each call contains ONE query and ONE selected-set state. Never compare unrelated queries.
        differences = targets[:, None] - targets[None, :]
        valid = differences > 1e-5
        ranking = (
            F.softplus(-(scores[:, None] - scores[None, :])[valid]).mean()
            if valid.any()
            else scores.sum() * 0
        )
    else:
        raise ValueError(f"Unknown loss: {kind}")
    # Calibration matters for marginal-gain stopping, not only relative ordering.
    return ranking + regression_weight * regression


def heuristic_scores(query, candidates, cfg, method):
    scores = []
    for candidate in candidates:
        value = cfg["alpha"] * (query["global_feature"] @ candidate["global_feature"])
        mode = cfg["feature_mode"]
        if method == "shape" or (method == "failure" and mode in {"shape", "full"}):
            value = value + cfg["gamma"] * F.cosine_similarity(query["shape"], candidate["shape"], dim=0)
        if method == "failure" and mode in {"boundary", "full"}:
            value = value + cfg["beta"] * (query["boundary"] @ candidate["boundary"])
        if method == "failure" and mode in {"failure", "full"}:
            value = value + cfg["delta"] * (query["hard"] @ candidate["hard"])
        scores.append(value)
    return torch.stack(scores)


@torch.no_grad()
def select(query, candidates, cfg, method, ranker=None):
    n = len(candidates)
    k = min(cfg["k"], n)
    gains = []
    if method == "adaptive":
        index = sum(float(query["uncertainty"]) >= t for t in cfg["adaptive_thresholds"])
        k = min(cfg["adaptive_ks"][index], cfg["max_k"], n)
        method = "mmr"
    if method in {"utility", "mmr"}:
        if ranker is None:
            raise ValueError("Utility and MMR require a trained utility checkpoint")
        scores = ranker(torch.stack([pair_features(query, c, cfg["feature_mode"]) for c in candidates]))
    elif method in {"set", "adaptive_learned"}:
        if ranker is None:
            raise ValueError("Set selection requires a marginal checkpoint")
        chosen = []
        limit = min(cfg["max_k"] if method == "adaptive_learned" else k, n)
        for _ in range(limit):
            available = [j for j in range(n) if j not in chosen]
            inputs = torch.stack(
                [
                    set_features(
                        query,
                        candidates[j],
                        [candidates[s] for s in chosen],
                        cfg["feature_mode"],
                        cfg["max_k"],
                    )
                    for j in available
                ]
            )
            predictions = ranker(inputs)
            best = int(predictions.argmax())
            gain = float(predictions[best])
            if method == "adaptive_learned" and len(chosen) >= cfg["min_k"] and gain < cfg["stop_epsilon"]:
                break
            chosen.append(available[best])
            gains.append(gain)
        return chosen, gains
    else:
        scores = heuristic_scores(query, candidates, cfg, method)
    if method != "mmr":
        return scores.topk(k).indices.tolist(), gains
    chosen = []
    vectors = torch.stack([c["global_feature"] for c in candidates])
    for _ in range(k):
        penalized = scores.clone()
        if chosen:
            penalized -= cfg["redundancy"] * (vectors @ vectors[chosen].T).max(1).values
            penalized[chosen] = -torch.inf
        chosen.append(int(penalized.argmax()))
    return chosen, gains
