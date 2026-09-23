"""Regional repair/harm supervision and complementary correction selection.

All query ground truth is confined to ``correction_targets``. Coverage is a
surrogate built from single-support predictions, not measured joint utility;
it is neither a submodularity claim nor a safety guarantee.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .features import boundary, pair_features


def _finite(name, value):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")


def _map(name, value):
    _finite(name, value)
    if value.ndim != 3 or value.shape[0] != 1 or min(value.shape[1:]) < 1:
        raise ValueError(f"{name} must have shape [1, H, W]")


def _grid(grid, height, width):
    if (
        isinstance(grid, bool)
        or not isinstance(grid, int)
        or grid < 1
        or 128 % grid
        or height % grid
        or width % grid
    ):
        raise ValueError("grid must be a positive factor of 128 and both map dimensions")


def _regions(value, grid):
    """Return row-major region means, with one column per input channel."""
    return F.adaptive_avg_pool2d(value.unsqueeze(0), (grid, grid))[0].flatten(1).T


def regional_state(fmap, prediction, uncertainty, grid):
    """Build [grid**2, C+5] label-free state; final coordinates are (y, x).

    Coordinates locate region centers on [0, 1]. The other five entries follow
    the normalized pooled image feature: probability mean, uncertainty mean,
    boundary density, y coordinate, and x coordinate.
    """
    _finite("fmap", fmap)
    _map("prediction", prediction)
    _map("uncertainty", uncertainty)
    if fmap.ndim != 3 or min(fmap.shape) < 1 or not fmap.is_floating_point():
        raise ValueError("fmap must be a floating tensor with shape [C, h, w]")
    if prediction.shape != uncertainty.shape:
        raise ValueError("prediction and uncertainty must have identical shapes")
    if fmap.device != prediction.device or prediction.device != uncertainty.device:
        raise ValueError("regional_state tensors must be on the same device")
    if ((prediction < 0) | (prediction > 1)).any() or (uncertainty < 0).any():
        raise ValueError("prediction must be in [0, 1] and uncertainty must be nonnegative")
    height, width = prediction.shape[-2:]
    _grid(grid, height, width)
    spatial = F.interpolate(fmap[None], (height, width), mode="bilinear", align_corners=False)[0]
    pooled = F.normalize(_regions(spatial, grid), dim=1)
    prediction = prediction.to(dtype=fmap.dtype)
    uncertainty = uncertainty.to(dtype=fmap.dtype)
    centers = (torch.arange(grid, device=fmap.device, dtype=fmap.dtype) + 0.5) / grid
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    return torch.cat(
        [
            pooled,
            _regions(prediction, grid),
            _regions(uncertainty, grid),
            _regions(boundary(prediction).to(dtype=fmap.dtype), grid),
            yy.flatten()[:, None],
            xx.flatten()[:, None],
        ],
        dim=1,
    )


def correction_targets(initial, candidate, truth, grid):
    """Return repair/harm fractions [R, 2], each divided by ALL region pixels.

    The two events are disjoint, so their sum cannot exceed one. Pixels correct
    before and after, or wrong before and after, belong to the unchanged class.
    Only this training-label helper accepts query ground truth.
    """
    for name, value in (("initial", initial), ("candidate", candidate), ("truth", truth)):
        _map(name, value)
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name} must be in [0, 1]")
    if initial.shape != candidate.shape or initial.shape != truth.shape:
        raise ValueError("initial, candidate, and truth must have identical shapes")
    if initial.device != candidate.device or initial.device != truth.device:
        raise ValueError("correction target tensors must be on the same device")
    _grid(grid, *initial.shape[-2:])
    was_correct = (initial > 0.5) == (truth > 0.5)
    now_correct = (candidate > 0.5) == (truth > 0.5)
    events = torch.cat([~was_correct & now_correct, was_correct & ~now_correct], dim=0).float()
    return _regions(events, grid)


def correction_inputs(query, candidates, state, feature_mode):
    """Assemble [N, R, D] on demand instead of storing expanded pair features."""
    _finite("state", state)
    if state.ndim != 2 or min(state.shape) < 1:
        raise ValueError("state must be a nonempty [R, C+5] tensor")
    if not candidates:
        raise ValueError("At least one correction candidate is required")
    if feature_mode not in {"image", "shape", "boundary", "failure", "full"}:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    pairs = torch.stack([pair_features(query, candidate, feature_mode) for candidate in candidates])
    _finite("pair features", pairs)
    if pairs.device != state.device:
        raise ValueError("pair features and regional state must be on the same device")
    state = state.to(dtype=pairs.dtype)
    return torch.cat(
        [pairs[:, None, :].expand(-1, len(state), -1), state[None].expand(len(pairs), -1, -1)],
        dim=-1,
    )


class CorrectionRanker(nn.Module):
    """Predict repair, harm, and implicit unchanged mass independently per region."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("Positive input_dim and hidden_dim >= 2 are required")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, features):
        _finite("correction inputs", features)
        if features.ndim != 3 or features.shape[-1] != self.net[0].in_features:
            raise ValueError("correction inputs must have shape [N, R, input_dim]")
        return self.net(features).softmax(dim=-1)[..., :2]


def _corrections(name, value):
    _finite(name, value)
    if value.ndim != 3 or value.shape[-1] != 2 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N, R, 2] with nonempty regions")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if (value < 0).any() or (value.sum(-1) > 1 + 1e-6).any():
        raise ValueError(f"{name} repair/harm probabilities must be nonnegative and sum to <= 1")


def _nonnegative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def correction_loss(pred, target, kind="pairwise", regression_weight=1.0, harm_weight=1.0):
    """One query per call: regional regression plus candidate-level ranking."""
    _corrections("prediction", pred)
    _corrections("target", target)
    if pred.shape != target.shape or pred.device != target.device or pred.shape[0] < 1:
        raise ValueError("prediction and target must share shape/device with nonempty candidates")
    regression_weight = _nonnegative("regression_weight", regression_weight)
    if regression_weight == 0:
        raise ValueError("regression_weight must be positive")
    harm_weight = _nonnegative("harm_weight", harm_weight)
    regression = regression_weight * F.mse_loss(pred, target)
    if kind == "regression":
        return regression
    scores = (pred[..., 0] - harm_weight * pred[..., 1]).mean(dim=1)
    utilities = (target[..., 0] - harm_weight * target[..., 1]).mean(dim=1)
    if kind == "pairwise":
        valid = utilities[:, None] - utilities[None, :] > 1e-5
        ranking = (
            F.softplus(-(scores[:, None] - scores[None, :])[valid]).mean()
            if valid.any()
            else scores.sum() * 0
        )
    elif kind == "listwise":
        ranking = -(F.softmax(utilities / 0.1, dim=0) * F.log_softmax(scores / 0.1, dim=0)).sum()
    else:
        raise ValueError(f"Unknown loss: {kind}")
    return regression + ranking


@torch.no_grad()
def select_corrections(pred, region_uncertainty, cfg, method):
    """Select supports using predicted repair/harm only, with no query labels.

    ``repair`` orders independent net gains. ``repair_cover`` greedily optimizes
    weighted regional max-repair minus max-harm at fixed K. ``repair_adaptive``
    stops after min_k when the best remaining coverage gain is <= the cost.
    Every trace entry is the actual change of this same coverage surrogate,
    including for ``repair``; fixed-K selection can have negative increments.
    """
    _corrections("prediction", pred)
    _finite("region_uncertainty", region_uncertainty)
    if region_uncertainty.shape != (pred.shape[1],) or (region_uncertainty < 0).any():
        raise ValueError("region_uncertainty must be nonnegative with shape [R]")
    if method not in {"repair", "repair_cover", "repair_adaptive"}:
        raise ValueError(f"Unknown correction method: {method}")
    harm_weight = _nonnegative("correction_harm_weight", cfg["correction_harm_weight"])
    uncertainty_weight = _nonnegative("correction_uncertainty_weight", cfg["correction_uncertainty_weight"])
    cost = _nonnegative("correction_cost", cfg["correction_cost"])
    for key in ("k", "max_k", "min_k"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["min_k"] > cfg["max_k"]:
        raise ValueError("min_k exceeds max_k")
    n, regions, _ = pred.shape
    if n == 0:
        return [], []
    uncertainty = region_uncertainty.to(device=pred.device, dtype=pred.dtype)
    weights = 1 + uncertainty_weight * uncertainty / uncertainty.mean().clamp_min(1e-8)
    weights = weights / weights.sum()
    limit = min(cfg["max_k"] if method == "repair_adaptive" else cfg["k"], n)
    current_repair = pred.new_zeros(regions)
    current_harm = pred.new_zeros(regions)
    value = pred.new_zeros(())
    selected, trace = [], []
    if method == "repair":
        independent = ((pred[..., 0] - harm_weight * pred[..., 1]) * weights).sum(dim=1)
        order = independent.argsort(descending=True, stable=True).tolist()
    for step in range(limit):
        next_repair = torch.maximum(current_repair[None], pred[..., 0])
        next_harm = torch.maximum(current_harm[None], pred[..., 1])
        next_value = ((next_repair - harm_weight * next_harm) * weights).sum(dim=1)
        gains = next_value - value
        if method == "repair":
            chosen = order[step]
        else:
            gains[selected] = -torch.inf
            chosen = int(gains.argmax())
        gain = float(gains[chosen])
        if method == "repair_adaptive" and len(selected) >= cfg["min_k"] and gain <= cost:
            break
        selected.append(chosen)
        trace.append(gain)
        current_repair, current_harm = next_repair[chosen], next_harm[chosen]
        value = next_value[chosen]
    return selected, trace
