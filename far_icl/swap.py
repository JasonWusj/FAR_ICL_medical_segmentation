"""Single-swap actions on a fixed-size in-context support set."""

import torch
import torch.nn.functional as F
from torch import nn

from .features import pair_features
from .retrieval import ranking_loss


def swap_actions(selected, n):
    """Return (outgoing, incoming) candidate indices in a deterministic order."""
    available = sorted(set(range(n)).difference(selected))
    return [(outgoing, incoming) for outgoing in sorted(selected) for incoming in available]


def swapped_set(selected, action):
    outgoing, incoming = action
    return tuple(sorted(incoming if item == outgoing else item for item in selected))


def transition_targets(before, after, truth, grid):
    """[region, event], normalized by all pixels in each equal-area grid cell."""
    old, new, fg = before > 0.5, after > 0.5, truth > 0.5
    became_fg, became_bg = ~old & new, old & ~new
    events = torch.cat([became_fg & fg, became_bg & ~fg, became_bg & fg, became_fg & ~fg, old == new])
    return F.adaptive_avg_pool2d(events.float(), (grid, grid)).flatten(1).T


def swap_inputs(query, features, state, selected, actions, feature_mode):
    """Build [action, region, feature] inputs from the currently observed state."""
    pairs = torch.stack([pair_features(query, candidate, feature_mode) for candidate in features])
    keys = ["global_feature"]
    if feature_mode in {"shape", "full"}:
        keys.append("shape")
    if feature_mode in {"boundary", "full"}:
        keys.append("boundary")
    if feature_mode in {"failure", "full"}:
        keys.append("hard")
    supports = torch.stack([torch.cat([candidate[key] for key in keys]) for candidate in features])
    outgoing, incoming = torch.tensor(actions, device=pairs.device, dtype=torch.long).reshape(-1, 2).T
    kept = (supports[sorted(selected)].sum(0) - supports[outgoing]) / max(len(selected) - 1, 1)
    action_features = torch.cat([pairs[incoming], supports[outgoing], kept], dim=1)
    return torch.cat(
        [action_features[:, None].expand(-1, len(state), -1), state[None].expand(len(actions), -1, -1)],
        dim=-1,
    )


class SwapRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.transition_head = nn.Linear(hidden_dim // 2, 5)
        self.gain_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, features):
        regions = self.encoder(features)
        return self.transition_head(regions), self.gain_head(regions.mean(dim=1)).squeeze(-1)


def swap_loss(outputs, transitions, gains, transition_weight=1.0, ranking_weight=1.0):
    """Train on all actions belonging to a single query and current support set."""
    logits, predicted = outputs
    loss = F.smooth_l1_loss(predicted, gains)
    if ranking_weight:
        loss = loss + ranking_weight * ranking_loss(predicted, gains, "pairwise", regression_weight=0.0)
    if transition_weight:
        loss = loss - transition_weight * (transitions * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    return loss


def choose_swap(predicted_gains, threshold):
    if predicted_gains.numel() == 0:
        return None
    index = int(predicted_gains.argmax())
    return index if predicted_gains[index] > threshold else None
