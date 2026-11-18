"""Single-swap logic checks on synthetic tensors; no weights or medical data."""

import pytest
import torch

from far_icl.metrics import dice
from far_icl.swap import (
    SwapRanker,
    choose_swap,
    swap_actions,
    swap_inputs,
    swap_loss,
    swapped_set,
    transition_targets,
)


def test_single_swap_neighborhood_and_abstention():
    initial = [3, 1]
    actions = swap_actions(initial, 12)
    neighbors = {swapped_set(initial, action) for action in actions}
    assert len(neighbors) == len(actions) == 20
    assert all(len(pair) == 2 and len(set(pair).intersection(initial)) == 1 for pair in neighbors)
    assert actions == swap_actions(initial[::-1], 12)
    assert neighbors == {swapped_set(initial[::-1], action) for action in actions}
    assert choose_swap(torch.tensor([0.125, -0.1]), 0.125) is None
    assert choose_swap(torch.tensor([0.125, 0.25]), 0.125) == 1
    assert choose_swap(torch.empty(0), 0.0) is None


def test_transition_fractions_recover_actual_dice_change():
    # Region 0 fixes a missed foreground, region 1 gains a false positive,
    # region 2 both removes a false positive and loses a true positive.
    old = torch.tensor([[[0, 0, 0, 1], [1, 1, 0, 0]]]).float()
    new = torch.tensor([[[1, 1, 1, 1], [0, 0, 0, 0]]]).float()
    truth = torch.tensor([[[1, 1, 0, 1], [0, 1, 0, 1]]]).float()
    proportions = transition_targets(old, new, truth, 2)
    expected = [[1, 0, 0, 0, 0], [0, 0, 0, 0.5, 0.5], [0, 0.5, 0.5, 0, 0], [0, 0, 0, 0, 1]]
    torch.testing.assert_close(proportions, torch.tensor(expected))
    torch.testing.assert_close(proportions.sum(1), torch.ones(4))
    fix_fn, fix_fp, add_fn, add_fp = (proportions * 2).sum(0)[:4]
    tp = ((old == 1) & (truth == 1)).sum() + fix_fn - add_fn
    fp = ((old == 1) & (truth == 0)).sum() - fix_fp + add_fp
    fn = ((old == 0) & (truth == 1)).sum() - fix_fn + add_fn
    reconstructed = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    assert float(reconstructed) - dice(old, truth) == pytest.approx(dice(new, truth) - dice(old, truth))


def test_inputs_preserve_set_symmetry_and_distinguish_outgoing_members():
    candidates = [{"global_feature": vector} for vector in torch.eye(4)]
    selected = [2, 0]
    state = torch.ones(4, 3)
    actions = swap_actions(selected, len(candidates))
    x = swap_inputs(candidates[0], candidates, state, selected, actions, "image")
    torch.testing.assert_close(
        x, swap_inputs(candidates[0], candidates, state, selected[::-1], actions, "image")
    )
    assert not torch.equal(x[actions.index((0, 1))], x[actions.index((2, 1))])
    model = SwapRanker(x.shape[-1], hidden_dim=8)
    logits, gains = model(x)
    assert logits.shape == (4, 4, 5)
    assert gains.shape == (4,)
    targets = torch.zeros_like(logits)
    targets[..., 4] = 1
    labels = torch.tensor([0.25, -0.125, 0.0, 0.1])
    swap_loss((logits, gains), targets, labels, transition_weight=0).backward(retain_graph=True)
    assert model.transition_head.weight.grad is None
    assert model.gain_head.weight.grad is not None
    swap_loss((logits, gains), targets, labels).backward()
    assert torch.isfinite(model.transition_head.weight.grad).all()
