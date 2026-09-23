"""Synthetic correction logic checks; no pretrained model or dataset required."""

import pytest
import torch
import torch.nn.functional as F

from far_icl.correction import (
    CorrectionRanker,
    correction_inputs,
    correction_loss,
    correction_targets,
    regional_state,
    select_corrections,
)
from far_icl.features import pair_features


def feature(vector):
    x = F.normalize(torch.tensor(vector, dtype=torch.float32), dim=0)
    return dict(global_feature=x, boundary=x, hard=x, shape=torch.ones(10), uncertainty=torch.tensor([0.02]))


def selection_config(**overrides):
    return (
        dict(
            k=2,
            max_k=3,
            min_k=1,
            correction_harm_weight=1.0,
            correction_uncertainty_weight=2.0,
            correction_cost=0.0,
        )
        | overrides
    )


def test_targets_use_entire_region_denominator_for_both_events():
    # One wrong pixel is repaired; one of the three correct pixels is damaged.
    truth = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    initial = torch.zeros_like(truth)
    candidate = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    target = correction_targets(initial, candidate, truth, grid=1)
    assert target.shape == (1, 2)
    assert torch.equal(target, torch.tensor([[0.25, 0.25]]))


def test_target_regions_are_row_major_and_unchanged_has_zero_mass():
    truth = torch.zeros(1, 4, 4)
    initial = torch.zeros_like(truth)
    initial[:, :2, :2] = 1
    candidate = torch.zeros_like(truth)
    candidate[:, 2:, 2:] = 1
    expected = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    assert torch.equal(correction_targets(initial, candidate, truth, 2), expected)


def test_regional_state_shape_norm_means_and_center_coordinates():
    fmap = torch.tensor([[[3.0]], [[4.0]]])
    prediction = torch.full((1, 4, 4), 0.25)
    uncertainty = torch.arange(16).reshape(1, 4, 4).float()
    state = regional_state(fmap, prediction, uncertainty, grid=2)
    assert state.shape == (4, 7)
    assert torch.allclose(state[:, :2], torch.tensor([[0.6, 0.8]]).expand(4, -1))
    assert torch.equal(state[:, 2], torch.full((4,), 0.25))
    assert torch.equal(state[:, 3], torch.tensor([2.5, 4.5, 10.5, 12.5]))
    assert torch.equal(state[:, 4], torch.zeros(4))
    assert torch.equal(state[:, 5:], torch.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]))
    assert torch.isfinite(state).all()


@pytest.mark.parametrize("mode", ["image", "shape", "boundary", "failure", "full"])
def test_inputs_combine_same_pairs_with_each_regional_state(mode):
    query, candidates = feature([1, 0]), [feature([0, 1]), feature([1, 1])]
    state = regional_state(torch.ones(2, 1, 1), torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), 2)
    inputs = correction_inputs(query, candidates, state, mode)
    pair = pair_features(query, candidates[0], mode)
    assert inputs.shape == (2, 4, pair.numel() + state.shape[1])
    assert torch.equal(inputs[0, :, : pair.numel()], pair[None].expand(4, -1))
    assert torch.equal(inputs[1, :, pair.numel() :], state)


def test_ranker_probabilities_shape_and_gradient_are_finite():
    model = CorrectionRanker(input_dim=8, hidden_dim=16)
    prediction = model(torch.ones(3, 4, 8))
    assert prediction.shape == (3, 4, 2)
    assert (prediction >= 0).all()
    assert (prediction.sum(-1) <= 1).all()
    loss = correction_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_regression_loss_does_not_duplicate_candidate_level_mse():
    prediction = torch.tensor([[[0.2, 0.3]], [[0.5, 0.1]]], requires_grad=True)
    target = torch.tensor([[[0.1, 0.1]], [[0.8, 0.1]]])
    actual = correction_loss(prediction, target, "regression", regression_weight=2.5)
    assert torch.allclose(actual, 2.5 * F.mse_loss(prediction, target))


@pytest.mark.parametrize("kind", ["pairwise", "listwise"])
def test_ranking_prefers_correct_order_and_handles_ties(kind):
    target = torch.tensor([[[0.1, 0.0]], [[0.8, 0.0]]])
    assert correction_loss(target, target, kind) < correction_loss(target.flip(0), target, kind)
    tied = torch.tensor([[[0.2, 0.1]], [[0.2, 0.1]]], requires_grad=True)
    loss = correction_loss(tied, tied.detach(), kind)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(tied.grad).all()


def test_zero_and_uniform_uncertainty_are_uniform_weights():
    prediction = torch.tensor([[[0.8, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.6, 0.0]]])
    cfg = selection_config(correction_uncertainty_weight=10.0)
    zero_ids, zero_trace = select_corrections(prediction, torch.zeros(2), cfg, "repair")
    uniform_ids, uniform_trace = select_corrections(prediction, torch.ones(2), cfg, "repair")
    assert zero_ids == uniform_ids == [0, 1]
    assert zero_trace == pytest.approx([0.4, 0.3])
    assert uniform_trace == pytest.approx(zero_trace)


def test_uncertainty_weights_change_the_region_preference():
    prediction = torch.tensor([[[0.8, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.6, 0.0]]])
    chosen, _ = select_corrections(prediction, torch.tensor([0.0, 1.0]), selection_config(k=1), "repair")
    assert chosen == [1]


def test_coverage_selects_complementary_candidate_and_records_actual_gain():
    prediction = torch.tensor([[[0.8, 0.0], [0.0, 0.0]], [[0.7, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.6, 0.0]]])
    plain, plain_trace = select_corrections(prediction, torch.zeros(2), selection_config(), "repair")
    cover, cover_trace = select_corrections(prediction, torch.zeros(2), selection_config(), "repair_cover")
    assert plain == [0, 1] and plain_trace == pytest.approx([0.4, 0.0])
    assert cover == [0, 2] and cover_trace == pytest.approx([0.4, 0.3])


def test_fixed_coverage_records_negative_gain_and_adaptive_honors_minimum():
    prediction = torch.tensor([[[0.1, 0.0]], [[0.0, 0.5]], [[0.0, 0.8]]])
    fixed, fixed_trace = select_corrections(prediction, torch.zeros(1), selection_config(), "repair_cover")
    adaptive, trace = select_corrections(prediction, torch.zeros(1), selection_config(), "repair_adaptive")
    forced, forced_trace = select_corrections(
        prediction, torch.zeros(1), selection_config(min_k=2), "repair_adaptive"
    )
    assert fixed == forced == [0, 1]
    assert fixed_trace == pytest.approx([0.1, -0.5])
    assert forced_trace == pytest.approx(fixed_trace)
    assert adaptive == [0] and trace == pytest.approx([0.1])


def test_all_negative_gains_still_honor_minimum_and_cost_stops_equality():
    negative = torch.tensor([[[0.0, 0.1]], [[0.0, 0.2]], [[0.0, 0.3]]])
    chosen, trace = select_corrections(negative, torch.zeros(1), selection_config(min_k=2), "repair_adaptive")
    assert chosen == [0, 1] and trace == pytest.approx([-0.1, -0.1])
    disjoint = torch.tensor([[[0.5, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.25, 0.0]]])
    chosen, _ = select_corrections(
        disjoint, torch.zeros(2), selection_config(correction_cost=0.125), "repair_adaptive"
    )
    assert chosen == [0]


@pytest.mark.parametrize("method", ["repair", "repair_cover", "repair_adaptive"])
def test_selection_is_unique_and_limited_by_available_candidates(method):
    prediction = torch.tensor([[[0.1, 0.0]], [[0.2, 0.0]]])
    chosen, trace = select_corrections(
        prediction, torch.zeros(1), selection_config(k=8, min_k=3, max_k=4), method
    )
    assert len(chosen) == len(set(chosen)) == len(trace) == 2
    assert select_corrections(prediction[:0], torch.zeros(1), selection_config(), method) == ([], [])


def test_invalid_shapes_nonfinite_values_and_probabilities_fail_explicitly():
    with pytest.raises(ValueError, match="factor"):
        regional_state(torch.zeros(2, 1, 1), torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), 3)
    with pytest.raises(ValueError, match="identical shapes"):
        correction_targets(torch.zeros(1, 4, 4), torch.zeros(1, 2, 2), torch.zeros(1, 4, 4), 1)
    with pytest.raises(ValueError, match="finite"):
        CorrectionRanker(8, 16)(torch.full((1, 1, 8), float("nan")))
    with pytest.raises(ValueError, match="sum"):
        select_corrections(torch.tensor([[[0.8, 0.8]]]), torch.zeros(1), selection_config(), "repair")
    with pytest.raises(ValueError, match="shape"):
        select_corrections(torch.zeros(1, 2, 2), torch.zeros(1), selection_config(), "repair")
    with pytest.raises(ValueError, match="finite"):
        select_corrections(torch.zeros(1, 1, 2), torch.tensor([float("inf")]), selection_config(), "repair")
