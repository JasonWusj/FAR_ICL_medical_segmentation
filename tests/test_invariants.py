"""Synthetic logic tests, no model weights, downloads or dataset experiments."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from far_icl.data import read_manifest
from far_icl.features import pair_features, set_features, shape_descriptor
from far_icl.metrics import dice, segmentation_metrics
from far_icl.retrieval import ranking_loss, select


def feature(vector):
    x = torch.nn.functional.normalize(torch.tensor(vector, dtype=torch.float32), dim=0)
    return dict(global_feature=x, boundary=x, hard=x, shape=torch.ones(10), uncertainty=torch.tensor([0.02]))


def test_empty_metrics_are_explicit():
    empty = torch.zeros(1, 8, 8)
    assert dice(empty, empty) == 1
    assert segmentation_metrics(empty, empty, (1, 1), 2)["nsd"] == 1
    mismatch = segmentation_metrics(empty, torch.ones_like(empty), (1, 1), 2)
    assert mismatch["hd95"] is None
    assert mismatch["empty_mismatch"]


def test_ranking_prefers_higher_true_utility_and_handles_ties():
    target = torch.tensor([0.1, 0.9])
    good = ranking_loss(torch.tensor([0.1, 0.9]), target)
    bad = ranking_loss(torch.tensor([0.9, 0.1]), target)
    assert good < bad
    scores = torch.tensor([0.4, 0.4], requires_grad=True)
    loss = ranking_loss(scores, torch.tensor([0.4, 0.4]))
    loss.backward()
    assert torch.isfinite(loss)


def test_set_pool_is_permutation_invariant_and_has_fixed_dimensions():
    q, a, b = feature([1, 0]), feature([0, 1]), feature([1, 1])
    for mode in ("image", "shape", "boundary", "failure", "full"):
        one = set_features(q, a, [a, b], mode, 16)
        two = set_features(q, a, [b, a], mode, 16)
        assert torch.allclose(one, two)
        assert one.shape == set_features(q, a, [], mode, 16).shape
        assert one.numel() == pair_features(q, a, mode).numel() + 6


def test_learned_stopping_honors_minimum_without_duplicate_supports():
    class NegativeGain(torch.nn.Module):
        def forward(self, x):
            return torch.full((len(x),), -0.1)

    features = [feature([1, 0]), feature([0, 1]), feature([1, 1])]
    cfg = dict(k=3, max_k=3, min_k=1, stop_epsilon=0.005, feature_mode="full")
    chosen, gains = select(features[0], features, cfg, "adaptive_learned", NegativeGain())
    assert len(chosen) == len(set(chosen)) == 1
    assert len(gains) == 1


def test_mmr_selects_complementary_support():
    class EqualUtility(torch.nn.Module):
        def forward(self, x):
            return torch.ones(len(x))

    f = [feature([1, 0]), feature([1, 0]), feature([0, 1])]
    chosen, _ = select(f[0], f, dict(k=2, feature_mode="image", redundancy=0.5), "mmr", EqualUtility())
    assert chosen == [0, 2]


def test_shape_empty_and_single_pixel_are_finite():
    mask = torch.zeros(1, 8, 8)
    assert torch.equal(shape_descriptor(mask), torch.zeros(10))
    mask[0, 3, 3] = 1
    assert torch.isfinite(shape_descriptor(mask)).all()


def write_manifest(tmp_path, patients, splits, identical=False):
    rows = []
    for i, (patient, split) in enumerate(zip(patients, splits)):
        image = np.zeros((8, 8), dtype=np.uint8)
        image[0, 0] = 100 if identical else i + 1
        Image.fromarray(image).save(tmp_path / f"{i}.png")
        rows.append(
            dict(
                case_id=str(i),
                patient_id=patient,
                split=split,
                task="lesion",
                image_path=f"{i}.png",
                mask_path=f"{i}.png",
            )
        )
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_patient_leakage_rejected(tmp_path):
    path = write_manifest(tmp_path, ["patient1", "patient1"], ["train", "test"])
    with pytest.raises(ValueError, match="Patient leakage"):
        read_manifest(path)


def test_image_exploration_requires_explicit_scope(tmp_path):
    path = write_manifest(tmp_path, ["image:0", "image:1"], ["train", "val"])
    with pytest.raises(ValueError, match="identity_scope=image"):
        read_manifest(path)
    cases = read_manifest(path, identity_scope="image")
    assert [case.patient_id for case in cases] == ["image:0", "image:1"]
    wrong = write_manifest(tmp_path, ["patient1", "patient2"], ["train", "val"])
    with pytest.raises(ValueError, match="Image-level exploration"):
        read_manifest(wrong, identity_scope="image")


def test_image_level_statistics_are_not_labeled_patient_level():
    from far_icl.pipeline import summarize
    from far_icl.report import paired_identity_bootstrap

    a = [dict(case_id="a", patient_id="image:a", dice=0.4, hd95=1.0)]
    b = [dict(case_id="a", patient_id="image:a", dice=0.6, hd95=1.0)]
    summary = summarize(b, identity_scope="image")
    assert summary["identity_scope"] == "image"
    assert "patient_macro_dice" not in summary
    assert summary["image_macro_dice"] == pytest.approx(0.6)
    delta = paired_identity_bootstrap(a, b, identity_scope="image", samples=10)
    assert delta["image_macro_delta"] == pytest.approx(0.2)
    assert "patient_macro_delta" not in delta


def test_duplicate_image_leakage_rejected(tmp_path):
    path = write_manifest(tmp_path, ["patient1", "patient2"], ["train", "val"], identical=True)
    with pytest.raises(ValueError, match="duplicated"):
        read_manifest(path)


def test_bank_excludes_patient_and_wrong_task():
    from types import SimpleNamespace
    from far_icl.bank import CaseBank

    bank = object.__new__(CaseBank)
    bank.entries = [
        {"case": {"patient_id": "q", "task": "lesion"}},
        {"case": {"patient_id": "other", "task": "cup"}},
        {"case": {"patient_id": "safe", "task": "lesion"}},
    ]
    bank.vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    case = SimpleNamespace(patient_id="q", task="lesion", case_id="query")
    assert bank.candidates(case, torch.tensor([1.0, 0.0]), 50) == [bank.entries[2]]


def test_tta_inverse_and_support_alignment():
    from far_icl.segmentation import estimate_uncertainty

    class IdentitySegmenter:
        device = torch.device("cpu")

        def __call__(self, query, supports):
            # Equality survives only if support image/mask received same transform.
            assert torch.equal(query, supports[0]["image"])
            assert torch.equal(query, supports[0]["mask"])
            return query

    query = torch.arange(16).reshape(1, 4, 4).float() / 16
    supports = [{"image": query, "mask": query}]
    u = estimate_uncertainty(
        IdentitySegmenter(), query, supports, supports, {"uncertainty": "tta"}, np.random.default_rng(42)
    )
    assert torch.equal(u, torch.zeros_like(query))
