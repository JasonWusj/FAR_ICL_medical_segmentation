"""Frozen baseline selection and cache checks using synthetic cases only."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from far_icl.bank import CaseBank
from far_icl.baselines import dual_similarity_indices, mmr_indices
from far_icl.config import DEFAULTS
from far_icl.data import Case, load_case, manifest_signature
from far_icl.features import ImageEncoder
from far_icl.metrics import dice
from far_icl.segmentation import Segmenter


def test_segmentation_and_retrieval_preprocessing(tmp_path):
    pixels = np.zeros((64, 32, 3), dtype=np.uint8)
    pixels[:, 16:] = [255, 128, 64]
    image, mask = tmp_path / "rgb.png", tmp_path / "label.png"
    Image.fromarray(pixels).save(image)
    Image.fromarray(pixels[:, :, 0]).save(mask)
    data = load_case(Case("s", "p", str(image), str(mask), "train", "task"), DEFAULTS)
    assert data["rgb"].shape == (3, 128, 128)
    assert data["image"].shape == data["mask"].shape == (1, 128, 128)
    assert data["image"].min() == 0 and data["image"].max() == 1
    assert data["mask"].unique().tolist() == [0, 1]
    assert data["spacing"] == (0.5, 0.25)

    # Inspect the input and pooling of the real encoder forward without loading weights.
    encoder = ImageEncoder.__new__(ImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.name = "resnet50"
    encoder.model = torch.nn.Identity()
    vector, fmap = encoder(data["rgb"][None])
    assert fmap.shape == (1, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(fmap[0, :, 0, 0], -mean / std)
    torch.testing.assert_close(fmap[0, :, -1, -1], (torch.tensor([255, 128, 64]) / 255 - mean) / std)
    torch.testing.assert_close(vector, F.normalize(fmap.mean((-2, -1))))


@pytest.mark.parametrize("local_weights,k", [(False, 2), (True, 8)])
def test_universeg_frozen_inference_contract(local_weights, k, monkeypatch, tmp_path):
    query = torch.zeros(1, 128, 128)
    query[:, 0, :2] = torch.tensor([0.5, 1.0])
    supports = [{"image": torch.full_like(query, i / k), "mask": (query > i / k).float()} for i in range(k)]

    class TinyUniverSeg(torch.nn.Module):
        def __init__(self, *, pretrained):
            super().__init__()
            assert pretrained is (not local_weights)
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, target, support_images, support_masks):
            assert target.shape == (1, 1, 128, 128)
            assert support_images.shape == support_masks.shape == (1, k, 1, 128, 128)
            assert torch.equal(target[0], query)
            assert torch.equal(support_images[0], torch.stack([s["image"] for s in supports]))
            assert torch.equal(support_masks[0], torch.stack([s["mask"] for s in supports]))
            return (target - 0.5) * self.scale

    monkeypatch.setitem(sys.modules, "universeg", SimpleNamespace(universeg=TinyUniverSeg))
    cfg = dict(device="cpu", segmenter="universeg", segmenter_weights=None)
    if local_weights:
        cfg["segmenter_weights"] = tmp_path / "weights.pt"
        torch.save(dict(scale=torch.tensor(4.0)), cfg["segmenter_weights"])
    segmenter = Segmenter(cfg)
    assert not segmenter.model.training
    assert all(not param.requires_grad for param in segmenter.model.parameters())
    prediction = segmenter(query, supports)
    assert torch.equal(prediction, ((query - 0.5) * (4.0 if local_weights else 2.0)).sigmoid())
    assert prediction[0, 0, 0].item() == 0.5
    assert dice(prediction.numpy(), (query > 0.5).numpy()) == 1.0
    assert (segmenter.calls, segmenter.support_evaluations) == (1, k)


def test_knn_and_random_use_legal_supports_and_exact_k():
    query = Case("q", "patient-q", "", "", "train", "lesion")
    bank = object.__new__(CaseBank)
    bank.entries = [
        {"case": {"case_id": query.case_id, "patient_id": query.patient_id, "task": query.task}},
        {"case": {"case_id": "same-patient", "patient_id": query.patient_id, "task": query.task}},
        {"case": {"case_id": "wrong-task", "patient_id": "p", "task": "disc"}},
        *[{"case": {"case_id": str(i), "patient_id": str(i), "task": query.task}} for i in range(72)],
    ]
    # The last two cases tie for highest cosine. Pool ties retain manifest order.
    bank.vectors = F.normalize(
        torch.tensor([[1.0, 0.0]] * 3 + [[i, 1.0] for i in range(70)] + [[1.0, 0.0], [1.0, 0.0]])
    )
    q = torch.tensor([1.0, 0.0])
    order = ["70", "71", *map(str, reversed(range(70)))]
    for k in (2, 8, 16, 64):
        assert [row["case"]["case_id"] for row in bank.candidates(query, q, k, stable=True)] == order[:k]
        supports = bank.candidates(query, q, k, np.random.default_rng(42))
        assert supports == bank.candidates(query, q, k, np.random.default_rng(42))
        assert supports != bank.candidates(query, q, k, np.random.default_rng(43))
        ids = {row["case"]["case_id"] for row in supports}
        assert len(ids) == k
        assert ids.issubset(order)


@pytest.mark.parametrize("weight,expected", [(0.0, [0, 3, 2]), (0.25, [0, 3, 2]), (1.0, [0, 1, 2])])
def test_mmr_balances_relevance_and_maximum_redundancy(weight, expected):
    candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert mmr_indices(candidates[0], candidates, 3, weight) == expected


def test_adapted_dual_combines_shapes_and_semantics_with_stable_ties():
    candidates = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.8, 0.6]])
    shapes = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    assert dual_similarity_indices(candidates[0], candidates, shapes[1], shapes, 3) == [1, 2, 0]
    assert dual_similarity_indices(candidates[0], candidates, torch.zeros(2), shapes, 2) == [0, 1]


def test_train_bank_reuses_cached_features_across_methods_and_k(tmp_path, monkeypatch):
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoded = 0

        def forward(self, rgb):
            self.encoded += len(rgb)
            pooled = rgb.mean((-2, -1))
            return F.normalize(pooled), pooled[:, :, None, None]

    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    Image.fromarray(pixels).save(image)
    Image.fromarray((pixels > 30).astype(np.uint8)).save(mask)
    case = Case("train", "p", str(image), str(mask), "train", "lesion")
    cases = [case, replace(case, case_id="val", patient_id="v", split="val")]
    signature = manifest_signature(cases)
    monkeypatch.setenv("FARICL_BANK_CACHE_ROOT", str(tmp_path / "shared_cache"))
    encoder = TinyEncoder()
    banks = [
        CaseBank(
            cases, DEFAULTS | {"output": str(tmp_path / method), "k": k, "device": "cpu"}, encoder, signature
        )
        for method, k in (("random", 2), ("knn", 8), ("mmr", 16), ("dual_similarity_adapted", 64))
    ]
    assert encoder.encoded == 1
    assert len({bank.root for bank in banks}) == 1
    assert len(list(banks[0].root.glob("*.pt"))) == 1
    for bank in banks:
        assert [entry["case"]["case_id"] for entry in bank.entries] == ["train"]
        assert torch.equal(bank.vectors, banks[0].vectors)
