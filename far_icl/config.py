"""Configuration, reproducibility and artifact provenance."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

DEFAULTS = dict(
    manifest="data/manifest.csv",
    output="runs/isic",
    seed=42,
    device="cuda",
    image_size=128,
    encoder="resnet50",
    encoder_repo=None,
    encoder_weights=None,
    segmenter="universeg",
    segmenter_weights=None,
    tyche_samples=8,
    k=4,
    initial_k=4,
    candidate_n=50,
    uncertainty="context",
    uncertainty_samples=4,
    hard_quantile=0.8,
    feature_mode="full",
    alpha=1.0,
    beta=0.3,
    gamma=0.3,
    delta=0.5,
    redundancy=0.25,
    adaptive_thresholds=[0.005, 0.02],
    adaptive_ks=[1, 2, 4],
    min_k=1,
    max_k=16,
    stop_epsilon=0.005,
    set_trajectories=2,
    epochs=40,
    lr=0.0001,
    hidden_dim=256,
    loss="pairwise",
    regression_weight=1.0,
    patience=8,
    surface_tolerance=2.0,
    mask_values=None,
    save_predictions=True,
    bank_domains=[],
    query_domains=[],
    train_query_domains=[],
    deterministic=True,
    retrieval_diagnostics=False,
    correction_grid=4,
    correction_harm_weight=1.0,
    correction_uncertainty_weight=0.0,
    correction_cost=0.001,
)


def load_config(path, overrides=()):
    cfg = DEFAULTS | (yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise ValueError(f"Unknown override: {item}")
        cfg[key] = yaml.safe_load(value)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    for key in ("k", "initial_k", "candidate_n", "max_k", "min_k", "epochs", "set_trajectories"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["image_size"] != 128:
        raise ValueError("This implementation follows the 128x128 backbone protocol")
    if cfg["feature_mode"] not in {"image", "shape", "boundary", "failure", "full"}:
        raise ValueError("Unknown feature_mode")
    if cfg["uncertainty"] not in {"none", "context", "tta", "tyche"}:
        raise ValueError("Unknown uncertainty mode")
    if cfg["uncertainty_samples"] < 2 or not 0 < cfg["hard_quantile"] < 1:
        raise ValueError("Need >=2 uncertainty samples and hard_quantile in (0,1)")
    if (
        cfg["adaptive_thresholds"] != sorted(cfg["adaptive_thresholds"])
        or len(cfg["adaptive_ks"]) != len(cfg["adaptive_thresholds"]) + 1
    ):
        raise ValueError("Adaptive thresholds must be ordered, with one more K than thresholds")
    if cfg["min_k"] > cfg["max_k"]:
        raise ValueError("min_k exceeds max_k")
    if cfg["candidate_n"] < max(cfg["k"], cfg["initial_k"], cfg["max_k"]):
        raise ValueError("candidate_n must cover k, initial_k and max_k")
    if any(k < 1 for k in cfg["adaptive_ks"]):
        raise ValueError("Adaptive K values must be positive")
    if cfg["loss"] not in {"pairwise", "regression", "listwise"}:
        raise ValueError("Unknown loss")
    if cfg["regression_weight"] <= 0 or cfg["lr"] <= 0 or cfg["patience"] < 1:
        raise ValueError("Positive regression_weight/lr/patience required")
    if cfg["surface_tolerance"] < 0 or cfg["hidden_dim"] < 2:
        raise ValueError("Invalid surface_tolerance or hidden_dim")
    if cfg["uncertainty"] == "tyche" and cfg["segmenter"] != "tyche":
        raise ValueError("Tyche uncertainty requires segmenter=tyche")
    if cfg["correction_grid"] not in {1, 2, 4, 8, 16}:
        raise ValueError("correction_grid must be one of 1,2,4,8,16")
    for key in ("correction_harm_weight", "correction_uncertainty_weight", "correction_cost"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    return cfg


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def read_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
