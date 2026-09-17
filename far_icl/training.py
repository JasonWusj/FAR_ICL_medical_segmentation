"""Query-grouped ranking, validation early stopping and resumable checkpoints."""

import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import atomic_save, digest, read_tensor, seed_all, write_json
from .retrieval import UtilityRanker, ranking_loss


def checkpoint_path(pipeline, kind):
    train_id = digest({k: pipeline.cfg[k] for k in ("loss", "lr", "hidden_dim", "regression_weight")})[:12]
    return (
        Path(pipeline.cfg["output"]) / "checkpoints" / pipeline.signature[:20] / kind / train_id / "best.pt"
    )


def train(pipeline, kind, resume=False):
    cfg, device = pipeline.cfg, pipeline.device
    seed_all(cfg["seed"], cfg["deterministic"])
    paths = {split: sorted((pipeline.artifacts / kind / split).glob("*.pt")) for split in ("train", "val")}
    if not all(paths.values()):
        raise ValueError("Generate both train and val supervision before training")
    patients = {}
    for split, files in paths.items():
        patients[split] = set()
        expected_ids = {c.case_id for c in pipeline.queries(split)}
        actual_ids = set()
        for path in files:
            sample = read_tensor(path)
            if (
                sample["signature"] != pipeline.signature
                or sample["split"] != split
                or sample["kind"] != kind
            ):
                raise ValueError(f"Invalid supervision provenance: {path}")
            actual_ids.add(sample["case_id"])
            patients[split].add(sample["patient_id"])
        if expected_ids != actual_ids:
            raise ValueError(f"Incomplete {split} supervision; rerun generate")
    if patients["train"] & patients["val"]:
        raise ValueError("Patient leakage in ranker supervision")
    example = read_tensor(paths["train"][0])["groups"][0]["x"]
    model = UtilityRanker(example.shape[1], cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    best_path = checkpoint_path(pipeline, kind)
    last_path = best_path.with_name("last.pt")
    first_epoch, best_loss, stale, history = 0, float("inf"), 0, []
    if resume and last_path.exists():
        state = read_tensor(last_path)
        if state["signature"] != pipeline.signature or state["kind"] != kind:
            raise ValueError("Resume provenance mismatch")
        for key in ("hidden_dim", "lr", "loss", "regression_weight"):
            if state["config"][key] != cfg[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch, best_loss, stale, history = (
            state["epoch"] + 1,
            state["best_loss"],
            state["stale"],
            state["history"],
        )
    for epoch in range(first_epoch, cfg["epochs"]):
        order = list(paths["train"])
        random.Random(cfg["seed"] + epoch).shuffle(order)
        model.train()
        losses = []
        for path in tqdm(order, desc=f"Train {kind} {epoch + 1}/{cfg['epochs']}"):
            groups = read_tensor(path)["groups"]
            # One update per query: long marginal trajectories do not overweight a patient slice.
            optimizer.zero_grad(set_to_none=True)
            query_loss = 0.0
            for group in groups:
                x, y = group["x"].to(device), group["y"].to(device)
                loss = ranking_loss(model(x), y, cfg["loss"], cfg["regression_weight"]) / len(groups)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {path}")
                loss.backward()
                query_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(query_loss)
        model.eval()
        validation, regrets = [], []
        with torch.no_grad():
            for path in paths["val"]:
                group_losses, group_regrets = [], []
                for group in read_tensor(path)["groups"]:
                    x, y = group["x"].to(device), group["y"].to(device)
                    scores = model(x)
                    group_losses.append(float(ranking_loss(scores, y, cfg["loss"], cfg["regression_weight"])))
                    group_regrets.append(float(y.max() - y[scores.argmax()]))
                validation.append(float(np.mean(group_losses)))
                regrets.append(float(np.mean(group_regrets)))
        val = float(np.mean(validation))
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        improved = val < best_loss
        best_loss, stale = (val, 0) if improved else (best_loss, stale + 1)
        log = dict(
            epoch=epoch + 1,
            train_loss=float(np.mean(losses)),
            val_loss=val,
            val_selection_regret=float(np.mean(regrets)),
        )
        history.append(log)
        print(log, flush=True)
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            input_dim=example.shape[1],
            hidden_dim=cfg["hidden_dim"],
            kind=kind,
            signature=pipeline.signature,
            best_loss=best_loss,
            stale=stale,
            history=history,
            config=cfg,
        )
        atomic_save(state, last_path)
        if improved:
            atomic_save(state, best_path)
        write_json(history, best_path.with_name("history.json"))
        if stale >= cfg["patience"]:
            break
    if not best_path.exists():
        raise RuntimeError("No valid best checkpoint produced")
    return best_path
