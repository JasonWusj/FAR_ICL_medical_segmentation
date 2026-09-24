"""Transparent adaptations of published retrieval ideas to the shared ISIC/UniverSeg protocol."""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

from far_icl.bank import CaseBank
from far_icl.config import load_config
from far_icl.data import load_case
from far_icl.features import ImageEncoder
from far_icl.metrics import diversity, segmentation_metrics
from far_icl.pipeline import Pipeline

from .result_io import save_results


def stability_s3(prediction):
    mask = np.asarray(prediction.detach().cpu().squeeze() > 0.5)
    if not mask.any() or mask.all():
        return 0.0
    return float(ndimage.binary_erosion(mask, structure=np.ones((3, 3))).sum() / mask.sum())


@torch.no_grad()
def run(cfg, method, split):
    pipe = Pipeline(cfg)
    second_encoder = second_bank = None
    if method == "ires_s3_adapted":
        alt = dict(cfg, encoder="dinov2")
        second_encoder = ImageEncoder(alt).to(pipe.device).eval()
        second_bank = CaseBank(pipe.cases, alt, second_encoder, pipe.manifest_hash)
    rows = []
    for case in tqdm(pipe.queries(split), desc=f"{method}/{split}"):
        pipe.segmenter.reset_stats()
        if pipe.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pipe.device)
        pipe.segmenter.sync()
        start = time.perf_counter()
        if method == "dual_similarity_adapted":
            data, query, candidates, features, _, _, _ = pipe.context(case, need_failure=True)
            scores = [
                float(query["global_feature"] @ feat["global_feature"])
                + float(F.cosine_similarity(query["shape"], feat["shape"], dim=0))
                for feat in features
            ]
            chosen = np.argsort(scores)[::-1][: cfg["k"]].tolist()
            supports = [candidates[j] for j in chosen]
            prediction = pipe.segmenter(data["image"], supports)
            selection_info = {"semantic_shape_scores": [scores[j] for j in chosen]}
        else:
            data = load_case(case, cfg, with_mask=False)
            rng = pipe.rng_for(case)
            z1, _ = pipe.encoder(data["rgb"][None].to(pipe.device))
            z2, _ = second_encoder(data["rgb"][None].to(pipe.device))
            supports1 = pipe.bank.candidates(case, z1[0], cfg["k"])
            supports2 = second_bank.candidates(case, z2[0], cfg["k"])
            # Reset deterministic stochastic-model state for each trial.
            torch.manual_seed(int(rng.integers(0, 2**31)))
            p1 = pipe.segmenter(data["image"], supports1)
            torch.manual_seed(int(rng.integers(0, 2**31)))
            p2 = pipe.segmenter(data["image"], supports2)
            s1, s2 = stability_s3(p1), stability_s3(p2)
            supports, prediction = (supports1, p1) if s1 >= s2 else (supports2, p2)
            chosen_encoder = "resnet50" if s1 >= s2 else "dinov2"
            selection_info = {"s3_resnet50": s1, "s3_dinov2": s2,
                              "chosen_encoder": chosen_encoder}
        pipe.segmenter.sync()
        elapsed = time.perf_counter() - start
        truth = load_case(case, cfg)["mask"]
        selected_features = [s["features"]["global_feature"] for s in supports]
        rows.append(dict(
            case_id=case.case_id, patient_id=case.patient_id, task=case.task, domain=case.domain,
            method=method, split=split, k=len(supports),
            support_ids=[s["case"]["case_id"] for s in supports],
            support_patient_ids=[s["case"]["patient_id"] for s in supports],
            diversity=diversity(torch.stack(selected_features)), elapsed_seconds=elapsed,
            segmentation_seconds=pipe.segmenter.seconds,
            retrieval_seconds=max(0.0, elapsed - pipe.segmenter.seconds),
            segmentation_calls=pipe.segmenter.calls,
            support_evaluations=pipe.segmenter.support_evaluations,
            peak_gpu_mb=torch.cuda.max_memory_allocated(pipe.device) / 1024**2
            if pipe.device.type == "cuda" else 0.0,
            **selection_info,
            **segmentation_metrics(prediction, truth, data["spacing"], cfg["surface_tolerance"]),
        ))
    note = (
        "Semantic ResNet cosine plus predicted-mask shape cosine; an adaptation, not Gao et al. "
        "official dual-similarity augmentation/sampling pipeline."
        if method == "dual_similarity_adapted" else
        "IRES S3 encoder selection with ResNet50/DINOv2, K=2 by default; no P2R reuse. "
        "Adaptation, not exact AAAI implementation."
    )
    return save_results(cfg, pipe.manifest_hash, method, split, rows, extra={"adaptation_note": note})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isic.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--method", choices=("dual_similarity_adapted", "ires_s3_adapted"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args()
    run(load_config(args.config, args.set), args.method, args.split)


if __name__ == "__main__":
    main()
