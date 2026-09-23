"""Explicit patient-level manifests; no filename-based patient inference."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import digest, file_digest


@dataclass(frozen=True)
class Case:
    case_id: str
    patient_id: str
    image_path: str
    mask_path: str
    split: str
    task: str
    domain: str = "unknown"
    spacing_y: float = 1.0
    spacing_x: float = 1.0


def read_manifest(path, identity_scope="patient"):
    if identity_scope not in {"patient", "image"}:
        raise ValueError("identity_scope must be patient or image")
    path = Path(path).resolve()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty manifest")
    cases, ids, patients, images = [], set(), {}, {}
    for row in rows:
        for key in ("case_id", "patient_id", "image_path", "mask_path", "split", "task"):
            if not row.get(key):
                raise ValueError(f"Missing required {key}: {row}")
        image_identity = f"image:{row['case_id']}"
        if identity_scope == "image" and row["patient_id"] != image_identity:
            raise ValueError(f"Image-level exploration requires patient_id={image_identity}")
        if identity_scope == "patient" and row["patient_id"].startswith("image:"):
            raise ValueError("Image-level manifest requires --set identity_scope=image")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train/val/test")
        for key in ("image_path", "mask_path"):
            row[key] = str((path.parent / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        for key in ("spacing_y", "spacing_x"):
            row[key] = float(row.get(key) or 1)
            if not np.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("Spacing must be finite and positive")
        case = Case(**{k: v for k, v in row.items() if k in Case.__dataclass_fields__})
        if case.case_id in ids:
            raise ValueError(f"Duplicate case_id: {case.case_id}")
        ids.add(case.case_id)
        prev = patients.setdefault(case.patient_id, case.split)
        if prev != case.split:
            raise ValueError(f"Patient leakage: {case.patient_id}: {prev}/{case.split}")
        image_hash = file_digest(case.image_path)
        prior = images.setdefault(image_hash, (case.split, case.patient_id))
        if prior != (case.split, case.patient_id):
            raise ValueError(f"Image duplicated across patients/splits: {case.case_id}")
        cases.append(case)
    return cases


def manifest_signature(cases):
    return digest(
        [
            asdict(c) | {"image_sha256": file_digest(c.image_path), "mask_sha256": file_digest(c.mask_path)}
            for c in cases
        ]
    )


def read_array(path):
    if Path(path).suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as im:
        return np.asarray(im).copy()


def load_case(case, cfg, with_mask=True):
    image = read_array(case.image_path).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 2D/HWC image: {case.image_path}")
    image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite image: {case.image_path}")
    h, w = image.shape[:2]
    image = (image - image.min()) / max(float(np.ptp(image)), 1e-8)
    rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)[0]
    gray = (rgb * torch.tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)
    gray = (gray - gray.min()) / (gray.max() - gray.min()).clamp_min(1e-8)
    spacing = (case.spacing_y * h / 128, case.spacing_x * w / 128)
    result = dict(rgb=rgb, image=gray, spacing=spacing, original_size=(h, w))
    if with_mask:
        mask = read_array(case.mask_path)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2 or mask.shape != (h, w):
            raise ValueError(f"Mask must be a registered 2D label image: {case.mask_path}")
        if not np.isfinite(mask).all():
            raise ValueError("Nonfinite mask")
        values = cfg["mask_values"]
        if values is None:
            if not set(np.unique(mask)).issubset({0, 1, 255}):
                raise ValueError("Multiclass masks require explicit mask_values")
            mask = mask > 0
        else:
            mask = np.isin(mask, values)
        result["mask"] = F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None], (128, 128), mode="nearest"
        )[0]
    return result
