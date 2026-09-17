"""Global, boundary, hard-region and morphology representations."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = cfg["encoder"]
        if self.name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50

            model = resnet50(weights=None if cfg["encoder_weights"] else ResNet50_Weights.IMAGENET1K_V2)
            if cfg["encoder_weights"]:
                model.load_state_dict(
                    torch.load(cfg["encoder_weights"], map_location="cpu", weights_only=True)
                )
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.dim = 2048
        elif self.name == "dinov2":
            repo = cfg["encoder_repo"] or "facebookresearch/dinov2:main"
            self.model = torch.hub.load(
                repo,
                "dinov2_vits14",
                source="local" if cfg["encoder_repo"] else "github",
                pretrained=not bool(cfg["encoder_weights"]),
                trust_repo=True,
            )
            if cfg["encoder_weights"]:
                self.model.load_state_dict(
                    torch.load(cfg["encoder_weights"], map_location="cpu", weights_only=True)
                )
            self.dim = 384
        else:
            raise ValueError(f"Unknown encoder: {self.name}")
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, rgb):
        x = F.interpolate(rgb, (224, 224), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (x - mean) / std
        if self.name == "resnet50":
            fmap = self.model(x)
            global_feature = fmap.mean((-2, -1))
        else:
            features = self.model.forward_features(x)
            fmap = features["x_norm_patchtokens"].transpose(1, 2).reshape(x.shape[0], self.dim, 16, 16)
            global_feature = features["x_norm_clstoken"]
        return F.normalize(global_feature, dim=-1), fmap


def boundary(mask):
    mask = (mask > 0.5).float()
    dilation = F.max_pool2d(mask[None], 3, 1, 1)[0]
    erosion = -F.max_pool2d(-mask[None], 3, 1, 1)[0]
    return (dilation - erosion).clamp(0, 1)


def region_pool(fmap, region):
    # Upsampling the map avoids small-lesion regions vanishing during downsampling.
    fmap = F.interpolate(fmap[None], region.shape[-2:], mode="bilinear", align_corners=False)[0]
    count = region.sum()
    pooled = (fmap * region).sum((-2, -1)) / count.clamp_min(1)
    if count < 1:
        pooled = fmap.mean((-2, -1))
    return F.normalize(pooled, dim=0)


def shape_descriptor(mask):
    a = mask.detach().cpu().numpy().squeeze() > 0.5
    h, w = a.shape
    yy, xx = np.nonzero(a)
    if len(xx) == 0:
        return torch.zeros(10, device=mask.device)
    perimeter = (a & ~ndimage.binary_erosion(a)).sum()
    area = len(xx)
    points = np.stack([yy / h, xx / w], axis=1)
    cov = np.cov(points.T) if area > 1 else np.zeros((2, 2))
    eigen = np.maximum(np.linalg.eigvalsh(cov), 0)
    minor, major = np.sqrt(eigen)
    height, width = (yy.max() - yy.min() + 1) / h, (xx.max() - xx.min() + 1) / w
    components = ndimage.label(a)[1]
    values = [
        area / (h * w),
        perimeter / (h + w),
        yy.mean() / h,
        xx.mean() / w,
        np.sqrt(max(0, 1 - minor**2 / max(major**2, 1e-8))),
        4 * np.pi * area / max(perimeter**2, 1),
        np.log1p(components),
        height / max(width, 1e-8),
        major,
        minor,
    ]
    return torch.tensor(values, dtype=torch.float32, device=mask.device)


def describe(global_feature, fmap, prediction, uncertainty, quantile=0.8):
    edge = boundary(prediction)
    threshold = torch.quantile(uncertainty.float(), quantile)
    hard = (uncertainty > threshold).float()
    return dict(
        global_feature=global_feature,
        boundary=region_pool(fmap, edge),
        hard=region_pool(fmap, hard),
        shape=shape_descriptor(prediction),
        uncertainty=uncertainty.mean().reshape(1),
    )


def pair_features(query, candidate, mode="full"):
    q, r = query["global_feature"], candidate["global_feature"]
    parts = [q, r, (q - r).abs(), q * r]
    if mode in {"shape", "full"}:
        parts += [query["shape"], candidate["shape"], (query["shape"] - candidate["shape"]).abs()]
    if mode in {"boundary", "full"}:
        parts += [query["boundary"], candidate["boundary"], query["boundary"] * candidate["boundary"]]
    if mode in {"failure", "full"}:
        parts += [query["hard"], candidate["hard"], query["hard"] * candidate["hard"], query["uncertainty"]]
    return torch.cat(parts)


def set_features(query, candidate, selected, mode, max_k):
    pair = pair_features(query, candidate, mode)
    dim = query["global_feature"].numel()
    if selected:
        vectors = torch.stack([s["global_feature"] for s in selected])
        pooled = vectors.mean(0)
        redundancy = (vectors @ candidate["global_feature"]).max().reshape(1)
    else:
        pooled, redundancy = pair.new_zeros(dim), pair.new_zeros(1)
    return torch.cat(
        [
            pair,
            pooled,
            candidate["global_feature"] * pooled,
            redundancy,
            pair.new_tensor([len(selected) / max_k]),
        ]
    )
