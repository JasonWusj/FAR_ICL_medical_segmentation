"""Binary overlap and surfel-weighted distances on the resized physical grid."""

import numpy as np


def dice(prediction, truth):
    p, t = prediction > 0.5, truth > 0.5
    return float((2 * (p & t).sum() + 1e-8) / (p.sum() + t.sum() + 1e-8))


def segmentation_metrics(prediction, truth, spacing, tolerance):
    p = np.asarray(prediction.detach().cpu().squeeze() > 0.5)
    t = np.asarray(truth.detach().cpu().squeeze() > 0.5)
    union = (p | t).sum()
    result = {"dice": dice(p, t), "iou": float((p & t).sum() / union) if union else 1.0}
    if not p.any() and not t.any():
        result.update(hd95=0.0, nsd=1.0, empty_mismatch=False)
    elif not p.any() or not t.any():
        # JSON null is explicit and is never silently counted as a finite distance.
        result.update(hd95=None, nsd=0.0, empty_mismatch=True)
    else:
        from surface_distance import metrics

        distances = metrics.compute_surface_distances(t, p, spacing_mm=spacing)
        result.update(
            hd95=float(metrics.compute_robust_hausdorff(distances, 95)),
            nsd=float(metrics.compute_surface_dice_at_tolerance(distances, tolerance)),
            empty_mismatch=False,
        )
    return result


def diversity(features):
    if len(features) < 2:
        return 0.0
    matrix = features @ features.T
    n = len(features)
    return float(1 - (matrix.sum() - matrix.diag().sum()) / (n * (n - 1)))
