"""Auditable KITTI evaluation helpers, shared with the experiment checker."""
import hashlib
from pathlib import Path

import numpy as np

METRIC_NAMES = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eigen_mask(gt, minimum=1e-3, maximum=80.0):
    height, width = gt.shape
    mask = np.isfinite(gt) & (gt > minimum) & (gt < maximum)
    crop = (np.array([0.40810811, 0.99189189, 0.03594771, 0.96405229]) *
            np.array([height, height, width, width])).astype(np.int32)
    crop_mask = np.zeros_like(mask)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
    return mask & crop_mask


def depth_metrics(gt, pred):
    if gt.size == 0 or pred.shape != gt.shape:
        raise ValueError("Metrics require nonempty matching arrays")
    if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
        raise ValueError("Predicted depths must be positive and finite")
    ratio = np.maximum(gt / pred, pred / gt)
    return np.asarray([
        np.mean(np.abs(gt - pred) / gt), np.mean((gt - pred) ** 2 / gt),
        np.sqrt(np.mean((gt - pred) ** 2)), np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
        np.mean(ratio < 1.25), np.mean(ratio < 1.25 ** 2), np.mean(ratio < 1.25 ** 3)])


def align_depth(gt, pred, protocol="M"):
    if gt.size == 0 or not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError("Invalid depth input for alignment")
    if protocol == "M":
        scale = float(np.median(gt) / np.median(pred))
    elif protocol == "RM":
        scale = float(np.median(gt / pred))
    elif protocol == "none":
        scale = 1.0
    else:
        raise ValueError("Unknown alignment protocol")
    return pred * scale, scale


def metric_summary(rows):
    if not rows:
        return None
    return dict(zip(METRIC_NAMES, np.asarray(rows, dtype=float).mean(axis=0).tolist()))
