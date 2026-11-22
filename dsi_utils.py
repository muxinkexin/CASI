"""Paper-based DSI primitives. No scale calibration or adaptive threshold is used."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dd_mask(early, late, beta=0.7, mu=0.1):
    """Use the network's native [0,1] sigmoid disparity, as in released FlexDepth.

    Return [B,1,H,W]; never squeeze the batch or channel dimensions.
    """
    if early.shape != late.shape or early.ndim != 4 or early.shape[1] != 1:
        raise ValueError("DD-mask requires matching [B,1,H,W] disparities")
    with torch.no_grad():
        if not torch.isfinite(early).all() or not torch.isfinite(late).all():
            raise FloatingPointError("Nonfinite reference disparity")
        diff = (early.clamp(0, 1).pow(beta) - late.clamp(0, 1).pow(beta)).abs()
        return (diff < mu).to(early.dtype)


def nd_loss(residual, sigma, mask, sigma_min=1e-3):
    """Mask the complete ND expression; keep the paper's image-area average.

    For source-wise residuals, gather the sigma of the selected source first.
    Multiplying [B,1,H,W] by [B,H,W] would cross-broadcast across samples.
    """
    if residual.shape != sigma.shape or residual.shape != mask.shape:
        raise ValueError("ND residual/sigma/mask must have identical [B,1,H,W] shapes")
    sigma = sigma.clamp_min(sigma_min)
    return (mask * (torch.log1p(sigma) + residual.square() / (2 * sigma.square()))).mean()


def gds_loss(disp, image, mask, vertical_weight=100.0):
    """Ground-contact-prior smoothing; mask is averaged at vertical edge ends.

    This edge convention matches the released DSI layers.py implementation.
    """
    if disp.shape != mask.shape or disp.shape[-2:] != image.shape[-2:]:
        raise ValueError("GDS disparity, RGB and mask must be spatially aligned")
    norm = disp / (disp.mean((2, 3), keepdim=True) + 1e-7)
    dx = (norm[:, :, :, :-1] - norm[:, :, :, 1:]).abs()
    dy = (norm[:, :, :-1, :] - norm[:, :, 1:, :]).abs()
    ix = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean(1, keepdim=True)
    iy = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean(1, keepdim=True)
    edge_mask = (mask[:, :, :-1, :] + mask[:, :, 1:, :]) / 2
    wy = vertical_weight * (1 - edge_mask) + edge_mask
    return (dx * torch.exp(-ix)).mean() + (dy * torch.exp(-iy) * wy).mean()


def resize_mask(mask, size):
    # Native DD decisions are binary; nearest interpolation preserves that meaning.
    return F.interpolate(mask, size=size, mode="nearest")


def refresh_due(completed_epochs, first=1, every=5):
    return every > 0 and completed_epochs >= first and (completed_epochs - first) % every == 0
