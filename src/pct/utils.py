"""
Utility functions for normalization, display, and safe type conversion.
"""

import numpy as np


def normalize_to_uint8(img):
    """
    Normalize a single-channel float image to uint8 (0-255).
    Handles constant images safely by returning all zeros.
    """
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))
    if np.isclose(mx, mn):
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - mn) / (mx - mn)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def stack_to_uint8_images(stack):
    """
    Convert float HxWxB stack into list of uint8 images (per band).
    Normalization is done per band individually.
    """
    H, W, B = stack.shape
    imgs = []
    for b in range(B):
        u8 = normalize_to_uint8(stack[:, :, b])
        imgs.append(u8)
    return imgs


def float_stack_to_scaled_uint8(stack, orig_min=None, orig_max=None):
    """
    Convert float HxWxB stack to uint8 (0-255).
    - If orig_min and orig_max are provided, apply global scaling across all bands.
    - Otherwise, normalize per band.
    """
    H, W, B = stack.shape
    out = np.zeros((H, W, B), dtype=np.uint8)

    if (orig_min is not None) and (orig_max is not None) and (orig_max > orig_min):
        scale = orig_max - orig_min
        for b in range(B):
            tmp = (stack[:, :, b] - orig_min) / scale
            out[:, :, b] = (tmp * 255.0).clip(0, 255).astype(np.uint8)
    else:
        for b in range(B):
            out[:, :, b] = normalize_to_uint8(stack[:, :, b])

    return out


def float_stack_to_dtype(stack_float, dtype):
    """
    Convert float32/float64 stack to the given dtype safely.
    Clips values to valid dtype range before casting.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        stack_float = np.clip(stack_float, info.min, info.max)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        stack_float = np.clip(stack_float, info.min, info.max)
    return stack_float.astype(dtype)
