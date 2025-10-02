"""
Small utility functions for display and conversion.
"""
import numpy as np
from PIL import Image

def normalize_to_uint8(img):
    """
    Normalize a single-channel float image to uint8 0-255.
    Handles constant images safely.
    """
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))
    if np.isclose(mx, mn):
        return (np.zeros_like(img, dtype=np.uint8))
    scaled = (img - mn) / (mx - mn)
    scaled = (scaled * 255.0).clip(0,255)
    return scaled.astype(np.uint8)

def stack_to_uint8_images(stack, orig_min=None, orig_max=None):
    """
    Convert float HxWxB stack to list of uint8 images (per band)
    using normalization per band.
    """
    H,W,B = stack.shape
    imgs = []
    for b in range(B):
        u8 = normalize_to_uint8(stack[:,:,b])
        imgs.append(u8)
    return imgs

def float_stack_to_scaled_uint8(stack, orig_min=None, orig_max=None):
    """
    If original ranges (orig_min/orig_max) are provided, rescale stack
    into 0-255 using that range, else normalize per-band.
    Returns HxWxB uint8.
    """
    H,W,B = stack.shape
    out = np.zeros((H,W,B), dtype=np.uint8)
    if (orig_min is not None) and (orig_max is not None) and (orig_max > orig_min):
        # global scaling
        for b in range(B):
            tmp = (stack[:,:,b] - orig_min) / (orig_max - orig_min)
            out[:,:,b] = (tmp*255.0).clip(0,255).astype(np.uint8)
    else:
        for b in range(B):
            out[:,:,b] = normalize_to_uint8(stack[:,:,b])
    return out
