"""
Image I/O helpers:
 - read_images_as_stack(filepaths): returns HxWxB float64 stack and metadata
 - save_image_uint8(path, arr): wrapper to save uint8 arrays
"""
import numpy as np
import imageio

def read_images_as_stack(filepaths):
    """
    Accepts:
     - single path to a multi-channel image (e.g., RGB) OR
     - multiple grayscale files -> stack them in selection order
    Returns:
     - stack (H x W x B) as float64
     - original_dtype (numpy dtype of last read file)
     - orig_min, orig_max (floats)
    """
    if not filepaths:
        raise ValueError("No files provided to read.")
    if len(filepaths) == 1:
        arr = imageio.v2.imread(filepaths[0])
        arr = np.asarray(arr)
        if arr.ndim == 2:
            stack = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            stack = arr
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        orig_dtype = arr.dtype
    else:
        bands = []
        for path in filepaths:
            img = imageio.v2.imread(path)
            a = np.asarray(img)
            if a.ndim == 3:
                # convert color to grayscale by averaging channels
                a = a.mean(axis=2)
            bands.append(a)
        shapes = [b.shape for b in bands]
        if len(set(shapes)) != 1:
            raise ValueError(f"Selected band files have different shapes: {shapes}")
        stack = np.stack(bands, axis=2)
        orig_dtype = bands[0].dtype

    stack_f = stack.astype(np.float64)
    return stack_f, orig_dtype, float(stack_f.min()), float(stack_f.max())

def save_image_uint8(path, arr_uint8):
    """Save a uint8 HxW or HxWx3 array (imageio is used)."""
    imageio.v2.imwrite(path, arr_uint8)
