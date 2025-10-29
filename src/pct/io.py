"""
Image I/O helpers for PCA / PCT workflow
 - read_images_as_stack(filepaths): returns HxWxB float32 stack and metadata
"""

import numpy as np
import imageio
import tifffile
from skimage.transform import resize

def read_images_as_stack(filepaths):
    """
    Accepts:
     - single path to a multi-band image (e.g., Landsat GeoTIFF)
     - multiple grayscale files (stacked in selection order)

    Returns:
     - stack (H x W x B) as float32
     - original_dtype (numpy dtype)
     - orig_min, orig_max (floats)
    """
    if not filepaths:
        raise ValueError("No files provided to read.")

    if len(filepaths) == 1:
        path = filepaths[0].lower()
        if path.endswith((".tif", ".tiff")):
            arr = tifffile.imread(filepaths[0])
        else:
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
        ref_shape = None
        for path in filepaths:
            img = imageio.v2.imread(path)
            a = np.asarray(img)
            if a.ndim == 3:
                a = a.mean(axis=2)
            if ref_shape is None:
                ref_shape = a.shape
            elif a.shape != ref_shape:
                a = resize(a, ref_shape, preserve_range=True, anti_aliasing=True)
            bands.append(a.astype(np.float32))
        stack = np.stack(bands, axis=2)
        orig_dtype = bands[0].dtype

    stack_f = stack.astype(np.float32)
    return stack_f, orig_dtype, float(stack_f.min()), float(stack_f.max())
