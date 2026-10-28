"""Shared pixel decoding for enhancement, preview alignment, and segmentation."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb_array(path: str | Path) -> np.ndarray:
    """Decode visual RGB pixels in [0, 1], applying EXIF orientation exactly once.

    Pillow supports 16-bit grayscale directly. Palette indices and CMYK channels
    must first be converted to RGB; slicing their raw arrays is not RGB decoding.
    Floating-point TIFFs have no unambiguous intensity scale and are rejected.
    """
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode == "F":
            raise ValueError("Floating-point images are not supported; use 8-bit RGB or 16-bit grayscale.")
        if image.mode.startswith("I;16") or image.mode == "I":
            pixels = np.asarray(image, dtype=np.float32)
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 65535:
                raise ValueError("Integer grayscale images must contain values within [0, 65535].")
            pixels = np.repeat((pixels / 65535.0)[..., None], 3, axis=-1)
        else:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
        raise ValueError("The image must contain RGB pixels.")
    return np.ascontiguousarray(pixels)


def load_rgb_image(path: str | Path) -> Image.Image:
    """The same oriented pixels, quantized to 8-bit for the semantic processor."""
    return Image.fromarray(np.round(load_rgb_array(path) * 255).astype(np.uint8), "RGB")
