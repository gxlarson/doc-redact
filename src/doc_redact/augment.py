"""Degrade clean rendered text so it looks printed/scanned, then composite it."""

from __future__ import annotations

from functools import cache

import numpy as np
from PIL import Image, ImageOps


@cache
def _low_ink():
    from augraphy import LowInkRandomLines

    return LowInkRandomLines(count_range=(10, 70), use_consistent_lines=False, noise_probability=0.4)


@cache
def _slight_rotation():
    import albumentations as A

    return A.Compose([A.Rotate(limit=(-1, 1), interpolation=0, p=1.0)])


def darken_ink(image: np.ndarray, threshold: int = 150, increment: int = 140) -> np.ndarray:
    """Push strong strokes darker so low-ink lines don't wash out the text. Returns RGB.

    The strength of a stroke is judged on luminance, but each channel is darkened in proportion, so
    coloured ink stays its own colour. Rebuilding the layer from luminance, as this once did, turned
    every replacement grey.
    """
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    ink = 255.0 - np.asarray(ImageOps.grayscale(Image.fromarray(image)), dtype=np.float32)
    gain = np.where(ink > threshold, (ink + increment) / np.maximum(ink, 1e-6), 1.0)
    # Darkening multiplies each channel's ink, so the strongest channel decides how far we can go.
    # Letting it clip instead would shift the colour: blue ink came out violet once its red and
    # green channels hit the ceiling together.
    headroom = 255.0 / np.maximum((255.0 - rgb).max(axis=2), 1e-6)
    out = 255.0 - (255.0 - rgb) * np.minimum(gain, headroom)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(layer: Image.Image, rotate: bool = False) -> Image.Image:
    arr = np.asarray(layer)
    arr = _low_ink()(arr)
    arr = darken_ink(arr)
    if rotate:
        arr = _slight_rotation()(image=arr)["image"]
    return Image.fromarray(arr)


def overlay(background: Image.Image, layer: Image.Image) -> Image.Image:
    """Copy every non-white pixel of ``layer`` onto ``background`` (same size, RGB)."""
    bg = np.array(background.convert("RGB"))
    fg = np.asarray(layer.convert("RGB"))
    ink = (fg != 255).any(axis=-1)
    bg[ink] = fg[ink]
    return Image.fromarray(bg)
