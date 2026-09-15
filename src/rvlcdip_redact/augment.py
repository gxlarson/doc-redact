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
    """Push strong strokes darker so low-ink lines don't wash out the text. Returns RGB."""
    ink = 255 - np.asarray(ImageOps.grayscale(Image.fromarray(image)), dtype=np.int16)
    ink[ink > threshold] += increment
    np.clip(ink, 0, 255, out=ink)
    out = (255 - ink).astype(np.uint8)
    return np.asarray(Image.fromarray(out).convert("RGB"))


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
