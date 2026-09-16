"""Degrade clean rendered text so it looks printed/scanned, then composite it."""

from __future__ import annotations

from functools import cache

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

# Midtone pixels per unit of stroke perimeter. A born-digital glyph has a one-pixel antialiased rim;
# a scanned one has a smeared, noisy edge. Measured over 40 RVL-CDIP scans and 15 clean renders at
# five sizes and three faces, the two never came close: scans ran 0.19 to 0.45 (median 0.26), clean
# renders 0.07 to 0.15 (median 0.09). The threshold sits in the gap, nearer the clean side.
SCAN_SOFTNESS = 0.17
_MIN_INK_PIXELS = 200  # fewer than this and the page is too empty to judge


def edge_softness(gray: np.ndarray) -> float | None:
    """How smeared this page's stroke edges are, or None when there is too little ink to tell.

    Normalised by the ink's perimeter rather than its area, so the measure does not drift with font
    size: by area, large scanned text reads as clean.
    """
    g = np.asarray(gray, dtype=np.float32)
    paper = float(np.percentile(g, 90))
    ink = g < paper - 60
    if int(ink.sum()) < _MIN_INK_PIXELS:
        return None
    midtones = (g >= paper - 60) & (g < paper - 12)
    perimeter = ndimage.binary_dilation(ink) & ~ndimage.binary_erosion(ink)
    return float(midtones.sum() / max(1.0, perimeter.sum()))


def looks_scanned(gray: np.ndarray) -> bool:
    """Whether a page carries scanner wear, so a replacement should be degraded to match.

    An empty page counts as scanned: degrading is what this did before there was a choice, and on a
    page with no text to compare against, the replacement's own texture is no giveaway.
    """
    softness = edge_softness(gray)
    return True if softness is None else softness >= SCAN_SOFTNESS


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
