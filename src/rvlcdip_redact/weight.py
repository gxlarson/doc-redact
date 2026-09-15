"""Match the stroke weight of replacement text to the scanned original."""

from __future__ import annotations

import re
from statistics import median

import numpy as np

from rvlcdip_redact import layout
from rvlcdip_redact.ocr import Word

MAX_STROKE = 6  # supersampled outline steps; 6 adds 3 px of thickness
MIN_CONTRAST = 60
MAX_SAMPLES = 6


def stroke_thickness(gray: np.ndarray) -> float | None:
    """Mean stroke thickness of dark ink on a light background, in pixels.

    Binarizes halfway between background and ink, then divides ink area by
    half the boundary length (a stroke of width w and length L has area wL
    and two edges of length L). The boundary is counted on the pixel grid,
    so absolute values are biased on curves; comparisons are made against
    text rendered and measured the same way, which cancels that bias.
    """
    if gray.size == 0:
        return None
    background = float(np.percentile(gray, 90))
    darkest = float(gray.min())
    if background - darkest < MIN_CONTRAST:
        return None
    ink = np.pad(gray < (background + darkest) / 2, 1)
    area = int(ink.sum())
    if area < 15:
        return None
    edges = int((ink[:, 1:] != ink[:, :-1]).sum() + (ink[1:, :] != ink[:-1, :]).sum())
    return 2 * area / edges


def _word_crop(gray: np.ndarray, word: Word) -> np.ndarray:
    b = word.bbox
    h, w = gray.shape
    return gray[max(0, int(b.top)):min(h, int(b.bottom) + 1), max(0, int(b.left)):min(w, int(b.right) + 1)]


def estimate_stroke(spec: str, size: int, words: list[Word], gray: np.ndarray) -> int | None:
    """Outline steps that make ``words`` re-rendered in ``spec`` as thick as they are in ``gray``.

    Returns ``None`` when none of the words can be measured (too short or no
    contrast). Fonts that are already heavier than the scan get 0.
    """
    samples = []
    for word in words:
        if len(re.findall(r"[A-Za-z0-9]", word.text)) < 3:
            continue
        thickness = stroke_thickness(_word_crop(gray, word))
        if thickness is not None:
            samples.append((word.text, thickness))
        if len(samples) == MAX_SAMPLES:
            break
    if not samples:
        return None
    target = median(t for _, t in samples)

    best_stroke, best_error = 0, float("inf")
    for stroke in range(MAX_STROKE + 1):
        rendered = [
            stroke_thickness(np.asarray(layout.render_text(text, spec, size, stroke)[0]))
            for text, _ in samples
        ]
        rendered = [r for r in rendered if r is not None]
        if not rendered:
            break
        thickness = median(rendered)
        error = abs(thickness - target)
        if error < best_error:
            best_stroke, best_error = stroke, error
        if thickness >= target:
            break  # thickness only grows with the outline
    return best_stroke
