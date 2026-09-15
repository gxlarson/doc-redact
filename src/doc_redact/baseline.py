"""Where scanned text sits: baselines and line skew measured from pixels.

Textract word boxes are padded: on the paper datasets their tops sit ~8% of
the box height above the ink and their bottoms ~1-2 px below the baseline,
and their polygons report far less skew than the scans have (median 0.05 deg
vs 0.44 deg on Tobacco3482). So both are measured on the image instead.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median

import numpy as np
from scipy import ndimage

from doc_redact.geometry import group_lines
from doc_redact.ocr import Word

BODY_ROW_FRACTION = 0.35  # rows with less ink than this (vs the densest row) are ascenders/descenders
MIN_LINE_SPAN = 150  # px between the first and last word before a line's slope is trusted
MAX_SKEW_DEG = 5.0


def word_baseline(gray: np.ndarray, word: Word, vpad: int = 3) -> float | None:
    """Page y of the word's baseline: the lowest row where the ink is dense."""
    if len(re.findall(r"[A-Za-z0-9]", word.text)) < 2:
        return None
    b = word.bbox
    top = max(0, int(b.top) - vpad)
    left, right = max(0, int(b.left)), int(b.right) + 1
    crop = gray[top:int(b.bottom) + vpad + 1, left:right]
    if crop.size == 0:
        return None
    background, darkest = float(np.percentile(crop, 90)), float(crop.min())
    if background - darkest < 60:
        return None
    ink = crop < (background + darkest) / 2
    # An underline or form rule is the densest row under a word; drop long horizontal runs first
    # so the baseline is where the letters stop, not where the line is.
    run = max(20, int(b.height * 1.5))
    if ink.shape[1] > run:
        ink &= ~ndimage.binary_opening(ink, structure=np.ones((1, run), bool))
    rows = ink.sum(axis=1)
    if rows.max() == 0:
        return None
    body = np.nonzero(rows >= BODY_ROW_FRACTION * rows.max())[0]
    return float(top + body.max() + 1)


@dataclass(frozen=True)
class Baseline:
    """A text line's baseline: y = intercept + slope * x (page pixels)."""

    slope: float
    intercept: float

    def y_at(self, x: float) -> float:
        return self.intercept + self.slope * x

    @property
    def angle_deg(self) -> float:
        """Clockwise tilt as seen on the page (positive when the line runs downhill to the right)."""
        return math.degrees(math.atan(self.slope))


def _measured(gray: np.ndarray, words: list[Word]) -> list[tuple[float, float]]:
    points = []
    for w in words:
        y = word_baseline(gray, w)
        if y is not None:
            points.append((w.bbox.left + w.bbox.width / 2, y))
    return points


def fit_line(gray: np.ndarray, words: list[Word], fallback_slope: float = 0.0) -> Baseline | None:
    """Baseline through the words' measured baselines.

    The slope is fitted only when the words span enough width to make it
    reliable; otherwise ``fallback_slope`` (typically the page's skew) is used
    and only the height is taken from the words.
    """
    points = _measured(gray, words)
    if not points:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    slope = fallback_slope
    if len(points) >= 3 and xs.max() - xs.min() >= MIN_LINE_SPAN:
        fitted = float(np.polyfit(xs, ys, 1)[0])
        if abs(math.degrees(math.atan(fitted))) <= MAX_SKEW_DEG:
            slope = fitted
    intercept = float(np.median(ys - slope * xs))
    return Baseline(slope=slope, intercept=intercept)


def page_skew(gray: np.ndarray, words: list[Word]) -> float:
    """Median baseline slope over the page's well-populated lines (0 when none qualify)."""
    slopes = []
    for line in group_lines(words):
        if len(line) < 3 or line[-1].bbox.right - line[0].bbox.left < 2 * MIN_LINE_SPAN:
            continue
        fitted = fit_line(gray, line)
        if fitted is not None and fitted.slope != 0.0:
            slopes.append(fitted.slope)
    return median(slopes) if slopes else 0.0
