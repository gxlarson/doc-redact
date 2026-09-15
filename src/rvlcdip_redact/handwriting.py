"""Draw text in a handwriting font with per-letter variation.

A handwriting font repeats identical glyphs, which reads as a font. Each
character is drawn on its own with a small random change in size, height,
spacing and rotation, so repeated letters differ and the baseline wanders a
little, the way a pen does. Randomness comes from the ``random`` module, so
``--seed`` makes it reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PIL import Image

from rvlcdip_redact import layout


@dataclass(frozen=True)
class Jitter:
    size: float = 0.06  # relative change in letter size
    rise: float = 0.05  # vertical wander, as a fraction of the font size
    spacing: float = 0.08  # relative change in each letter's advance
    rotation_deg: float = 4.0  # per-letter tilt
    line_drift_deg: float = 1.0  # a slow tilt of the whole line on top of the scanned skew


DEFAULT_JITTER = Jitter()


def _rotate_about(point: tuple[float, float], pivot: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    """Rotate ``point`` clockwise on the page (y points down) around ``pivot``."""
    a = math.radians(angle_deg)
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    return pivot[0] + dx * math.cos(a) - dy * math.sin(a), pivot[1] + dx * math.sin(a) + dy * math.cos(a)


def _quantize(size: float) -> float:
    return round(size * 4) / 4  # keeps the font cache small


def draw(
    layer: Image.Image,
    xy: tuple[float, float],
    text: str,
    spec: str,
    size: float,
    stroke: int = 0,
    spacing: float = layout.DEFAULT_SPACING,
    angle_deg: float = 0.0,
    pivot: tuple[float, float] | None = None,
    jitter: Jitter = DEFAULT_JITTER,
) -> None:
    """Like :func:`layout.draw_text`, but letter by letter with ``jitter``."""
    font = layout.load_font(spec, size)
    ascent = font.getmetrics()[0]
    line_advance = layout.text_bbox(font, "A")[3] + spacing  # Pillow's multiline line step
    x0, y0 = xy
    pivot = pivot if pivot is not None else (x0, y0 + ascent)
    line_angle = angle_deg + random.uniform(-jitter.line_drift_deg, jitter.line_drift_deg)

    for line_number, line in enumerate(text.split("\n")):
        baseline_y = y0 + ascent + line_number * line_advance
        cursor = 0.0
        for i, ch in enumerate(line):
            # Advance from prefix widths so kerning pairs are respected.
            advance = font.getlength(line[: i + 1]) - font.getlength(line[:i])
            if not ch.isspace():
                letter_size = _quantize(size * random.uniform(1 - jitter.size, 1 + jitter.size))
                letter_font = layout.load_font(spec, letter_size)
                letter_ascent = letter_font.getmetrics()[0]
                base = (x0 + cursor, baseline_y + random.uniform(-jitter.rise, jitter.rise) * size)
                base = _rotate_about(base, pivot, line_angle)
                layout.draw_text(
                    layer, (base[0], base[1] - letter_ascent), ch, spec, letter_size, stroke, spacing,
                    angle_deg=line_angle + random.uniform(-jitter.rotation_deg, jitter.rotation_deg), pivot=base,
                )
            cursor += advance * random.uniform(1 - jitter.spacing, 1 + jitter.spacing)
