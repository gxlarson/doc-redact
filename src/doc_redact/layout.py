"""Font sizing, width fitting and placement of replacement text."""

from __future__ import annotations

import math
import re
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from doc_redact.ocr import Word

MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 300
DEFAULT_SPACING = 4  # Pillow's default multiline spacing
SUPERSAMPLE = 4  # text is drawn at this scale so outline strokes can add fractions of a pixel

_measure = ImageDraw.Draw(Image.new("L", (1, 1)))


@cache
def load_font(spec: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font by path, by a file bundled in ``doc_redact/fonts``, or by system font name."""
    if Path(spec).is_file():
        return ImageFont.truetype(spec, size)
    bundled = files("doc_redact").joinpath("fonts", spec)
    if bundled.is_file():
        with as_file(bundled) as path:
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(spec, size)


def text_bbox(font: ImageFont.FreeTypeFont, text: str, spacing: float = DEFAULT_SPACING) -> tuple[float, float, float, float]:
    """Ink bounding box (left, top, right, bottom) of ``text`` drawn at the origin."""
    return _measure.multiline_textbbox((0, 0), text, font=font, spacing=spacing)


def text_width(font: ImageFont.FreeTypeFont, text: str, spacing: float = DEFAULT_SPACING) -> float:
    return text_bbox(font, text, spacing)[2]


def estimate_font_size(spec: str, words: list[Word]) -> int | None:
    """Font size at which the OCR words, re-rendered in this font, span the same width.

    Width is used rather than height because Textract pads word boxes
    vertically; on the paper datasets width-based estimates land within ~10%
    of hand-tuned sizes, height-based ones up to 40% too large.
    """
    reference = load_font(spec, 100)
    estimates = []
    for word in words:
        if len(re.findall(r"[A-Za-z0-9]", word.text)) < 2:
            continue  # punctuation and single glyphs give unreliable measurements
        left, _, right, _ = reference.getbbox(word.text)
        if right - left > 0:
            estimates.append(100 * word.bbox.width / (right - left))
    if not estimates:
        return None
    return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, round(median(estimates))))


def line_spacing_for_pitch(font: ImageFont.FreeTypeFont, pitch: float | None) -> float:
    """Multiline spacing that reproduces an observed line pitch (top-to-top distance)."""
    if pitch is None:
        return DEFAULT_SPACING
    # Pillow advances each line by the height of "A" plus the spacing.
    return max(0.0, pitch - _measure.textbbox((0, 0), "A", font=font)[3])


def choose_text(
    generate: Callable[[], str],
    font: ImageFont.FreeTypeFont,
    max_width: float,
    target_width: float,
    attempts: int = 25,
    tolerance: float = 0.15,
    spacing: float = DEFAULT_SPACING,
) -> str:
    """Generate candidates and keep the one closest to ``target_width`` that fits ``max_width``."""
    best, best_score = "", float("inf")
    for _ in range(attempts):
        candidate = generate()
        width = text_width(font, candidate, spacing)
        overflow = width > max_width
        score = (1e9 if overflow else 0) + abs(width - target_width)
        if score < best_score:
            best, best_score = candidate, score
        if not overflow and abs(width - target_width) <= tolerance * target_width:
            break
    return best


def stroke_px(stroke: int) -> float:
    """Extra ink thickness, in output pixels, added by a supersampled outline of ``stroke``."""
    return 2 * stroke / SUPERSAMPLE


def render_text(
    text: str, spec: str, size: int, stroke: int = 0, spacing: float = DEFAULT_SPACING, offset: tuple[float, float] = (0, 0)
) -> tuple[Image.Image, tuple[int, int]]:
    """Render black-on-white text at SUPERSAMPLE scale and downsample it.

    ``offset`` is the fractional part of the drawing position. Returns the
    grayscale tile and the integer position of its top-left corner relative
    to the requested text origin.
    """
    k = SUPERSAMPLE
    font = load_font(spec, size * k)
    left, top, right, bottom = _measure.multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing * k, stroke_width=stroke
    )
    pad = 2 * k
    ox, oy = offset
    # Integer tile origin in output pixels, chosen so the tile fully contains the ink.
    tile_x = int((ox * k + left - pad) // k)
    tile_y = int((oy * k + top - pad) // k)
    w = -(-int(ox * k + right + pad - tile_x * k) // k)
    h = -(-int(oy * k + bottom + pad - tile_y * k) // k)
    hi = Image.new("L", (w * k, h * k), 255)
    ImageDraw.Draw(hi).multiline_text(
        (ox * k - tile_x * k, oy * k - tile_y * k), text, font=font, fill=0,
        spacing=spacing * k, stroke_width=stroke, stroke_fill=0,
    )
    return hi.resize((w, h), Image.BOX), (tile_x, tile_y)


def tint(tile: Image.Image, ink: tuple[int, int, int]) -> Image.Image:
    """A grayscale coverage tile as coloured ink on white: full coverage is ``ink``, none is paper."""
    coverage = 1.0 - np.asarray(tile, dtype=np.float32) / 255.0
    rgb = 255.0 + coverage[..., None] * (np.array(ink, dtype=np.float32) - 255.0)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def draw_text(
    layer: Image.Image, xy: tuple[float, float], text: str, spec: str, size: int, stroke: int = 0,
    spacing: float = DEFAULT_SPACING, angle_deg: float = 0.0, pivot: tuple[float, float] | None = None,
    ink: tuple[int, int, int] | None = None,
) -> None:
    """Darken ``layer`` (RGB, white background) with text whose origin is at ``xy``.

    ``angle_deg`` tilts the text clockwise (downhill to the right, like a skewed
    scan line) around ``pivot`` (default: the origin). ``ink`` colours the strokes;
    black when omitted.
    """
    x, y = xy
    fx, fy = int(x // 1), int(y // 1)
    tile, (tx, ty) = render_text(text, spec, size, stroke, spacing, offset=(x - fx, y - fy))
    left, top = fx + tx, fy + ty
    if abs(angle_deg) >= 0.05:
        px, py = pivot if pivot is not None else xy
        slant = abs(math.sin(math.radians(angle_deg)))
        pad_x = math.ceil(tile.height * slant) + 2
        pad_y = math.ceil(tile.width * slant) + 2
        padded = Image.new("L", (tile.width + 2 * pad_x, tile.height + 2 * pad_y), 255)
        padded.paste(tile, (pad_x, pad_y))
        # PIL rotates counter-clockwise for positive angles.
        tile = padded.rotate(
            -angle_deg, resample=Image.BICUBIC, center=(px - left + pad_x, py - top + pad_y), fillcolor=255
        )
        left, top = left - pad_x, top - pad_y
    # Clip to the layer; PIL fills out-of-bounds crops with black.
    l, t = max(0, left), max(0, top)
    r, b = min(layer.width, left + tile.width), min(layer.height, top + tile.height)
    if r <= l or b <= t:
        return
    tile = tile.crop((l - left, t - top, r - left, b - top))
    tile = tint(tile, ink) if ink is not None else tile.convert("RGB")
    layer.paste(ImageChops.darker(layer.crop((l, t, r, b)), tile), (l, t))


def shrink_to_fit(spec: str, size: int, text: str, max_width: float, spacing: float = DEFAULT_SPACING) -> int:
    """Largest size <= ``size`` at which ``text`` fits in ``max_width``."""
    while size > MIN_FONT_SIZE:
        width = text_width(load_font(spec, size), text, spacing)
        if width <= max_width:
            break
        size = max(MIN_FONT_SIZE, min(size - 1, int(size * max_width / width)))
    return size
