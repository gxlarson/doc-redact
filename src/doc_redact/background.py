"""Erase a box so it blends with the page: generated paper texture plus redrawn form lines.

Nothing inside the box is copied from the original. Paper tone, grain and
speck density are measured on a ring around the box (ignoring text), and
long horizontal lines that cross the box (form rules, underlines) are
measured and redrawn as clean synthetic lines. The only information carried
into the box is where a line ran, which the rest of the page already shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from doc_redact.annotations import BBox
from doc_redact.ocr import Word

RING_MIN = 6  # px of paper sampled around the box, at least
MAX_NOISE_STD = 25.0
SPECK_MAX_AREA = 6  # dark components up to this many px count as paper specks, not text
SPECK_ISOLATION = 4  # px a speck must be from any larger ink
LINE_MIN_LENGTH = 30
LINE_MIN_CONTRAST = 18  # gray levels below the paper for a pixel to count toward a line
LINE_GAP_BRIDGE = 5  # px breaks along a faint line that are bridged
LINE_MIN_COVERAGE = 0.8  # share of the between-words stretch where a real line's pixels are present
LINE_MIN_FREE_FRACTION = 0.2  # a line hidden behind words for more than 80% of its length is not trusted
LINE_CLEAR_BAND = 3  # px just above and below a line that should be paper
LINE_MAX_CLUTTER = 0.35  # share of the between-words stretch allowed to have ink both above and below
LINE_MAX_SLOPE = math.tan(math.radians(5))
LINE_OVERHANG = 5  # px past the box edge that marks a line as part of the form
# A line that stays inside the box must instead span most of it and be ruler-straight;
# handwritten strokes wobble by more than this.
CONTAINED_MIN_SPAN = 0.6
CONTAINED_MAX_RMS = 1.0


@dataclass(frozen=True)
class Paper:
    color: tuple[float, float, float]
    noise_std: float
    speck_density: float  # specks per px
    speck_gray: float
    speck_area: float = 1.0  # mean px per speck


@dataclass(frozen=True)
class Rule:
    """A horizontal line: center y = y0 + slope * (x - x0), from x_start to x_end (page px)."""

    x0: float
    y0: float
    slope: float
    thickness: float
    x_start: float
    x_end: float
    gray: float

    def y_at(self, x):
        return self.y0 + self.slope * (x - self.x0)


def _clip_box(bbox: BBox, shape: tuple[int, ...], margin_x: int = 0, margin_y: int = 0) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    return (max(0, int(math.floor(bbox.left)) - margin_x), max(0, int(math.floor(bbox.top)) - margin_y),
            min(w, int(math.ceil(bbox.right)) + 1 + margin_x), min(h, int(math.ceil(bbox.bottom)) + 1 + margin_y))


def sample_paper(rgb: np.ndarray, gray: np.ndarray, bbox: BBox) -> Paper:
    """Paper statistics from a ring around ``bbox``, ignoring text, lines and anything else dark."""
    margin = max(RING_MIN, int(bbox.height * 0.5))
    x0, y0, x1, y1 = _clip_box(bbox, gray.shape, margin, margin)
    bx0, by0, bx1, by1 = _clip_box(bbox, gray.shape)
    region = gray[y0:y1, x0:x1].astype(np.float32)
    ring = np.ones(region.shape, bool)
    ring[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = False
    if not ring.any():
        return Paper((255.0, 255.0, 255.0), 0.0, 0.0, 0.0)

    background = float(np.percentile(region[ring], 90))
    paper = ring & (region >= background - 40)
    if paper.sum() < 20:
        return Paper((255.0, 255.0, 255.0), 0.0, 0.0, 0.0)
    color = tuple(float(np.median(rgb[y0:y1, x0:x1, c][paper])) for c in range(3))
    noise_std = min(MAX_NOISE_STD, float(region[paper].std()))

    speck_values, speck_areas = [], []
    # Label dark ink over the whole region (box included) so letters that cross into the ring stay whole.
    dark = region < background - 60
    labels, count = ndimage.label(dark, structure=np.ones((3, 3)))
    if count:
        areas = np.bincount(labels.ravel())[1:]
        big = np.isin(labels, np.nonzero(areas > SPECK_MAX_AREA)[0] + 1)
        near_text = ndimage.binary_dilation(big, iterations=SPECK_ISOLATION)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if areas[i - 1] > SPECK_MAX_AREA:
                continue
            ys, xs = sl
            at_edge = ys.start == 0 or xs.start == 0 or ys.stop == region.shape[0] or xs.stop == region.shape[1]
            component = labels[sl] == i
            if at_edge or not ring[sl][component].all() or near_text[sl][component].any():
                continue  # cut off by the crop, inside the box, or a piece of nearby writing (i-dot, period)
            speck_areas.append(int(areas[i - 1]))
            speck_values.append(float(region[sl][component].min()))
    density = len(speck_areas) / max(1, paper.sum() + sum(speck_areas))
    speck_gray = float(np.median(speck_values)) if speck_values else 0.0
    speck_area = float(np.mean(speck_areas)) if speck_areas else 1.0
    return Paper(color, noise_std, density, speck_gray, speck_area)


def generate_paper(paper: Paper, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """An RGB uint8 patch of paper with the sampled tone, grain and specks."""
    patch = np.empty((height, width, 3), np.float32)
    patch[:] = paper.color
    if paper.noise_std > 0.5:
        grain = ndimage.gaussian_filter(rng.standard_normal((height, width)).astype(np.float32), 0.7)
        grain *= paper.noise_std / max(1e-6, float(grain.std()))
        patch += grain[..., None]
    specks = rng.poisson(paper.speck_density * height * width) if paper.speck_density > 0 else 0
    side = max(1, round(math.sqrt(paper.speck_area)))  # square specks with the sampled mean area
    for _ in range(specks):
        y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
        patch[y:y + side, x:x + side] = paper.speck_gray
    return np.clip(patch, 0, 255).astype(np.uint8)


def _band_has_ink(gray: np.ndarray, threshold: float, x: int, y_from: float, y_to: float) -> bool:
    ys = np.arange(int(np.floor(y_from)), int(np.ceil(y_to)) + 1)
    ys = ys[(ys >= 0) & (ys < gray.shape[0])]
    return bool(len(ys)) and 0 <= x < gray.shape[1] and bool((gray[ys, x] < threshold).any())


def _looks_like_a_ruled_line(
    gray: np.ndarray, threshold: float, left: float, right: float, intercept: float, slope: float,
    thickness: float, words: list[Word],
) -> bool:
    """Checks that separate real lines from things that merely contain long dark runs.

    1. Visible between words: a ruled line continues through the spaces between
       words; a row of letter bottoms or serifs (joined by bridging) does not. A
       word only hides the line where the line crosses the word's middle, so an
       underline just below a word still counts as visible there.
    2. Clean on both sides: a ruled line has paper just above and below it,
       except where writing crosses it; halftone shading or scan texture is dark
       above and below as well.
    """
    columns = np.arange(int(left), int(right))
    centers = intercept + slope * columns
    half = thickness / 2 + 1
    covered = np.zeros(len(columns), bool)
    for w in words:
        b = w.bbox
        covered |= (columns >= b.left) & (columns <= b.right) & (centers >= b.top + 0.15 * b.height) \
            & (centers <= b.bottom - 0.15 * b.height)
    free = np.nonzero(~covered)[0]
    if len(free) < LINE_MIN_FREE_FRACTION * len(columns):
        return False  # almost all of it runs through words: can't tell a line from letters
    present = cluttered = 0
    for k in free:
        x, c = int(columns[k]), centers[k]
        present += _band_has_ink(gray, threshold, x, c - half, c + half)
        above = _band_has_ink(gray, threshold, x, c - half - LINE_CLEAR_BAND, c - half - 1)
        below = _band_has_ink(gray, threshold, x, c + half + 1, c + half + LINE_CLEAR_BAND)
        cluttered += above and below
    return present / len(free) >= LINE_MIN_COVERAGE and cluttered / len(free) <= LINE_MAX_CLUTTER


def detect_rules(gray: np.ndarray, bbox: BBox, words: list[Word] | None = None) -> list[Rule]:
    """Long, thin, nearly horizontal lines that run through ``bbox``.

    ``words`` (the page's OCR words) let a candidate be checked in the gaps
    between words, which rejects rows of letter bottoms that merely look like lines.
    Candidates must also have clean paper above and below (see _looks_like_a_ruled_line).
    """
    reach = max(LINE_MIN_LENGTH, int(bbox.height * 2))
    x0, y0, x1, y1 = _clip_box(bbox, gray.shape, reach, 3)
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return []
    background, darkest = float(np.percentile(region, 90)), float(region.min())
    if background - darkest < LINE_MIN_CONTRAST:
        return []
    # Ruled notebook lines can be only ~25 levels darker than the paper, far lighter than the
    # writing on them, so lines get their own threshold just below the paper's noise level.
    paper = region[region >= background - 40]
    noise = float(paper.std()) if paper.size else 0.0
    threshold = background - max(LINE_MIN_CONTRAST, 3 * noise)
    ink = region < threshold

    # Bridge the 1 px steps of slightly tilted lines and small breaks in faint ones,
    # then keep only long horizontal runs.
    length = max(LINE_MIN_LENGTH, int(bbox.height * 1.5))
    bridged = ndimage.binary_dilation(ink, structure=np.ones((3, 1), bool))
    bridged = ndimage.binary_closing(bridged, structure=np.ones((1, LINE_GAP_BRIDGE), bool))
    runs = ndimage.binary_opening(bridged, structure=np.ones((1, length), bool))
    # Group the bridged runs (so a dashed faint line stays one line), but measure only real ink.
    labels, count = ndimage.label(runs, structure=np.ones((3, 3)))
    rules = []
    max_thickness = max(3.0, bbox.height * 0.25)
    for sl, i in zip(ndimage.find_objects(labels), range(1, count + 1)):
        ys, xs = np.nonzero((labels[sl] == i) & ink[sl])
        if len(xs) == 0:
            continue
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        left, right = xs.min() + x0, xs.max() + x0 + 1
        if right - left < length:
            continue
        columns = np.unique(xs)
        centers = np.array([ys[xs == c].mean() for c in columns]) + y0
        widths = np.array([(xs == c).sum() for c in columns])
        slope, intercept = np.polyfit(columns + x0, centers, 1)
        thickness = float(np.median(widths))
        if abs(slope) > LINE_MAX_SLOPE or thickness > max_thickness:
            continue
        extends_past_box = left <= bbox.left - LINE_OVERHANG or right >= bbox.right + LINE_OVERHANG
        if not extends_past_box:
            rms = float(np.sqrt(np.mean((centers - (intercept + slope * (columns + x0))) ** 2)))
            if (right - left) < CONTAINED_MIN_SPAN * bbox.width or rms > CONTAINED_MAX_RMS:
                continue  # short or wobbly: more likely a handwritten stroke than a ruled line
        mid = (left + right) / 2
        y_mid = intercept + slope * mid
        if not (bbox.top - thickness <= y_mid <= bbox.bottom + thickness):
            continue
        if not _looks_like_a_ruled_line(gray, threshold, left, right, intercept, slope, thickness, words or []):
            continue
        values = region[ys, xs]
        rules.append(Rule(x0=mid, y0=float(y_mid), slope=float(slope), thickness=thickness,
                          x_start=float(left), x_end=float(right), gray=float(np.median(values))))
    return rules


def draw_rule(patch: np.ndarray, rule: Rule, origin: tuple[int, int]) -> None:
    """Anti-aliased line onto an RGB patch whose top-left is at page ``origin``."""
    ox, oy = origin
    h, w = patch.shape[:2]
    xs = np.arange(w) + ox + 0.5
    inside = (xs >= rule.x_start) & (xs <= rule.x_end)
    if not inside.any():
        return
    ys = np.arange(h)[:, None] + oy + 0.5
    center = rule.y_at(xs)[None, :]
    half = rule.thickness / 2
    coverage = np.clip(np.minimum(ys + 0.5, center + half) - np.maximum(ys - 0.5, center - half), 0, 1)
    coverage *= inside[None, :]
    shaded = patch.astype(np.float32)
    shaded = shaded * (1 - coverage[..., None]) + rule.gray * coverage[..., None]
    patch[:] = np.clip(shaded, 0, 255).astype(np.uint8)


def blank(
    image_rgb: np.ndarray, source_rgb: np.ndarray, source_gray: np.ndarray, bbox: BBox,
    rng: np.random.Generator, paper_fill: bool = True, keep_lines: bool = True, words: list[Word] | None = None,
) -> list[Rule]:
    """Erase ``bbox`` in ``image_rgb`` in place, measuring paper and lines on the untouched source page."""
    x0, y0, x1, y1 = _clip_box(bbox, image_rgb.shape)
    if x1 <= x0 or y1 <= y0:
        return []
    if paper_fill:
        patch = generate_paper(sample_paper(source_rgb, source_gray, bbox), y1 - y0, x1 - x0, rng)
    else:
        patch = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    rules = detect_rules(source_gray, bbox, words) if keep_lines else []
    for rule in rules:
        draw_rule(patch, rule, (x0, y0))
    image_rgb[y0:y1, x0:x1] = patch
    return rules
