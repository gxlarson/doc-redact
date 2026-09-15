"""Pick the typeface that best matches the scanned words near a box.

Each candidate font renders the same words the OCR read, sized to the scanned
ink width. Both images are reduced to 1-px skeletons, so stroke weight drops
out, and compared with a symmetric chamfer distance (mean distance from each
skeleton pixel to the nearest pixel of the other skeleton), minimized over
small shifts and scales. Words vote by rank.

Evaluated against 92 hand-picked words from the paper datasets, this chose
the same font as the human 47% of the time and the same kind of font
(serif / sans / typewriter with or without serifs) 76% of the time; general
image embeddings (DINOv2, ResNet-50) did far worse.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from rvlcdip_redact import layout
from rvlcdip_redact.ocr import Word

SHIFT = 3  # px searched in each direction when aligning a rendering to the scan
SCALES = (0.97, 1.0, 1.03)  # around the width-matched size, for scans with uneven letter spacing
MAX_WORDS = 3
MIN_LETTERS = 4
MIN_HEIGHT = 10
CROP_PAD = 3


@dataclass(frozen=True)
class Typeface:
    name: str
    spec: str  # bundled filename, absolute path, or anything layout.load_font accepts
    kind: str  # "serif", "sans", "slab_mono" (typewriter with serifs), "sans_mono" or "handwriting"


BUNDLED = (
    Typeface("Liberation Serif", "LiberationSerif-Regular.ttf", "serif"),
    Typeface("Liberation Serif Bold", "LiberationSerif-Bold.ttf", "serif"),
    Typeface("Gelasio", "Gelasio-Variable.ttf", "serif"),
    Typeface("TeX Gyre Pagella", "texgyrepagella-regular.otf", "serif"),
    Typeface("TeX Gyre Schola", "texgyreschola-regular.otf", "serif"),
    Typeface("TeX Gyre Bonum", "texgyrebonum-regular.otf", "serif"),
    Typeface("Liberation Sans", "LiberationSans-Regular.ttf", "sans"),
    Typeface("Liberation Sans Bold", "LiberationSans-Bold.ttf", "sans"),
    Typeface("Carlito", "Carlito-Regular.ttf", "sans"),
    Typeface("DejaVu Sans", "DejaVuSans.ttf", "sans"),
    Typeface("Courier Prime", "CourierPrime-Regular.ttf", "slab_mono"),
    Typeface("Courier Prime Bold", "CourierPrime-Bold.ttf", "slab_mono"),
    Typeface("Courier 10 Pitch", "Courier10Pitch-Regular.pfb", "slab_mono"),
    Typeface("Courier 10 Pitch Bold", "Courier10Pitch-Bold.pfb", "slab_mono"),
    Typeface("Cutive Mono", "CutiveMono-Regular.ttf", "slab_mono"),
    Typeface("Special Elite", "SpecialElite-Regular.ttf", "slab_mono"),
    Typeface("Liberation Mono", "LiberationMono-Regular.ttf", "sans_mono"),
    Typeface("Inconsolata", "Inconsolata-Variable.ttf", "sans_mono"),
    Typeface("DejaVu Sans Mono", "DejaVuSansMono.ttf", "sans_mono"),
)

# Used for boxes whose OCR words Textract marked as handwriting; drawn with per-letter jitter.
HANDWRITING = (
    Typeface("Patrick Hand", "PatrickHand-Regular.ttf", "handwriting"),
    Typeface("Kalam", "Kalam-Regular.ttf", "handwriting"),
    Typeface("Architects Daughter", "ArchitectsDaughter-Regular.ttf", "handwriting"),
    Typeface("Caveat", "Caveat-Variable.ttf", "handwriting"),
    Typeface("Indie Flower", "IndieFlower-Regular.ttf", "handwriting"),
    Typeface("Gochi Hand", "GochiHand-Regular.ttf", "handwriting"),
    Typeface("Shadows Into Light", "ShadowsIntoLight-Regular.ttf", "handwriting"),
    Typeface("Reenie Beanie", "ReenieBeanie-Regular.ttf", "handwriting"),
    Typeface("Nothing You Could Do", "NothingYouCouldDo-Regular.ttf", "handwriting"),
    Typeface("Covered By Your Grace", "CoveredByYourGrace-Regular.ttf", "handwriting"),
    Typeface("Just Another Hand", "JustAnotherHand-Regular.ttf", "handwriting"),
    Typeface("Homemade Apple", "HomemadeApple-Regular.ttf", "handwriting"),
)

# Common fonts that can't be redistributed but are used when installed (--system-fonts).
SYSTEM = (
    ("Times New Roman", "times.ttf", "serif"), ("Times New Roman Bold", "timesbd.ttf", "serif"),
    ("Georgia", "georgia.ttf", "serif"), ("Palatino Linotype", "pala.ttf", "serif"),
    ("Century", "century.ttf", "serif"), ("Bookman Old Style", "bookos.ttf", "serif"),
    ("Garamond", "gara.ttf", "serif"),
    ("Arial", "arial.ttf", "sans"), ("Arial Bold", "arialbd.ttf", "sans"), ("Arial Narrow", "arialn.ttf", "sans"),
    ("Verdana", "verdana.ttf", "sans"), ("Tahoma", "tahoma.ttf", "sans"), ("Calibri", "calibri.ttf", "sans"),
    ("Courier New", "cour.ttf", "slab_mono"), ("Courier New Bold", "courbd.ttf", "slab_mono"),
    ("Consolas", "consola.ttf", "sans_mono"), ("Lucida Console", "lucon.ttf", "sans_mono"),
    ("Lucida Sans Typewriter", "ltype.ttf", "sans_mono"),
)


def _font_dirs() -> list[Path]:
    dirs = []
    if windir := os.environ.get("WINDIR"):
        dirs.append(Path(windir) / "Fonts")
    if local := os.environ.get("LOCALAPPDATA"):
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    dirs += [Path("/Library/Fonts"), Path("/System/Library/Fonts"), Path.home() / "Library" / "Fonts",
             Path("/usr/share/fonts"), Path.home() / ".local" / "share" / "fonts"]
    return [d for d in dirs if d.is_dir()]


@cache
def _installed_fonts() -> dict[str, str]:
    """Lower-cased filename -> path for every font file in the usual system font folders."""
    found = {}
    for d in _font_dirs():
        for root, _, names in os.walk(d):
            for name in names:
                found.setdefault(name.lower(), os.path.join(root, name))
    return found


@cache
def candidates(include_system: bool = False, handwriting: bool = False) -> tuple[Typeface, ...]:
    """Print typefaces (optionally plus installed system fonts), or the handwriting typefaces."""
    faces = list(HANDWRITING if handwriting else BUNDLED)
    if include_system and not handwriting:
        installed = _installed_fonts()
        for name, filename, kind in SYSTEM:
            if path := installed.get(filename):
                faces.append(Typeface(name, path, kind))
    usable = []
    for face in faces:
        try:
            layout.load_font(face.spec, 20)
        except OSError:
            continue
        usable.append(face)
    return tuple(usable)


# --- word images ----------------------------------------------------------

def _binarize(gray: np.ndarray) -> np.ndarray | None:
    background, darkest = float(np.percentile(gray, 90)), float(gray.min())
    if background - darkest < 60:
        return None
    return gray < (background + darkest) / 2


def _tight(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def scan_word_mask(gray: np.ndarray, word: Word, pad: int = CROP_PAD) -> np.ndarray | None:
    """Binary ink of one OCR word, without specks or pieces of neighboring text."""
    b = word.bbox
    top, left = int(b.top) - pad, int(b.left) - pad
    if top < 0 or left < 0:
        return None
    crop = gray[top:int(b.bottom) + pad + 1, left:int(b.right) + pad + 1]
    if crop.size == 0:
        return None
    ink = _binarize(crop)
    if ink is None:
        return None
    labels, count = ndimage.label(ink, structure=np.ones((3, 3)))
    if count == 0:
        return None
    areas = np.bincount(labels.ravel())[1:]
    boxes = ndimage.find_objects(labels)
    h, w = ink.shape
    keep = np.zeros(count + 1, bool)
    for i, (sl, area) in enumerate(zip(boxes, areas), start=1):
        if area < max(6, 0.02 * areas.max()):
            continue  # specks
        ys, xs = sl
        if xs.stop <= pad or xs.start >= w - pad or ys.stop <= pad or ys.start >= h - pad:
            continue  # entirely in the padding: belongs to a neighbor
        keep[i] = True
    mask = keep[labels]
    return _tight(mask) if mask.sum() >= 40 else None


def _render_mask(text: str, spec: str, size: int) -> np.ndarray | None:
    ink = _binarize(np.asarray(layout.render_text(text, spec, size)[0]))
    return None if ink is None else _tight(ink)


def _place(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), bool)
    y, x = (height - mask.shape[0]) // 2, (width - mask.shape[1]) // 2
    canvas[y:y + mask.shape[0], x:x + mask.shape[1]] = mask
    return canvas


def chamfer_distance(a: np.ndarray, b: np.ndarray, shift: int = SHIFT) -> float:
    """Symmetric chamfer distance between same-shaped boolean images, minimized over shifts of ``b``."""
    ay, ax = np.nonzero(a)
    by, bx = np.nonzero(b)
    if not len(ay) or not len(by):
        return float("inf")
    h, w = a.shape
    dist_a = ndimage.distance_transform_edt(~a)
    dist_b = ndimage.distance_transform_edt(~b)
    best = float("inf")
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            a_to_b = dist_b[np.clip(ay - dy, 0, h - 1), np.clip(ax - dx, 0, w - 1)].mean()
            b_to_a = dist_a[np.clip(by + dy, 0, h - 1), np.clip(bx + dx, 0, w - 1)].mean()
            best = min(best, 0.5 * (a_to_b + b_to_a))
    return best


def _width_matched_size(scan_width: int, text: str, spec: str) -> float | None:
    """Fractional font size at which ``text`` renders exactly as wide as the scan.

    The outline-metric estimate is refined with one test render, because
    hinting and binarization make rendered ink a pixel or two narrower; an
    integer size can be 3% off, which is enough for a rougher font to win.
    """
    left, _, right, _ = layout.load_font(spec, 100).getbbox(text)
    size = max(float(layout.MIN_FONT_SIZE), 100 * scan_width / max(1, right - left))
    rendered = _render_mask(text, spec, round(size, 2))
    if rendered is None:
        return None
    return max(float(layout.MIN_FONT_SIZE), size * scan_width / rendered.shape[1])


def skeleton_distance(scan: np.ndarray, text: str, spec: str) -> float:
    """How far ``text`` rendered in ``spec`` is from the scanned word shape (lower is closer)."""
    base = _width_matched_size(scan.shape[1], text, spec)
    if base is None:
        return float("inf")
    best = float("inf")
    for scale in SCALES:
        rendered = _render_mask(text, spec, round(base * scale, 2))
        if rendered is None:
            continue
        height = max(scan.shape[0], rendered.shape[0]) + 2 * SHIFT + 4
        width = max(scan.shape[1], rendered.shape[1]) + 2 * SHIFT + 4
        best = min(best, chamfer_distance(
            skeletonize(_place(scan, height, width)), skeletonize(_place(rendered, height, width))))
    return best


BODY_ROW_FRACTION = 0.2  # rows with less ink than this (vs the densest row) don't count toward height


def body_height(mask: np.ndarray, line_run: int | None = None) -> int | None:
    """Height of the dense rows of ``mask``, ignoring sparse strokes and long horizontal lines.

    On tightly lined handwriting, ascenders and descenders from the lines above
    and below reach into a word's crop; they are sparse rows and drop out here,
    as does a ruled line under the word (removed as a long horizontal run).
    """
    if line_run and mask.shape[1] > line_run:
        mask = mask & ~ndimage.binary_opening(mask, structure=np.ones((1, line_run), bool))
    rows = mask.sum(axis=1)
    if rows.max() == 0:
        return None
    dense = np.nonzero(rows >= BODY_ROW_FRACTION * rows.max())[0]
    return int(dense.max() - dense.min() + 1)


def height_matched_size(spec: str, words: list[Word], gray: np.ndarray) -> float | None:
    """Font size at which ``words`` rendered in ``spec`` have the same body height as the scan (median).

    Handwriting fonts differ wildly in width for the same height, so sizing them
    by width alone can make a wide cursive face far taller than the writer's hand.
    """
    estimates = []
    for word in words:
        if len(re.findall(r"[A-Za-z0-9]", word.text)) < 3:
            continue
        scan = scan_word_mask(gray, word)
        rendered = _render_mask(word.text, spec, 100)
        if scan is None or rendered is None:
            continue
        scan_h = body_height(scan, line_run=max(20, int(word.bbox.height * 1.5)))
        rendered_h = body_height(rendered)
        if scan_h and rendered_h:
            estimates.append(100 * scan_h / rendered_h)
    return float(np.median(estimates)) if estimates else None


# --- choosing -------------------------------------------------------------

def usable_words(words: list[Word]) -> list[Word]:
    return [
        w for w in words
        if len(re.findall(r"[A-Za-z]", w.text)) >= MIN_LETTERS and w.bbox.height >= MIN_HEIGHT
    ]


class TypefaceMatcher:
    """Chooses typefaces for boxes on one page, caching per-word scores across boxes."""

    def __init__(self, gray: np.ndarray, faces: tuple[Typeface, ...]):
        self.gray = gray
        self.faces = faces
        self._scores: dict[tuple, dict[Typeface, float] | None] = {}

    def word_scores(self, word: Word) -> dict[Typeface, float] | None:
        key = (word.text, round(word.bbox.left), round(word.bbox.top))
        if key not in self._scores:
            scan = scan_word_mask(self.gray, word)
            self._scores[key] = None if scan is None else {
                face: skeleton_distance(scan, word.text, face.spec) for face in self.faces
            }
        return self._scores[key]

    def choose(self, words: list[Word]) -> Typeface | None:
        """Best typeface for ``words`` (rank vote over the first few measurable ones, in the order given).

        Returns None if none of them can be measured.
        """
        totals: dict[Typeface, int] = {}
        measured = 0
        for word in usable_words(words):
            scores = self.word_scores(word)
            if scores is None:
                continue
            for rank, face in enumerate(sorted(scores, key=scores.get)):
                totals[face] = totals.get(face, 0) + rank
            measured += 1
            if measured == MAX_WORDS:
                break
        return min(totals, key=totals.get) if totals else None
