import numpy as np
import pytest
from PIL import Image

from doc_redact import layout, render, typeface
from doc_redact.annotations import Annotation, BBox
from doc_redact.ocr import Word

FACES = tuple(f for f in typeface.BUNDLED if f.name in {
    "Liberation Serif", "Liberation Sans", "Special Elite", "Courier Prime", "Inconsolata", "TeX Gyre Bonum",
})


def _by_name(name):
    return next(f for f in typeface.BUNDLED if f.name == name)


def _synthetic_page(lines, size=30, specks=True):
    """lines: [(font spec, [words])]. Returns (gray page, OCR words), one text line per entry."""
    page = Image.new("RGB", (1400, 90 * len(lines) + 40), "white")
    words = []
    for row, (spec, texts) in enumerate(lines):
        x, y = 30, 30 + 90 * row
        font = layout.load_font(spec, size)
        for text in texts:
            layout.draw_text(page, (x, y), text, spec, size)
            left, top, right, bottom = font.getbbox(text)
            words.append(Word(text, BBox(x + left - 2, y + top - 2, right - left + 4, bottom - top + 4)))
            x += right + 40
    gray = np.array(page.convert("L"))
    if specks:
        rng = np.random.default_rng(0)
        for _ in range(300):
            yy, xx = rng.integers(0, gray.shape[0]), rng.integers(0, gray.shape[1])
            gray[yy, xx] = 0
    return gray, words


def test_all_bundled_fonts_load():
    assert set(typeface.candidates(False)) == set(typeface.BUNDLED)


def test_scan_word_mask_drops_specks_and_neighbors():
    gray = np.full((40, 120), 255, np.uint8)
    gray[10:30, 20:30] = 0      # the word's ink
    gray[10:30, 50:60] = 0
    gray[15, 90] = 0            # a speck inside the box
    gray[10:30, 0:2] = 0        # a neighbor's stroke, only in the left padding
    word = Word("ab", BBox(left=5, top=8, width=100, height=24))
    mask = typeface.scan_word_mask(gray, word)
    assert mask.shape == (20, 40)  # just the two strokes, tightly cropped
    assert mask.sum() == 2 * 20 * 10


def test_chamfer_distance_ignores_small_shifts():
    a = np.zeros((30, 60), bool)
    a[10:20, 10:40] = True
    b = np.roll(a, (2, -2), axis=(0, 1))
    assert typeface.chamfer_distance(a, b) == 0
    far = np.zeros_like(a)
    far[0:3, 50:58] = True
    assert typeface.chamfer_distance(a, far) > 3


@pytest.mark.parametrize("font_name", ["Special Elite", "Liberation Serif", "Inconsolata"])
def test_matcher_recovers_the_rendering_font(font_name):
    face = _by_name(font_name)
    gray, words = _synthetic_page([(face.spec, ["Laboratory", "Department", "Professional"])])
    matcher = typeface.TypefaceMatcher(gray, FACES)
    assert matcher.choose(words) == face


def test_matcher_skips_unmeasurable_words():
    gray, words = _synthetic_page([("LiberationSans-Regular.ttf", ["of", "to", "Medicine"])], specks=False)
    matcher = typeface.TypefaceMatcher(gray, FACES)
    assert typeface.usable_words(words) == [words[2]]
    assert matcher.choose(words[:2]) is None
    assert matcher.choose(words) == _by_name("Liberation Sans")


def test_plan_text_uses_the_matched_font_and_falls_back_to_the_line():
    serif, typewriter = _by_name("Liberation Serif"), _by_name("Special Elite")
    gray, words = _synthetic_page([(typewriter.spec, ["Birthplace", "Philadelphia", "Pennsylvania"]),
                                   (serif.spec, ["Laboratory", "Department", "Professional"])])
    page = render.Page(words=words, gray=gray, typefaces=typeface.TypefaceMatcher(gray, FACES))
    settings = render.TextSettings(font="CourierPrime-Regular.ttf", font_size=20, match_style=False)

    philadelphia = words[1].bbox
    plan = render.plan_text(Annotation("birth_place", philadelphia), page, settings)
    assert plan.font == typewriter.spec

    # A box over empty space on the serif line has no words of its own: the line decides.
    serif_row = words[3].bbox
    empty = BBox(left=1250, top=serif_row.top, width=120, height=serif_row.height)
    assert render.plan_text(Annotation("birth_date", empty), page, settings).font == serif.spec

    off = render.TextSettings(font="CourierPrime-Regular.ttf", font_size=20, match_style=False, match_font=False)
    assert render.plan_text(Annotation("birth_place", philadelphia), page, off).font == "CourierPrime-Regular.ttf"
