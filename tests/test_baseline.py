import math

import numpy as np
import pytest
from PIL import Image

from doc_redact import baseline, layout, render
from doc_redact.annotations import Annotation, BBox
from doc_redact.ocr import Word

FONT = "LiberationSerif-Regular.ttf"
SIZE = 28
WORDS = ["Laboratory", "Department", "University", "Professional", "Hospital"]


def _padded_box(text, x, baseline_y):
    """An OCR-like axis-aligned box: the word's ink plus Textract-style vertical padding."""
    font = layout.load_font(FONT, SIZE)
    ascent = font.getmetrics()[0]
    left, top, right, bottom = font.getbbox(text)
    y = baseline_y - ascent
    return BBox(x + left, y + top - 3, right - left, bottom - top + 5)


def _page(angle_deg=0.0, baseline_y=100.0):
    """One scanned line: words on a baseline tilted by angle_deg, drawn with the real rotation path."""
    page = Image.new("RGB", (1400, 260), "white")
    slope = math.tan(math.radians(angle_deg))
    words, x = [], 40.0
    font = layout.load_font(FONT, SIZE)
    ascent = font.getmetrics()[0]
    for text in WORDS:
        by = baseline_y + slope * x
        layout.draw_text(page, (x, by - ascent), text, FONT, SIZE, angle_deg=angle_deg, pivot=(x, by))
        words.append(Word(text, _padded_box(text, x, by)))
        x += font.getlength(text) + 45
    return np.asarray(page.convert("L")), words


def test_word_baseline_ignores_descenders_and_box_padding():
    gray, words = _page()
    # "Laboratory" has a descender (y); the baseline is still the bottom of the dense body rows.
    assert baseline.word_baseline(gray, words[0]) == pytest.approx(100, abs=1)


def test_underline_below_the_word_does_not_pull_the_baseline_down():
    page = Image.new("RGB", (500, 160), "white")
    font = layout.load_font(FONT, SIZE)
    ascent = font.getmetrics()[0]
    layout.draw_text(page, (40, 100 - ascent), "987-65-4320", FONT, SIZE)
    arr = np.array(page)
    arr[106:109, 30:260] = 0  # underline 6 px below the baseline, inside the word's padded box
    gray = np.asarray(Image.fromarray(arr).convert("L"))
    left, top, right, bottom = font.getbbox("987-65-4320")
    word = Word("987-65-4320", BBox(40 + left, 100 - ascent + top - 3, right - left, 109 - (100 - ascent + top - 3)))
    assert baseline.word_baseline(gray, word) == pytest.approx(100, abs=1)


@pytest.mark.parametrize("angle", [0.0, 0.8, -1.5])
def test_fit_line_recovers_skew_and_height(angle):
    gray, words = _page(angle_deg=angle)
    line = baseline.fit_line(gray, words)
    assert line.angle_deg == pytest.approx(angle, abs=0.25)
    assert line.y_at(40) == pytest.approx(100, abs=1.5)


def test_short_lines_use_the_fallback_slope():
    gray, words = _page(angle_deg=1.0)
    line = baseline.fit_line(gray, words[:1], fallback_slope=0.02)
    assert line.slope == 0.02


def test_page_skew_is_the_median_line_slope():
    gray, words = _page(angle_deg=0.9)
    assert math.degrees(math.atan(baseline.page_skew(gray, words))) == pytest.approx(0.9, abs=0.25)


def test_replacement_sits_on_the_scanned_baseline_and_follows_skew():
    gray, words = _page(angle_deg=1.2)
    page = render.Page(words=words, gray=gray, skew=baseline.page_skew(gray, words))
    target = words[2].bbox
    settings = render.TextSettings(font=FONT, font_size=SIZE, auto_font_size=False, match_style=False,
                                   match_weight=False, match_font=False, fit_width=False, overrides={"ssn": "123-45-6789"})
    plan = render.plan_text(Annotation("ssn", target), page, settings)
    expected_baseline = 100 + math.tan(math.radians(1.2)) * plan.xy[0]
    assert plan.baseline_y == pytest.approx(expected_baseline, abs=1.5)
    assert plan.angle_deg == pytest.approx(1.2, abs=0.25)
    # Pillow's default anchor: the first baseline is `ascent` below the origin.
    assert plan.xy[1] == pytest.approx(plan.baseline_y - layout.load_font(FONT, SIZE).getmetrics()[0])

    flat = render.plan_text(Annotation("ssn", target), page, render.TextSettings(**{**settings.__dict__, "match_skew": False}))
    assert flat.angle_deg == 0.0 and flat.baseline_y == pytest.approx(plan.baseline_y)


def test_drawn_baseline_matches_the_origin_convention():
    layer = Image.new("RGB", (300, 120), "white")
    font = layout.load_font(FONT, 40)
    layout.draw_text(layer, (10, 20), "HxH", FONT, 40)  # no descenders: ink bottom == baseline
    rows = np.nonzero((np.asarray(layer.convert("L")) < 128).any(axis=1))[0]
    assert rows.max() + 1 == pytest.approx(20 + font.getmetrics()[0], abs=1)
