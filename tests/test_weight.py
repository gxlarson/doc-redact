import numpy as np
import pytest
from PIL import Image

from rvlcdip_redact import layout, weight
from rvlcdip_redact.annotations import BBox
from rvlcdip_redact.ocr import Word

FONT = "CourierPrime-Regular.ttf"


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_stroke_thickness_of_vertical_bars(width):
    gray = np.full((60, 80), 255, dtype=np.uint8)
    for left in (10, 30, 50):
        gray[5:55, left:left + width] = 0
    assert weight.stroke_thickness(gray) == pytest.approx(width, rel=0.12)


def test_stroke_thickness_needs_contrast():
    assert weight.stroke_thickness(np.full((20, 20), 200, dtype=np.uint8)) is None


def test_outline_thickens_rendered_text_monotonically():
    thickness = [weight.stroke_thickness(np.asarray(layout.render_text("Medical", FONT, 24, s)[0])) for s in range(4)]
    assert thickness == sorted(thickness) and thickness[-1] > thickness[0] + 1


def _page_with_text(texts, size, stroke):
    """A synthetic scan: each text rendered at (x, 20) with a known outline, plus matching OCR words."""
    page = Image.new("RGB", (900, 80), "white")
    words, x = [], 10
    for text in texts:
        layout.draw_text(page, (x, 20), text, FONT, size, stroke)
        left, top, right, bottom = layout.load_font(FONT, size).getbbox(text)
        words.append(Word(text, BBox(x + left - 2, 20 + top - 2, right - left + 4, bottom - top + 4)))
        x += right + 30
    return np.asarray(page.convert("L")), words


@pytest.mark.parametrize("true_stroke", [0, 2, 4])
def test_estimate_stroke_recovers_known_outline(true_stroke):
    gray, words = _page_with_text(["Married", "children", "Hartley"], size=24, stroke=true_stroke)
    assert abs(weight.estimate_stroke(FONT, 24, words, gray) - true_stroke) <= 1


def test_draw_text_clips_at_page_edges():
    page = Image.new("RGB", (40, 30), "white")
    layout.draw_text(page, (-15, -8), "Wide text", FONT, 20, stroke=2)
    layout.draw_text(page, (30, 20), "Wide text", FONT, 20, stroke=2)
    arr = np.asarray(page)
    # No black rectangles from out-of-bounds crops: the ink covers well under the whole page.
    assert (arr < 128).mean() < 0.5
