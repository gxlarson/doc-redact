import itertools

from rvlcdip_redact import checks, geometry, layout
from rvlcdip_redact.annotations import Annotation, BBox
from rvlcdip_redact.ocr import Word

FONT = "CourierPrime-Regular.ttf"


def _rendered_word(text, size, left=0, top=0):
    """A Word whose box is exactly the ink box of ``text`` at ``size``."""
    l, t, r, b = layout.load_font(FONT, size).getbbox(text)
    return Word(text=text, bbox=BBox(left=left, top=top, width=r - l, height=b - t))


def test_estimate_font_size_recovers_rendered_size():
    words = [_rendered_word(t, 27) for t in ("Social", "Security", "123-45-6789", ",")]
    assert abs(layout.estimate_font_size(FONT, words) - 27) <= 1


def test_estimate_font_size_needs_real_words():
    assert layout.estimate_font_size(FONT, [_rendered_word(",", 20)]) is None


def test_choose_text_prefers_fitting_candidate_near_target():
    font = layout.load_font(FONT, 20)
    candidates = itertools.cycle(["x" * 40, "x" * 3, "x" * 10])
    width_10 = layout.text_width(font, "x" * 10)
    chosen = layout.choose_text(lambda: next(candidates), font, max_width=width_10 + 5, target_width=width_10)
    assert chosen == "x" * 10


def test_group_lines_orders_words_for_reading():
    words = [
        Word("Del.,", BBox(150, 131, 40, 10)), Word("41", BBox(100, 100, 20, 10)),
        Word("Dover", BBox(100, 130, 30, 10)), Word("Oak", BBox(130, 101, 30, 10)),
    ]
    lines = geometry.group_lines(words)
    assert geometry.reading_text(lines) == "41 Oak\nDover Del.,"
    assert geometry.line_pitch(lines) == 30


def test_check_flags_date_in_place_box():
    words = [Word("June", BBox(10, 10, 40, 10)), Word("14,", BBox(55, 10, 20, 10)), Word("1938", BBox(80, 10, 30, 10))]
    report = checks.check_annotation(0, Annotation("birth_place", BBox(0, 5, 150, 20)), words)
    assert report.text == "June 14, 1938"
    assert "looks_like_date" in report.flags
    empty = checks.check_annotation(1, Annotation("ssn", BBox(500, 500, 10, 10)), words)
    assert empty.flags == ["no_ocr_words"]
