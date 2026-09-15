import numpy as np
from PIL import Image

from doc_redact import augment, fakedata, layout, render
from doc_redact.annotations import Annotation, BBox
from doc_redact.ocr import Word

FONT = "CourierPrime-Regular.ttf"


def test_white_out_fills_inclusive_rectangle():
    image = Image.new("RGB", (20, 20), "black")
    out = np.asarray(render.white_out(image, BBox(left=2, top=3, width=4, height=5)))
    assert (out[3:9, 2:7] == 255).all()
    assert out[:3].max() == 0 and out[9:].max() == 0
    assert np.asarray(image).max() == 0  # input untouched


def test_overlay_copies_only_non_white_pixels():
    background = Image.new("RGB", (4, 4), (10, 20, 30))
    layer = Image.new("RGB", (4, 4), "white")
    layer.putpixel((1, 2), (0, 0, 0))
    out = np.asarray(augment.overlay(background, layer))
    assert tuple(out[2, 1]) == (0, 0, 0)
    assert tuple(out[0, 0]) == (10, 20, 30)


def _word(text, left, top, height=20, width=None):
    return Word(text=text, bbox=BBox(left=left, top=top, width=width or 12 * len(text), height=height))


def test_single_line_address_is_collapsed_and_aligned_to_original_words():
    fakedata.seed(0)
    region = BBox(left=90, top=95, width=600, height=30)
    words = [_word("41", 100, 100), _word("Oak", 140, 100), _word("Street,", 200, 100)]
    settings = render.TextSettings(font=FONT, font_size=20, match_style=False)
    plan = render.plan_text(Annotation("home_address", region), render.Page(words), settings)
    assert "\n" not in plan.text
    assert plan.xy[0] == 100
    font = layout.load_font(FONT, plan.font_size)
    assert layout.text_width(font, plan.text, plan.spacing) <= region.right - 100


def test_two_line_address_stays_two_lines():
    fakedata.seed(0)
    region = BBox(left=90, top=95, width=600, height=60)
    words = [_word("41", 100, 100), _word("Oak", 140, 100), _word("Dover", 100, 130), _word("Del.,", 150, 130)]
    settings = render.TextSettings(font=FONT, font_size=20)
    plan = render.plan_text(Annotation("home_address", region), render.Page(words), settings)
    assert plan.text.count("\n") == 1
    assert plan.original == "41 Oak\nDover Del.,"


def test_style_matching_can_be_disabled():
    region = BBox(left=0, top=0, width=400, height=30)
    words = [_word("4/12/50", 5, 5)]
    fakedata.seed(3)
    on = [render.plan_text(Annotation("birth_date", region), render.Page(words), render.TextSettings(font=FONT, font_size=20)).text for _ in range(10)]
    assert all(t.count("/") == 2 and len(t.rsplit("/", 1)[1]) == 2 for t in on)
    off_settings = render.TextSettings(font=FONT, font_size=20, match_style=False, fit_width=False)
    off = [render.plan_text(Annotation("birth_date", region), render.Page(words), off_settings).text for _ in range(30)]
    assert any(not (t.count("/") == 2 and len(t.rsplit("/", 1)[1]) == 2) for t in off)


def test_override_text_is_shrunk_to_fit():
    region = BBox(left=0, top=0, width=60, height=30)
    settings = render.TextSettings(font=FONT, font_size=30, auto_font_size=False, overrides={"ssn": "123-45-6789"})
    plan = render.plan_text(Annotation("ssn", region), render.Page([]), settings)
    assert plan.text == "123-45-6789"
    assert plan.font_size < 30
    assert layout.text_width(layout.load_font(FONT, plan.font_size), plan.text) <= region.width - render.TEXT_INSET
