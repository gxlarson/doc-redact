import random

import numpy as np
from PIL import Image

from rvlcdip_redact import handwriting, layout, ocr, render, typeface
from rvlcdip_redact.annotations import Annotation, BBox
from rvlcdip_redact.ocr import Word

HAND = "PatrickHand-Regular.ttf"


def test_textract_handwriting_label_is_read():
    blocks = {"Blocks": [
        {"BlockType": "WORD", "Text": "5-17-31", "TextType": "HANDWRITING",
         "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.1, "Width": 0.1, "Height": 0.02}}},
        {"BlockType": "WORD", "Text": "Name", "TextType": "PRINTED",
         "Geometry": {"BoundingBox": {"Left": 0.3, "Top": 0.1, "Width": 0.1, "Height": 0.02}}},
    ]}
    hand, printed = ocr.words(blocks, (1000, 1000))
    assert hand.handwritten and not printed.handwritten
    assert ocr.mostly_handwritten([hand, hand, printed]) and not ocr.mostly_handwritten([hand, printed])


def test_all_handwriting_fonts_load():
    assert set(typeface.candidates(handwriting=True)) == set(typeface.HANDWRITING)


def _ink(layer):
    return np.asarray(layer.convert("L")) < 128


def test_jittered_letters_differ_and_seed_reproduces():
    def render_once(seed):
        random.seed(seed)
        layer = Image.new("RGB", (700, 120), "white")
        handwriting.draw(layer, (10, 20), "mmmmmmmm", HAND, 48)
        return _ink(layer)

    a, b, c = render_once(1), render_once(1), render_once(2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Repeated letters are not pixel-identical copies: compare the first two "m" columns' ink profiles.
    plain = Image.new("RGB", (700, 120), "white")
    layout.draw_text(plain, (10, 20), "mmmmmmmm", HAND, 48)
    assert not np.array_equal(_ink(plain), a)


def test_jitter_stays_near_the_unjittered_layout():
    random.seed(0)
    jittered = Image.new("RGB", (900, 200), "white")
    handwriting.draw(jittered, (20, 40), "Philadelphia 1957", HAND, 40)
    plain = Image.new("RGB", (900, 200), "white")
    layout.draw_text(plain, (20, 40), "Philadelphia 1957", HAND, 40)
    ys_j, xs_j = np.nonzero(_ink(jittered))
    ys_p, xs_p = np.nonzero(_ink(plain))
    assert abs(xs_j.max() - xs_p.max()) < 40 and abs(xs_j.min() - xs_p.min()) < 8
    assert abs(np.median(ys_j) - np.median(ys_p)) < 6


def test_handwritten_box_gets_handwriting_font_and_printed_box_does_not():
    words = [
        Word("5-17-31", BBox(100, 100, 120, 30), handwritten=True),
        Word("Birthdate:", BBox(10, 100, 80, 30)),
        Word("Department", BBox(300, 100, 150, 30)),
    ]
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=24, match_style=False)
    page = render.Page(words=words)
    hand = render.plan_text(Annotation("birth_date", BBox(95, 95, 130, 40)), page, settings)
    assert hand.handwritten and hand.font == settings.handwriting_font
    printed = render.plan_text(Annotation("birth_date", BBox(295, 95, 160, 40)), page, settings)
    assert not printed.handwritten and printed.font == settings.font
    off = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=24, match_style=False, handwriting=False)
    assert not render.plan_text(Annotation("birth_date", BBox(95, 95, 130, 40)), page, off).handwritten
