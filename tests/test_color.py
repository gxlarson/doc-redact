"""Coloured ink: sampled from the original and drawn in the same colour. Values are invented."""
import colorsys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from doc_redact import background, fakedata, layout, pipeline, render
from doc_redact.annotations import Annotation, BBox, Document

FONT = "CourierPrime-Regular.ttf"
BOX = BBox(38, 38, 110, 30)
PAPER = (252, 250, 245)
OCR = {"Blocks": [{"BlockType": "WORD", "Text": "4/12/50", "TextType": "PRINTED",
                   "Geometry": {"BoundingBox": {"Left": 40 / 500, "Top": 42 / 120,
                                                "Width": 95 / 500, "Height": 20 / 120}}}]}


def _page(ink, rule_color=None):
    page = Image.new("RGB", (500, 120), PAPER)
    draw = ImageDraw.Draw(page)
    draw.text((40, 40), "4/12/50", font=layout.load_font(FONT, 22), fill=ink)
    if rule_color:
        draw.line([(20, 66), (480, 66)], fill=rule_color, width=1)
    return page


def _redact(page, **settings):
    doc = Document(url="https://host/page.png", annotations=[Annotation("birth_date", BOX)])
    fakedata.seed(0)
    np.random.seed(0)
    options = pipeline.Options(pipeline.Mode.PSEUDO, render.TextSettings(font=FONT, font_size=22, **settings))
    return pipeline.redact_document(page, doc, OCR, options)


def _ink_of(image, rows=slice(38, 64), cols=slice(38, 148)):
    pixels = np.asarray(image.convert("RGB")).astype(int)[rows, cols].reshape(-1, 3)
    dark = pixels[pixels.sum(axis=1) < 400]
    return tuple(dark.mean(axis=0).round().astype(int)) if len(dark) else None


def _hue(color):
    return colorsys.rgb_to_hsv(*[v / 255 for v in color])[0] * 360


@pytest.mark.parametrize("ink", [(190, 30, 30), (30, 60, 170), (0, 110, 60)])
def test_a_replacement_keeps_the_original_ink_colour(ink):
    drawn = _ink_of(_redact(_page(ink)))
    assert min(abs(_hue(drawn) - _hue(ink)), 360 - abs(_hue(drawn) - _hue(ink))) < 8
    assert np.mean(drawn) < np.mean(PAPER) - 60  # and it is ink, not a pale tint


def test_grey_ink_stays_black_so_grayscale_scans_are_unchanged():
    assert background.sample_ink(*_arrays(_page((25, 25, 25))), BOX) is None
    assert background.sample_ink(*_arrays(_page((120, 120, 125))), BOX) is None  # near-grey too
    drawn = _ink_of(_redact(_page((25, 25, 25))))
    assert max(drawn) - min(drawn) <= 3


def _arrays(page):
    return np.asarray(page.convert("RGB")), np.asarray(page.convert("L"))


def test_colour_matching_can_be_turned_off():
    drawn = _ink_of(_redact(_page((190, 30, 30)), match_color=False))
    assert max(drawn) - min(drawn) <= 3  # black, whatever the original was


def test_sample_ink_reads_the_stroke_not_its_edges_or_the_paper():
    for ink in [(190, 30, 30), (30, 60, 170)]:
        sampled = background.sample_ink(*_arrays(_page(ink)), BOX)
        assert max(abs(a - b) for a, b in zip(sampled, ink)) <= 6
    empty = Image.new("RGB", (200, 60), PAPER)
    assert background.sample_ink(*_arrays(empty), BBox(5, 5, 180, 40)) is None


def test_a_coloured_rule_is_redrawn_in_its_own_colour():
    blue = (30, 60, 170)
    out = np.asarray(_redact(_page(blue, rule_color=blue)).convert("RGB")).astype(int)
    row = out[66, 50:140]
    line = row[row.sum(axis=1) < 600]
    assert len(line) > 50
    assert max(abs(a - b) for a, b in zip(line.mean(axis=0), blue)) <= 12


def test_darkening_ink_does_not_shift_its_colour():
    """Darkening multiplies each channel's ink; letting the strongest clip turned blue violet."""
    from doc_redact import augment
    for ink in [(30, 60, 170), (190, 30, 30), (0, 110, 60)]:
        patch = np.full((8, 8, 3), ink, np.uint8)
        out = augment.darken_ink(patch)[4, 4]
        assert min(abs(_hue(out) - _hue(ink)), 360 - abs(_hue(out) - _hue(ink))) < 8
        assert out.sum() <= sum(ink) + 3  # never lighter than it was
