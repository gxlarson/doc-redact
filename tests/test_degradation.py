"""Telling a scan from a born-digital page, so replacements are only worn where the page is."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from doc_redact import augment, fakedata, layout, pipeline, render
from doc_redact.annotations import Annotation, BBox, Document

FONT = "LiberationSerif-Regular.ttf"
BOX = BBox(58, 58, 110, 30)
OCR = {"Blocks": [{"BlockType": "WORD", "Text": "4/12/50", "TextType": "PRINTED",
                   "Geometry": {"BoundingBox": {"Left": 60 / 900, "Top": 62 / 300,
                                                "Width": 95 / 900, "Height": 20 / 300}}}]}


def _clean_page(size=20):
    page = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(page)
    font = layout.load_font(FONT, size)
    draw.text((60, 60), "4/12/50", font=font, fill=(0, 0, 0))
    for i in range(1, 7):
        draw.text((60, 60 + i * (size + 12)), "Invoice line: payment due for services", font=font, fill=(0, 0, 0))
    return page


def _scanned_page():
    """A crisp page blurred and speckled the way a scanner does, which is what softens stroke edges."""
    page = np.asarray(_clean_page().convert("L")).astype(np.float32)
    rng = np.random.default_rng(0)
    from scipy import ndimage
    page = ndimage.gaussian_filter(page, 0.8)
    page += rng.normal(0, 4.0, page.shape)
    return Image.fromarray(np.clip(page, 0, 255).astype(np.uint8)).convert("RGB")


def _gray(image):
    return np.asarray(image.convert("L"))


@pytest.mark.parametrize("size", [12, 20, 34])
def test_edge_softness_does_not_drift_with_font_size(size):
    """Normalising by ink area instead of perimeter made large scanned text read as clean."""
    assert augment.edge_softness(_gray(_clean_page(size))) < augment.SCAN_SOFTNESS


def test_a_scanned_page_reads_as_scanned_and_a_crisp_one_does_not():
    assert augment.looks_scanned(_gray(_scanned_page()))
    assert not augment.looks_scanned(_gray(_clean_page()))


def test_a_page_with_no_ink_is_left_to_the_old_behaviour():
    """Too little to judge: degrade, which is what happened before there was a choice."""
    blank = Image.new("RGB", (400, 200), "white")
    assert augment.edge_softness(_gray(blank)) is None
    assert augment.looks_scanned(_gray(blank))


def _redact(page, degrade):
    doc = Document(url="https://host/page.png", annotations=[Annotation("birth_date", BOX)])
    fakedata.seed(0)
    np.random.seed(0)
    options = pipeline.Options(pipeline.Mode.PSEUDO,
                               render.TextSettings(font=FONT, font_size=20, degrade=degrade))
    return np.asarray(pipeline.redact_document(page, doc, OCR, options).convert("L")).astype(int)


def test_auto_leaves_a_born_digital_page_undegraded():
    page = _clean_page()
    assert np.array_equal(_redact(page, "auto"), _redact(page, "never"))
    assert not np.array_equal(_redact(page, "auto"), _redact(page, "always"))


def test_auto_degrades_a_scanned_page():
    page = _scanned_page()
    assert np.array_equal(_redact(page, "auto"), _redact(page, "always"))
