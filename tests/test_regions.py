"""Dense, overlapping boxes: each replacement keeps to its own share of the page. Values are invented."""
import numpy as np
import pytest
from PIL import Image

from doc_redact import geometry, pipeline, render
from doc_redact.annotations import Annotation, BBox, Document


def test_stacked_overlap_is_split_at_its_middle():
    upper, lower = BBox(100, 100, 200, 20), BBox(95, 114, 210, 20)  # 6 px vertical overlap
    a, b = geometry.exclusive_regions([upper, lower])
    assert a.bbox.bottom == pytest.approx(117) and b.bbox.top == pytest.approx(117)
    assert a.cut_bottom and b.cut_top and not (a.cut_top or b.cut_bottom)
    assert a.bbox.left == upper.left and b.bbox.right == lower.right  # the long axis is untouched


def test_side_by_side_overlap_is_split_vertically_and_separate_boxes_are_untouched():
    left, right, apart = BBox(100, 100, 80, 20), BBox(170, 102, 80, 20), BBox(400, 400, 50, 20)
    a, b, c = geometry.exclusive_regions([left, right, apart])
    assert a.bbox.right == pytest.approx(175) and b.bbox.left == pytest.approx(175)
    assert not c.is_cut and c.bbox == apart


def test_near_duplicate_boxes_of_the_same_category():
    boxes = [BBox(100, 100, 100, 20), BBox(102, 101, 98, 20), BBox(102, 101, 98, 20), BBox(100, 130, 100, 20)]
    assert geometry.near_duplicates(boxes, ["ssn", "ssn", "name", "ssn"]) == {1}


def test_erase_extent_covers_word_ends_sticking_out_of_a_tight_box():
    from doc_redact.ocr import Word
    box = BBox(100, 50, 60, 15)
    words = [Word("Qwertyson", BBox(98, 49, 67, 16)), Word("other", BBox(300, 50, 40, 15))]
    grown = geometry.erase_extent(box, words)
    assert grown.left == 98 and grown.right == pytest.approx(165)
    assert (grown.top, grown.height) == (box.top, box.height)
    assert geometry.erase_extent(box, [Word("far", BBox(90, 50, 200, 15))]) == box  # mostly outside: not grown


def _ink_rows(image):
    return np.nonzero((np.asarray(image.convert("L")) < 128).any(axis=1))[0]


def test_dense_stacked_ssns_do_not_draw_into_each_other():
    image = Image.new("RGB", (500, 200), "white")
    boxes = [BBox(100, 40 + 14 * i, 150, 17) for i in range(4)]  # 17 px tall rows, 3 px overlaps, no OCR words
    doc = Document(url="https://host/page.jpg", annotations=[Annotation("ssn", b) for b in boxes])
    settings = render.TextSettings(font="LiberationSans-Regular.ttf", font_size=24, match_font=False, match_weight=False,
                                   paper_fill=False, keep_lines=False,
                                   overrides={"ssn": "987-65-4320"})
    options = pipeline.Options(mode=pipeline.Mode.PSEUDO, text=settings)
    out = pipeline.redact_document(image, doc, {"Blocks": []}, options)
    regions = geometry.exclusive_regions(boxes)
    pixels = np.asarray(out.convert("L")) < 128
    for i, region in enumerate(regions):
        top, bottom = region.limits(0, render.REGION_MARGIN * boxes[i].height)[1::2]
        assert pixels[int(top):int(bottom) + 1, 100:250].any(), f"box {i} drew nothing"
    assert _ink_rows(out).min() >= boxes[0].top - render.REGION_MARGIN * 17 - 1
    assert _ink_rows(out).max() <= boxes[-1].bottom + render.REGION_MARGIN * 17 + 1
    # Within each cut edge nothing crosses: the rows at the split lines between boxes have no ink run spanning both.
    for upper, lower in zip(regions, regions[1:]):
        split = int(round(upper.bbox.bottom))
        assert not (pixels[split - 1, 100:250] & pixels[split + 1, 100:250]).any()


def test_erasing_a_later_box_does_not_cut_an_earlier_replacement():
    image = Image.new("RGB", (400, 120), "white")
    first, second = BBox(50, 30, 200, 24), BBox(50, 50, 200, 24)  # overlapping by 4 px
    doc = Document(url="https://host/page.jpg", annotations=[Annotation("ssn", first), Annotation("other", second)])
    settings = render.TextSettings(font="LiberationSans-Regular.ttf", font_size=18, match_font=False, match_weight=False,
                                   paper_fill=False, keep_lines=False, overrides={"ssn": "987-65-4320"})
    out = pipeline.redact_document(image, doc, {"Blocks": []}, pipeline.Options(pipeline.Mode.PSEUDO, settings))
    # The 'other' box is only erased; the SSN drawn in the first box survives even where the boxes overlap.
    assert (np.asarray(out.convert("L"))[30:55, 50:250] < 128).sum() > 50
