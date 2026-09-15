"""Marital status boxes: status and children, often annotated as separate boxes, stay consistent per document."""
import re

import pytest

from rvlcdip_redact import fakedata, render, style
from rvlcdip_redact.annotations import Annotation, BBox
from rvlcdip_redact.fakedata import Family
from rvlcdip_redact.ocr import Word
from rvlcdip_redact.style import ChildrenStyle, CombinedStyle, StatusStyle


@pytest.mark.parametrize("text, expected", [
    ("Married", StatusStyle(capitalized=True)),
    ("single", StatusStyle(capitalized=False)),
    ("2 children", ChildrenStyle("digit", "children", label_first=False)),
    ("3children", ChildrenStyle("digit", "children", label_first=False)),
    ("Three children", ChildrenStyle("word", "children", label_first=False, capitalized=True)),
    ("one daughter", ChildrenStyle("word", "daughters", label_first=False)),
    ("Children: None", ChildrenStyle("word", "children", label_first=True, colon=True, capitalized=True)),
    ("Married, 2 children", CombinedStyle(StatusStyle(True), ", ", ChildrenStyle("digit", "children", False))),
    ("Married - Three Children",
     CombinedStyle(StatusStyle(True), " - ", ChildrenStyle("word", "children", False, capitalized=True))),
])
def test_detect_marital_parts(text, expected):
    assert style.detect_marital(text) == expected


@pytest.mark.parametrize("detail, family, text", [
    (StatusStyle(True), Family("Married", 2), "Married"),
    (StatusStyle(False), Family("Divorced", 0), "divorced"),
    (ChildrenStyle("digit", "children", False), Family("Married", 1), "1 child"),
    (ChildrenStyle("word", "children", False, capitalized=True), Family("Married", 3), "Three children"),
    (ChildrenStyle("word", "children", False), Family("Single", 0), "no children"),
    (ChildrenStyle("word", "daughters", False), Family("Married", 1), "one daughter"),
    (ChildrenStyle("digit", "children", True, colon=True, capitalized=True), Family("Married", 2), "Children: 2"),
    (ChildrenStyle("word", "children", True, colon=True, capitalized=True), Family("Single", 0), "Children: None"),
    (CombinedStyle(StatusStyle(True), " - ", ChildrenStyle("digit", "kids", False)), Family("Widowed", 4), "Widowed - 4 kids"),
])
def test_render_marital_parts_from_the_family(detail, family, text):
    assert fakedata.fake_marital_status(detail, identity=family) == text


def test_no_children_reads_naturally_in_every_form():
    none = Family("Single", 0)
    assert fakedata.fake_marital_status(ChildrenStyle("digit", "children", False), identity=none) == "no children"
    combined = CombinedStyle(StatusStyle(True), ", ", ChildrenStyle("digit", "children", False))
    assert fakedata.fake_marital_status(combined, identity=none) == "Single, no children"
    assert fakedata.fake_marital_status(ChildrenStyle("digit", "children", True, colon=True, capitalized=True),
                                        identity=none) == "Children: None"


def test_several_codes_in_one_box_keep_their_layout_and_the_children_count():
    detected = style.detect("marital_status", "m. 51; c. 3.")
    assert isinstance(detected.detail, style.DigitTemplate)
    fakedata.seed(0)
    out = fakedata.fake_text("marital_status", style.Style(detail=detected.detail, identity=Family("Married", 2)))
    assert re.fullmatch(r"m\. \d\d; c\. 2\.", out) and not out.startswith("m. 51")


def test_families_are_plausible():
    fakedata.seed(0)
    for _ in range(300):
        family = fakedata.fake_family()
        assert not (family.status == "Single" and family.children)


def test_separate_status_and_children_boxes_describe_one_family():
    # "Married" on one line and "2 children" on the next, annotated as two boxes.
    words = [Word("Married", BBox(100, 100, 80, 20)), Word("2", BBox(100, 130, 10, 20)), Word("children", BBox(115, 130, 80, 20))]
    status_box = Annotation("marital_status", BBox(97, 96, 90, 28))
    children_box = Annotation("marital_status", BBox(97, 126, 105, 28))
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=18, auto_font_size=False, fit_width=False)
    for family in (Family("Married", 3), Family("Divorced", 1), Family("Single", 0)):
        page = render.Page(words=words, family=family)
        status = render.plan_text(status_box, page, settings).text
        children = render.plan_text(children_box, page, settings).text
        assert status == family.status
        count = "no" if family.children == 0 else str(family.children)
        assert children == f"{count} {'child' if family.children == 1 else 'children'}"
