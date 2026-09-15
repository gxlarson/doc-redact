"""Name style detection and generation. All names here are invented."""
import re

import pytest

from rvlcdip_redact import fakedata, style
from rvlcdip_redact.style import NameStyle, NameToken


@pytest.mark.parametrize(
    "text, kinds, last_first",
    [
        ("Jonas Hartley", ["first", "last"], False),
        ("Jonas P. Hartley", ["first", "initial", "last"], False),
        ("Hartley, Jonas P.", ["last", "first", "initial"], True),
        ("Mr. Ansel Quimby Jr.", ["title", "first", "last", "suffix"], False),
        ("R. T. Wendover", ["initial", "initial", "last"], False),
        ("Mrs. Fell", ["title", "last"], False),
        ("Margo", ["first"], False),
        ("Delia Ann Marsh-Oxley", ["first", "middle", "last"], False),
    ],
)
def test_detect_name_shapes(text, kinds, last_first):
    detected = style.detect_name(text)
    assert [t.kind for t in detected.tokens] == kinds
    assert detected.last_first == last_first


def test_non_names_are_not_detected():
    assert style.detect_name("555-014-7788") is None
    assert style.detect_name("12 Maple Avenue") is None
    assert style.detect_name("") is None


def _shape(text):
    return [t.kind for t in style.detect_name(text).tokens], style.detect_name(text).last_first


@pytest.mark.parametrize("text", [
    "Jonas P. Hartley", "Hartley, Jonas P.", "Mr. Ansel Quimby Jr.", "R. T. Wendover", "Delia Ann Marsh-Oxley",
])
def test_generated_names_keep_the_original_shape(text):
    detected = style.detect_name(text)
    fakedata.seed(3)
    for _ in range(20):
        name = fakedata.fake_name(detected)
        assert name != text
        assert _shape(name) == _shape(text)


def test_title_is_kept_and_first_name_agrees_with_it():
    detected = style.detect_name("Mrs. Delia Marsh")
    assert detected.gender == "female"
    fakedata.seed(0)
    assert all(fakedata.fake_name(detected).startswith("Mrs. ") for _ in range(10))


def test_hyphenated_surnames_stay_hyphenated():
    detected = style.detect_name("Delia Marsh-Oxley")
    fakedata.seed(0)
    assert re.search(r"\w-\w", fakedata.fake_name(detected).split()[-1])


def test_letters_in_unrecognized_tokens_are_replaced():
    detected = NameStyle((NameToken("first"), NameToken("literal", text="(Nell)"), NameToken("last")), last_first=False)
    fakedata.seed(0)
    for _ in range(10):
        name = fakedata.fake_name(detected)
        assert "Nell" not in name and re.search(r"\(\w+\)", name)


def test_capitalization_is_kept_per_word():
    detected = style.detect("name", "HARTLEY, Jonas P.")
    fakedata.seed(0)
    for _ in range(10):
        last, first = fakedata.fake_text("name", detected).split(" ")[:2]
        assert last.rstrip(",").isupper() and not first.isupper()


def test_title_just_left_of_the_box_sets_gender():
    from rvlcdip_redact import render
    from rvlcdip_redact.annotations import Annotation, BBox
    from rvlcdip_redact.ocr import Word

    words = [Word("Mrs.", BBox(100, 50, 40, 20)), Word("Delia", BBox(150, 50, 60, 20)), Word("Marsh", BBox(220, 50, 70, 20))]
    box = BBox(146, 46, 150, 28)
    assert render.title_left_of(box, words) == "Mrs."
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=20, auto_font_size=False, fit_width=False)
    fakedata.seed(0)
    female = {n for n in (fakedata._faker.first_name_female() for _ in range(3000))}
    male = {n for n in (fakedata._faker.first_name_male() for _ in range(3000))}
    only_female = female - male
    for _ in range(15):
        first = render.plan_text(Annotation("name", box), render.Page(words=words), settings).text.split()[0]
        assert first not in (male - female)
    assert only_female  # sanity: the name lists differ


def test_all_caps_names_stay_all_caps():
    detected = style.detect("name", "HARTLEY, JONAS")
    fakedata.seed(0)
    assert fakedata.fake_text("name", detected).isupper()
