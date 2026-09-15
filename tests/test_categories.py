import re

import numpy as np
import pytest
from PIL import Image

from doc_redact import checks, fakedata, layout, render, style, typeface
from doc_redact.annotations import Annotation, BBox
from doc_redact.categories import ALIASES, canonical
from doc_redact.ocr import Word


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_every_alias_has_a_generator(alias):
    assert fakedata.has_generator(alias)
    fakedata.seed(0)
    assert fakedata.fake_text(alias)


def test_address_uses_the_home_address_generator_and_style():
    detected = style.detect("address", "12 Maple Avenue")
    assert detected.detail == style.AddressStyle(part="street", has_zip=False)
    fakedata.seed(0)
    street = fakedata.fake_text("address", detected)
    assert "\n" not in street and re.match(r"\d", street)


def test_ssn_like_keeps_an_undashed_layout():
    detected = style.detect("ssn_like", "987654320")
    fakedata.seed(0)
    assert re.fullmatch(r"\d{9}", fakedata.fake_text("ssn_like", detected))


@pytest.mark.parametrize("text, form", [("TX", "postal"), ("Tex.", "abbrev"), ("Texas", "full"), ("T.X.", "postal_dotted")])
def test_state_boxes_keep_their_form(text, form):
    detected = style.detect("state", text)
    assert detected.detail == style.PlaceStyle(form)
    fakedata.seed(1)
    assert style.classify_state(fakedata.fake_text("state", detected)) in {form, "abbrev" if form == "title_dot" else form}


def test_zip_and_city_generators():
    fakedata.seed(0)
    assert re.fullmatch(r"\d{5}-\d{4}", fakedata.fake_text("zip", style.detect("zip", "20707-1234")))
    city = fakedata.fake_text("city")
    assert city and len(city) <= 15


def test_multiline_and_checks_follow_aliases():
    assert canonical("address") in render.MULTILINE_CATEGORIES
    words = [Word("June", BBox(10, 10, 40, 10)), Word("14,", BBox(55, 10, 20, 10)), Word("1938", BBox(80, 10, 30, 10))]
    # "place_of_birth" is checked as birth_place, so a date in it is suspicious...
    report = checks.check_annotation(0, Annotation("place_of_birth", BBox(0, 5, 150, 20)), words)
    assert "looks_like_date" in report.flags
    assert report.category == "place_of_birth"  # reports keep the label as annotated
    # ...while "address" is checked as home_address, where dates are not flagged.
    assert checks.check_annotation(1, Annotation("address", BBox(0, 5, 150, 20)), words).flags == []


def test_handwriting_size_is_capped_by_the_writers_ink_height():
    # A small hand (about 24 px tall ink) that a wide cursive font would otherwise render much taller.
    page = Image.new("RGB", (900, 200), "white")
    layout.draw_text(page, (40, 60), "Laboratory", "PatrickHand-Regular.ttf", 34)
    gray = np.asarray(page.convert("L"))
    left, top, right, bottom = layout.load_font("PatrickHand-Regular.ttf", 34).getbbox("Laboratory")
    word = Word("Laboratory", BBox(40 + left, 60 + top, right - left, bottom - top), handwritten=True)
    size = typeface.height_matched_size("HomemadeApple-Regular.ttf", [word], gray)
    ink_h = layout.load_font("HomemadeApple-Regular.ttf", round(size)).getbbox("Laboratory")
    assert abs((ink_h[3] - ink_h[1]) - (bottom - top)) <= 2


def test_height_cap_ignores_strokes_from_neighboring_lines_and_rules():
    # Tightly lined handwriting: descenders from the line above and ascenders from the line below
    # poke into the word's crop, and a ruled line runs under it.
    page = Image.new("RGB", (900, 260), "white")
    hand = "PatrickHand-Regular.ttf"
    font = layout.load_font(hand, 34)
    layout.draw_text(page, (40, 100), "555-014-7788", hand, 34)
    layout.draw_text(page, (60, 62), "gyp jqy gyp", hand, 34)   # descenders reaching down
    layout.draw_text(page, (50, 138), "Hlk bdf Hlk", hand, 34)  # ascenders reaching up
    arr = np.array(page)
    left, top, right, bottom = font.getbbox("555-014-7788")
    arr[100 + bottom + 2:100 + bottom + 4, 20:880] = 0
    gray = np.asarray(Image.fromarray(arr).convert("L"))
    word = Word("555-014-7788", BBox(40 + left, 100 + top - 4, right - left, bottom - top + 8), handwritten=True)
    assert typeface.height_matched_size(hand, [word], gray) == pytest.approx(34, rel=0.12)


def test_replacement_ink_starts_where_the_original_ink_starts():
    words = [Word("5-17-31", BBox(120, 100, 150, 40), handwritten=True)]
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=30, match_style=False,
                                   handwriting_font="HomemadeApple-Regular.ttf", overrides={"birth_date": "7-12-39"})
    plan = render.plan_text(Annotation("birth_date", BBox(110, 95, 170, 50)), render.Page(words=words), settings)
    font = layout.load_font(plan.font, plan.font_size)
    assert plan.xy[0] + layout.text_bbox(font, plan.text, plan.spacing)[0] == pytest.approx(120, abs=0.01)
