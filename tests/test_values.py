"""Linking repeated SSNs, phones, dates and similar values. All values here are invented."""
import re

import pytest

from rvlcdip_redact import fakedata, render, style, values
from rvlcdip_redact.annotations import Annotation, BBox
from rvlcdip_redact.ocr import Word


def _page(entries):
    words, annotations = [], []
    for i, (category, text) in enumerate(entries):
        top = 100 + 120 * i
        x = 100
        for token in text.split():
            words.append(Word(token, BBox(x, top, 11 * len(token), 20)))
            x += 11 * len(token) + 9
        annotations.append(Annotation(category, BBox(97, top - 4, x - 100 + 6, 28)))
    return words, annotations


def _render(entries, seed=0):
    words, annotations = _page(entries)
    fakedata.seed(seed)
    shared = values.link(annotations, words)
    page = render.Page(words=words, shared_values=shared)
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=18, auto_font_size=False, fit_width=False)
    return shared, annotations, [render.plan_text(a, page, settings).text for a in annotations]


def test_parse_date_values():
    assert style.parse_date("4/12/50")[1:] == (1950, 4, 12)
    assert style.parse_date("April 12, 1950")[1:] == (1950, 4, 12)
    assert style.parse_date("12 Apr 1950")[1:] == (1950, 4, 12)
    assert style.parse_date("25.12.1944")[1:] == (1944, 12, 25)
    assert style.parse_date("Sept. 3rd, 1961")[1:] == (1961, 9, 3)
    assert style.parse_date("13/13/50") is None


def test_same_ssn_in_different_layouts_gets_one_fake_number():
    shared, annotations, texts = _render([("ssn", "987-65-4320"), ("ssn_like", "987654320"), ("ssn", "987-65-4321")])
    assert shared[annotations[0]] is shared[annotations[1]]
    assert annotations[2] not in shared  # a different number that appears once is left alone
    assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", texts[0]) and re.fullmatch(r"\d{9}", texts[1])
    assert texts[0].replace("-", "") == texts[1] != "987654320"


def test_phone_with_and_without_area_code_matches_on_the_last_seven_digits():
    shared, annotations, texts = _render([("home_phone", "(555) 014-7788"), ("phone", "014-7788")])
    assert shared[annotations[0]] is shared[annotations[1]]
    assert re.sub(r"\D", "", texts[0])[-7:] == re.sub(r"\D", "", texts[1])
    assert re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", texts[0]) and re.fullmatch(r"\d{3}-\d{4}", texts[1])


def test_same_birth_date_in_different_formats_gets_one_fake_date():
    shared, annotations, texts = _render([
        ("birth_date", "4/12/50"), ("date_of_birth", "April 12, 1950"), ("birth_date", "12 Apr. 1950"),
    ])
    date = shared[annotations[0]]
    assert shared[annotations[1]] is date and shared[annotations[2]] is date
    assert style.parse_date(texts[0])[1:] == style.parse_date(texts[1])[1:] == (date.year, date.month, date.day)
    assert re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", texts[0])
    assert re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", texts[1])
    assert re.fullmatch(r"\d{1,2} [A-Z][a-z]+\.? \d{4}", texts[2])


def test_birth_place_shares_city_and_state_with_each_boxs_state_form():
    shared, annotations, texts = _render([("birth_place", "Keene, NH"), ("place_of_birth", "Keene, New Hampshire")])
    place = shared[annotations[0]]
    assert shared[annotations[1]] is place
    assert texts[0] == f"{place.city}, {place.state}"
    assert texts[1] == f"{place.city}, {style.format_state(place.state, 'full')}"


def test_age_and_text_values_repeat_consistently():
    shared, annotations, texts = _render([
        ("age", "43"), ("age", "43 yrs"), ("religion", "Lutheran"), ("religious_affiliation", "LUTHERAN"),
        ("marital_status", "Married"), ("marital_status", "married"),
    ], seed=3)
    assert texts[0] == re.sub(r"\D", "", texts[1]) and texts[1].endswith("yrs")
    assert texts[2].lower() == texts[3].lower() and texts[3].isupper()
    assert texts[4].lower() == texts[5].lower()
