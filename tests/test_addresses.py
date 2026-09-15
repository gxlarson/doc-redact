"""Linking repeated addresses to shared fake components. All addresses here are invented."""
import re

import pytest

from doc_redact import addresses, fakedata, render, style
from doc_redact.annotations import Annotation, BBox
from doc_redact.ocr import Word


@pytest.mark.parametrize("text, street, city, state, zip_code", [
    ("12 Maple Avenue\nSpringfield, PA 19064", "12 Maple Avenue", "Springfield", "PA", "19064"),
    ("12 Maple Ave., Springfield, Pa. 19064", "12 Maple Ave.", "Springfield", "PA", "19064"),
    ("P.O. Box 481\nKeene, New Hampshire 03431-1234", "P.O. Box 481", "Keene", "NH", "034311234"),
    ("Route 2, Box 77\nLaurel Md. 20707", "Route 2, Box 77", "Laurel", "MD", "20707"),
    ("Springfield, Pennsylvania", None, "Springfield", "PA", None),
    ("Delia Marsh\nP.O. Box 481\nKeene NH 03431", "P.O. Box 481", "Keene", "NH", "03431"),  # name line on top
    ("P.O. Boy 481\nKeene NH 03431", "P.O. Boy 481", "Keene", "NH", "03431"),  # OCR misread of "Box"
    ("Rc. 7, Box 52\nLaurel, Md. 20707", "Rc. 7, Box 52", "Laurel", "MD", "20707"),  # misread of "Rt."
])
def test_parse_address(text, street, city, state, zip_code):
    parsed = style.parse_address(text)
    assert (parsed.street, parsed.city, parsed.state, parsed.zip) == (street, city, state, zip_code)


@pytest.mark.parametrize("code, form, text", [
    ("PA", "postal", "PA"), ("PA", "postal_dotted", "P.A."), ("PA", "abbrev", "Pa."),
    ("PA", "full", "Pennsylvania"), ("NM", "abbrev", "N.Mex."), ("UT", "abbrev", "Ut."),
])
def test_state_forms_round_trip(code, form, text):
    assert style.format_state(code, form) == text
    assert style.state_code(text) == code


def test_street_keys_ignore_abbreviations_and_case():
    assert addresses.same_street("12 Maple Avenue", "12 MAPLE AVE.")
    assert addresses.same_street("P.O. Box 481", "PO Box 481")
    assert not addresses.same_street("12 Maple Avenue", "14 Maple Avenue")
    assert not addresses.same_street("12 Maple Avenue", "12 Oak Avenue")


def _page(entries):
    words, annotations = [], []
    for category, text, left, top in entries:
        y = top
        right = left
        for line in text.split("\n"):
            x = left
            for token in line.split():
                words.append(Word(token, BBox(x, y, 11 * len(token), 20)))
                x += 11 * len(token) + 9
            right = max(right, x)
            y += 30
        annotations.append(Annotation(category, BBox(left - 3, top - 4, right - left + 6, y - top + 2)))
    return words, annotations


def test_repeated_address_shares_components_in_each_boxes_format():
    words, annotations = _page([
        ("address", "12 Maple Avenue\nSpringfield, PA 19064", 100, 100),
        ("home_address", "12 Maple Ave., Springfield, Pa. 19064", 100, 400),
        ("street", "12 Maple Ave", 100, 600),
        ("zip", "19064", 100, 700),
        ("home_address", "88 Elm Street\nKeene, NH 03431", 100, 800),
    ])
    fakedata.seed(2)
    linked = addresses.link(annotations, words)
    full, one_line, street_box, zip_box, other = (linked[a] for a in annotations)
    assert full.street == one_line.street == street_box.street
    assert (full.city, full.state, full.zip) == (one_line.city, one_line.state, one_line.zip)
    assert zip_box.zip == full.zip
    assert other.street != full.street and other.zip != full.zip

    page = render.Page(words=words, addresses=linked)
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=18, auto_font_size=False, fit_width=False)
    texts = [render.plan_text(a, page, settings).text for a in annotations]
    abbrev = style.format_state(full.state, "abbrev")
    assert texts[0] == f"{full.street}\n{full.city}, {full.state} {full.zip[:5]}"
    assert texts[1] == f"{full.street}, {full.city}, {abbrev} {full.zip[:5]}"
    assert texts[2] == full.street
    assert texts[3] == full.zip[:5]
    assert not re.search(r"Maple|Springfield|19064", "".join(texts))


def test_misread_street_name_still_links_by_house_number_and_zip():
    words, annotations = _page([
        ("home_address", "1204 Bellmont Pl\nKeene, NH 03431", 100, 100),
        ("home_address", "1204 Bellmant Bl.\nKeene, N.H. 03431", 100, 400),  # handwritten copy, OCR slips
        ("home_address", "1204 Bellmont Pl\nLaurel, MD 20707", 100, 700),    # same number, different zip
    ])
    fakedata.seed(0)
    linked = addresses.link(annotations, words)
    assert linked[annotations[0]].street == linked[annotations[1]].street
    assert linked[annotations[2]].street == linked[annotations[0]].street  # identical street text still links
    words, annotations = _page([
        ("home_address", "1204 Bellmont Pl\nKeene, NH 03431", 100, 100),
        ("home_address", "1204 Harrow Rd\nLaurel, MD 20707", 100, 400),
    ])
    linked = addresses.link(annotations, words)
    assert linked[annotations[0]].street != linked[annotations[1]].street


def test_route_box_misread_in_one_copy_still_links():
    words, annotations = _page([
        ("home_address", "Mr. Ansel Quimby\nRt. 7, Box 52\nLaurel, Md. 20707", 100, 100),
        ("home_address", "Rc. 7, Box 52\nLaurel, Hd. 20707", 100, 500),
    ])
    fakedata.seed(0)
    linked = addresses.link(annotations, words)
    first, second = linked[annotations[0]], linked[annotations[1]]
    assert first.street == second.street and first.zip == second.zip
    assert (first.city, first.state) == (second.city, second.state)  # same zip despite the garbled state


def test_po_box_format_is_kept_with_a_new_number_shared_across_boxes():
    words, annotations = _page([
        ("home_address", "P.O. Box 481\nKeene, NH 03431", 100, 100),
        ("street", "PO Box 481", 100, 400),
    ])
    fakedata.seed(0)
    linked = addresses.link(annotations, words)
    fake = linked[annotations[0]].street
    assert re.fullmatch(r"P\.O\. Box \d{3}", fake) and fake != "P.O. Box 481"
    assert linked[annotations[1]].street == fake


def test_marital_status_says_child_for_one():
    fakedata.seed(0)
    texts = {fakedata.fake_marital_status() for _ in range(300)}
    assert "Married, 1 child" in texts and not any("1 children" in t for t in texts)
