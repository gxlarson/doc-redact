import random
import re

import pytest

from rvlcdip_redact import fakedata, style
from rvlcdip_redact.style import AddressStyle, DigitTemplate, MaritalStyle, NumericDate, PlaceStyle, TextDate


@pytest.mark.parametrize(
    "text, expected",
    [
        ("4/12/1950", NumericDate(sep="/", year_digits=4, pad=False, day_first=False)),
        ("04-02-50", NumericDate(sep="-", year_digits=2, pad=True, day_first=False)),
        ("25.12.1944", NumericDate(sep=".", year_digits=4, pad=False, day_first=True)),
        ("June 14, 1938", TextDate(month="full", day_first=False, comma=True, year_digits=4)),
        ("Oct.8,14", TextDate(month="abbr_dot", day_first=False, comma=True, year_digits=2)),
        ("14 Jun 1938", TextDate(month="abbr", day_first=True, comma=False, year_digits=4)),
        ("Sept. 3rd, 1961", TextDate(month="abbr_dot", day_first=False, comma=True, year_digits=4)),
        ("Marfa, Texas", None),
        ("married", None),
    ],
)
def test_detect_date(text, expected):
    assert style.detect_date(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Marfa, Texas", PlaceStyle("full")),
        ("Marfa, TX", PlaceStyle("postal")),
        ("Marfa, T.X.", PlaceStyle("postal_dotted")),
        ("Marfa, Tex.", PlaceStyle("abbrev")),
        ("Albany, N. Dak.", PlaceStyle("abbrev")),
        ("Marfa, Tx.", PlaceStyle("title_dot")),
        ("Chicago", PlaceStyle("none")),
        ("London, England", None),
    ],
)
def test_detect_place(text, expected):
    assert style.detect_place(text) == expected


def test_random_styles_round_trip_through_detection():
    random.seed(0)
    for _ in range(300):
        s = style.random_date_style()
        month = random.choice([m for m in range(1, 13) if m != 5])  # "May" is both full and abbreviated
        assert style.detect_date(style.format_date(s, 1950, month, random.randint(13, 28))) == s
    fakedata.seed(0)
    for _ in range(300):
        s = style.random_place_style()
        detected = style.detect_place(fakedata.fake_birthplace(s))
        # "Ky.", "Pa." and "La." are both title-cased postal codes and traditional abbreviations.
        assert detected == s or {detected.state, s.state} == {"abbrev", "title_dot"}


def test_digit_templates_keep_layout():
    fakedata.seed(0)
    ssn = fakedata.fake_ssn(DigitTemplate("123 45 6789"))
    assert len(ssn) == 11 and ssn[3] == " " and ssn[6] == " " and ssn != "123 45 6789"
    phone = fakedata.fake_phone(DigitTemplate("1 (202) 555-0100"))
    assert phone.startswith("1 (") and phone[3] != "0"
    assert style.detect_digits("SSN 12", {9}) is None


def test_case_and_short_forms():
    assert style.detect("religion", "CATHOLIC").case == "upper"
    fakedata.seed(0)
    assert fakedata.fake_text("religion", style.Style(case="upper")).isupper()
    assert fakedata.fake_marital_status(MaritalStyle(short=True, dot=True)) in {"S.", "M."}
    assert style.detect_marital("M") == MaritalStyle(short=True, dot=False)


@pytest.mark.parametrize(
    "category, original, detail_type, suffix",
    [
        ("birth_date", "February 3, 1941;", TextDate, ";"),
        ("birth_date", "August 9, 1936.", TextDate, "."),
        ("birth_date", "July 4, 07;", TextDate, ";"),
        ("birth_place", "Utica, N.Y.,", PlaceStyle, ","),
        ("birth_place", "Keene, N.H.", PlaceStyle, ""),
        ("marital_status", "Married,", MaritalStyle, ","),
        ("citizenship_nationality", "American;", type(None), ";"),
    ],
)
def test_trailing_punctuation_is_kept_as_suffix(category, original, detail_type, suffix):
    detected = style.detect(category, original)
    assert isinstance(detected.detail, detail_type)
    assert detected.suffix == suffix
    fakedata.seed(0)
    assert fakedata.fake_text(category, detected).endswith(suffix)


def test_coded_marital_status_keeps_layout():
    detected = style.detect("marital_status", "m. 51;")
    assert detected.detail == DigitTemplate("m. 51;")
    fakedata.seed(0)
    out = fakedata.fake_text("marital_status", detected)
    assert out.startswith("m. ") and out.endswith(";") and len(out) == len("m. 51;")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 Main St\nTown, ST 12345", AddressStyle(part="full", has_zip=True, state="postal")),
        ("Unit 4, Old Mill Road, Easton, Maryland 21601", AddressStyle(part="full", has_zip=True, state="full")),
        ("12 Maple Avenue", AddressStyle(part="street", has_zip=False)),
        ("4410 S.W. 82nd St.", AddressStyle(part="street", has_zip=False)),
        ("Springfield, PA 19064", AddressStyle(part="city", has_zip=True, state="postal")),
        ("Laurel, Maryland 20707", AddressStyle(part="city", has_zip=True, state="full")),
        ("88 Elm Street, Hempstead, L.I., N.Y.", AddressStyle(part="full", has_zip=False, state="postal_dotted")),
    ],
)
def test_detect_address(text, expected):
    assert style.detect_address(text) == expected


def test_address_parts_generate_matching_lines():
    fakedata.seed(0)
    for _ in range(20):
        street = fakedata.fake_address(AddressStyle(part="street", has_zip=False))
        assert "\n" not in street and "," not in street
        city = fakedata.fake_address(AddressStyle(part="city", has_zip=False, state="full"))
        assert "\n" not in city and not city[-1].isdigit()
        assert style.detect_place(city) == style.PlaceStyle("full")
        full = fakedata.fake_address(AddressStyle(part="full", has_zip=True, state="postal"))
        assert full.count("\n") == 1 and re.search(r", [A-Z]{2} \d{5}", full)
