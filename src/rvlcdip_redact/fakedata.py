"""Generators for plausible replacement text, one per PII category.

Each generator takes an optional style detail from :mod:`rvlcdip_redact.style`;
without one it picks a random style.
"""

from __future__ import annotations

import calendar
import random
import re
from typing import Callable

from faker import Faker

from rvlcdip_redact import style as st
from rvlcdip_redact.categories import canonical
from rvlcdip_redact.resources import state_abbreviations, us_cities

_faker = Faker()

RELIGIONS = ["Hindu", "Christian", "Protestant", "Catholic", "Orthodox", "Buddhist", "Jewish"]
NATIONALITIES = [
    "Canadian", "American", "British", "English", "German", "South African", "Nigerian",
    "Israeli", "French", "Spanish", "Mexican", "Brazilian", "Irish", "Italian", "Korean",
    "Japanese", "Chinese", "Taiwanese", "Australian", "Egyptian", "Indian",
    "U.S.", "USA", "U.S.A.", "United States",
]


def seed(value: int) -> None:
    random.seed(value)
    Faker.seed(value)


def _fill_digits(template: str, keep_leading_one: bool = False, first_nonzero: bool = False) -> str:
    out = []
    first = True
    for ch in template:
        if ch.isdigit():
            if first and keep_leading_one and ch == "1":
                out.append(ch)
            else:
                out.append(str(random.randint(1 if first and first_nonzero else 0, 9)))
            first = False
        else:
            out.append(ch)
    return "".join(out)


def fake_ssn(detail=None) -> str:
    if isinstance(detail, st.DigitTemplate):
        return _fill_digits(detail.template, first_nonzero=True)
    return _faker.ssn()


def fake_phone(detail=None) -> str:
    if isinstance(detail, st.DigitTemplate):
        eleven = len(re.findall(r"\d", detail.template)) == 11
        return _fill_digits(detail.template, keep_leading_one=eleven, first_nonzero=True)
    area = "".join(str(random.randint(1, 9)) for _ in range(3))
    area = f"{area}-" if random.randint(0, 2) == 0 else f"({area}) "
    return f"{area}{random.randint(0, 999):03d}-{random.randint(0, 9999):04d}"


def fake_age(detail=None) -> str:
    age = str(random.randint(29, 59))
    if isinstance(detail, st.DigitTemplate):
        return re.sub(r"\d+", age, detail.template, count=1)
    return age


def fake_religion(detail=None) -> str:
    return random.choice(RELIGIONS)


def fake_marital_status(detail=None) -> str:
    if isinstance(detail, st.DigitTemplate):
        return _fill_digits(detail.template)
    if isinstance(detail, st.MaritalStyle) and detail.short:
        return random.choice("SM") + ("." if detail.dot else "")
    kind = random.randint(0, 3)
    if kind == 0:
        return "Single"
    if kind == 1:
        n = random.randint(1, 4)
        return f"Married, {n} {'child' if n == 1 else 'children'}"
    return "Married"


def fake_address(detail=None, max_line_length: int = 25, identity=None) -> str:
    """US address: "street\\ncity, ST 12345", or just the street or city line.

    ``identity`` (an addresses.FakeAddress) supplies components shared with other
    boxes on the page; missing components are generated.
    """
    if not isinstance(detail, st.AddressStyle):
        detail = st.AddressStyle(part="full", has_zip=True, state="postal")

    street = identity.street if identity and identity.street else None
    while street is None or len(street) > max_line_length:
        street = _faker.street_address()
    if detail.part == "street":
        return street

    city = identity.city if identity and identity.city else fake_city()
    if identity and identity.state:
        state = st.format_state(identity.state, detail.state or "postal")
    else:
        state = _state(detail.state or "postal")
    city_line = f"{city}, {state}"
    if detail.has_zip:
        city_line += f" {identity.zip[:5] if identity and identity.zip else _faker.zipcode()}"
    if detail.part == "city":
        return city_line
    return f"{street}\n{city_line}"


def fake_date(detail=None) -> str:
    style = detail if isinstance(detail, (st.NumericDate, st.TextDate)) else st.random_date_style()
    year = random.randint(1930, 1970)
    month = random.randint(1, 12)
    day = random.randint(1, calendar.monthrange(year, month)[1])
    return st.format_date(style, year, month, day)


def _state(form: str) -> str:
    if form == "abbrev":
        return random.choice(state_abbreviations())
    if form == "full":
        return _faker.state()
    postal = _faker.state_abbr(include_territories=False)
    if form == "title_dot":
        return postal.title() + "."
    if form == "postal_dotted":
        return f"{postal[0]}.{postal[1]}."
    return postal


def fake_birthplace(detail=None, max_city_length: int = 15) -> str:
    style = detail if isinstance(detail, st.PlaceStyle) else st.random_place_style()
    city_source = _faker.city if random.randint(0, 3) == 0 else (lambda: random.choice(us_cities()))
    city = city_source()
    while len(city) > max_city_length:
        city = city_source()
    if style.state == "none":
        return city
    return f"{city}, {_state(style.state)}"


def fake_nationality(detail=None) -> str:
    return random.choice(NATIONALITIES)


def fake_street(detail=None, identity=None) -> str:
    return fake_address(st.AddressStyle(part="street", has_zip=False), identity=identity)


def fake_city(detail=None, max_length: int = 15, identity=None) -> str:
    if identity and identity.city:
        return identity.city
    # Mostly real city names; Faker's invented ones ("Lake Amberfurt") a third of the time.
    city_source = _faker.city if random.randint(0, 2) == 0 else (lambda: random.choice(us_cities()))
    city = city_source()
    while len(city) > max_length:
        city = city_source()
    return city


def fake_state(detail=None, identity=None) -> str:
    form = detail.state if isinstance(detail, st.PlaceStyle) and detail.state != "none" else None
    form = form or st.random_place_style().state
    if identity and identity.state:
        return st.format_state(identity.state, form)
    return _state(form)


def fake_zip(detail=None, identity=None) -> str:
    if identity and identity.zip:
        if isinstance(detail, st.DigitTemplate):
            digits = iter(identity.zip)
            return re.sub(r"\d", lambda _: next(digits, "0"), detail.template)
        return identity.zip[:5]
    if isinstance(detail, st.DigitTemplate):
        return _fill_digits(detail.template, first_nonzero=True)
    return _faker.zipcode()


INITIALS = "ABCDEFGHJKLMNPRSTW"  # letters that commonly start given names
NAME_SUFFIXES = ["Jr", "Sr", "II", "III"]


def _random_name_style() -> st.NameStyle:
    first, last = st.NameToken("first"), st.NameToken("last")
    r = random.random()
    if r < 0.55:
        return st.NameStyle((first, last), last_first=False)
    if r < 0.8:
        return st.NameStyle((first, st.NameToken("initial", dot=True), last), last_first=False)
    return st.NameStyle((last, first), last_first=True)


def fake_name(detail=None) -> str:
    """A personal name in the same shape as the original (order, initials, title, suffix)."""
    style = detail if isinstance(detail, st.NameStyle) else _random_name_style()
    person = style.identity  # a people.Person when this name was linked to others on the page
    gender = person.gender if person else (style.gender or random.choice(["male", "female"]))
    given = _faker.first_name_male if gender == "male" else _faker.first_name_female

    parts, comma_done = [], False
    for tok in style.tokens:
        if tok.kind == "title":
            part = tok.text  # keeps gender agreement with the generated first name
        elif tok.kind in ("first", "middle"):
            part = (person.first if tok.kind == "first" else person.middle) if person else given()
            part = part.upper() if tok.upper else part
        elif tok.kind == "last":
            if person:
                part = person.last
            else:
                part = _faker.last_name()
                if tok.hyphenated:
                    part = f"{part}-{_faker.last_name()}"
            if tok.upper:
                part = part.upper()
        elif tok.kind == "initial":
            letter = (person.middle if tok.of == "middle" else person.first)[0] if person else random.choice(INITIALS)
            part = letter + ("." if tok.dot else "")
        elif tok.kind == "suffix":
            suffix = random.choice(NAME_SUFFIXES)
            part = suffix + ("." if tok.dot and suffix in ("Jr", "Sr") else "")
        elif re.search(r"[A-Za-z]", tok.text):
            # Unrecognized token with letters, e.g. a nickname in parentheses: replace the letters, keep the punctuation.
            part = re.sub(r"[A-Za-z][A-Za-z'-]*", lambda m: given(), tok.text)
        else:
            part = tok.text
        if style.last_first and tok.kind == "last" and not comma_done:
            part += ","
            comma_done = True
        parts.append(part)
    return " ".join(parts)


GENERATORS: dict[str, Callable[..., str]] = {  # keyed by canonical category (see categories.py)
    "age": fake_age,
    "birth_date": fake_date,
    "birth_place": fake_birthplace,
    "citizenship_nationality": fake_nationality,
    "home_address": fake_address,
    "home_phone": fake_phone,
    "marital_status": fake_marital_status,
    "religion": fake_religion,
    "ssn": fake_ssn,
    "street": fake_street,
    "city": fake_city,
    "state": fake_state,
    "zip": fake_zip,
    "name": fake_name,
}


def has_generator(category: str) -> bool:
    return canonical(category) in GENERATORS


def fake_text(category: str, style: st.Style | None = None) -> str:
    """Replacement text for ``category`` (any alias); raises KeyError for unknown categories."""
    generator = GENERATORS[canonical(category)]
    if style is None:
        return generator()
    if style.identity is not None and canonical(category) in IDENTITY_CATEGORIES:
        text = generator(style.detail, identity=style.identity)
    else:
        text = generator(style.detail)
    return st.apply_case(text, style.case) + style.suffix


# Generators that accept an ``identity`` shared with other boxes on the page.
IDENTITY_CATEGORIES = {"home_address", "street", "city", "state", "zip"}
