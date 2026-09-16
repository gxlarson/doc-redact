"""Generators for plausible replacement text, one per PII category.

Each generator takes an optional style detail from :mod:`doc_redact.style`;
without one it picks a random style.
"""

from __future__ import annotations

import calendar
import random
import re
from dataclasses import dataclass
from functools import cache
from typing import Callable

from faker import Faker

from doc_redact import style as st
from doc_redact.categories import canonical
from doc_redact.resources import US_STATES, ZIP3_RANGES, state_abbreviations, us_cities

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
                out.append(ch)  # a long-distance "1": the area code that follows still leads the number
            else:
                out.append(str(random.randint(1 if first and first_nonzero else 0, 9)))
                first = False
        else:
            out.append(ch)
    return "".join(out)


def ssn_digits() -> str:
    """Nine digits in the shape the SSA issues: area 001-899 (not 666), group 01-99, serial 0001-9999.

    These are ordinary-looking numbers, which is the point: the never-issued
    alternatives (a 9xx area, a 00 group, a 0000 serial) each put the same tell
    on every page of a corpus. The cost is that a generated number can coincide
    with a real one, so the value is meaningless but not provably unassigned.
    """
    area = random.randint(1, 899)
    while area == 666:
        area = random.randint(1, 899)
    return f"{area:03d}{random.randint(1, 99):02d}{random.randint(1, 9999):04d}"


def _pour_digits(template: str, digits: str) -> str:
    """Write ``digits`` into the template's digit positions, in order."""
    source = iter(digits)
    return re.sub(r"\d", lambda _: next(source, "0"), template)


def _shared(generator):
    """Let a text generator reuse one value across boxes linked by values.link (a SharedText identity)."""
    def wrapped(detail=None, identity=None):
        if identity is None:
            return generator(detail)
        if identity.value is None:
            identity.value = generator(detail)
        return identity.value
    wrapped.__name__ = generator.__name__
    return wrapped


def fake_ssn(detail=None, identity=None) -> str:
    d = identity.digits if identity is not None else ssn_digits()
    if isinstance(detail, st.DigitTemplate):
        # Partial fields ("last 4") have fewer than nine digits; fill those rather than truncating an SSN.
        if len(re.findall(r"\d", detail.template)) == len(d):
            return _pour_digits(detail.template, d)
        return _fill_digits(detail.template, first_nonzero=True)
    return f"{d[:3]}-{d[3:5]}-{d[5:]}"


def fake_phone(detail=None, identity=None) -> str:
    if identity is not None:
        d = identity.digits  # 10 digits
        if isinstance(detail, st.DigitTemplate):
            n = len(re.findall(r"\d", detail.template))
            return _pour_digits(detail.template, d[-7:] if n == 7 else ("1" + d if n == 11 else d))
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    if isinstance(detail, st.DigitTemplate):
        eleven = len(re.findall(r"\d", detail.template)) == 11
        return _fill_digits(detail.template, keep_leading_one=eleven, first_nonzero=True)
    area = "".join(str(random.randint(1, 9)) for _ in range(3))
    area = f"{area}-" if random.randint(0, 2) == 0 else f"({area}) "
    return f"{area}{random.randint(0, 999):03d}-{random.randint(0, 9999):04d}"


def fake_other_id(detail=None, identity=None) -> str:
    """Account, case, policy or licence numbers: 7-12 digits, sometimes grouped by dashes.

    When the original was digits and separators, its layout is kept (see style.detect_other_id);
    otherwise a shape is invented, since these numbers follow no common format.
    """
    if identity is not None:
        return _pour_digits(detail.template, identity.digits) if isinstance(detail, st.DigitTemplate) else identity.digits
    if isinstance(detail, st.DigitTemplate):
        return _fill_digits(detail.template, first_nonzero=True)
    digits = str(random.randint(1, 9)) + "".join(str(random.randint(0, 9)) for _ in range(random.randint(6, 11)))
    if random.randint(0, 2) == 0:  # grouped: "482-91-7734", "77394-2210"
        cut = random.randint(2, max(3, len(digits) - 3))
        return "-".join([digits[:cut]] + ([digits[cut:cut + 2], digits[cut + 2:]] if len(digits) - cut > 4 else [digits[cut:]]))
    return digits


def fake_age(detail=None, identity=None) -> str:
    age = identity.digits if identity is not None else str(random.randint(29, 59))
    if isinstance(detail, st.DigitTemplate):
        return re.sub(r"\d+", age, detail.template, count=1)
    return age


@_shared
def fake_religion(detail=None) -> str:
    return random.choice(RELIGIONS)


@dataclass(frozen=True)
class Family:
    """One document's fake marital status and number of children, shared by all its marital boxes."""

    status: str  # "Married", "Single", ...
    children: int


def fake_family() -> Family:
    children = random.choice([0, 0, 1, 2, 2, 3, 4])
    statuses = ["Married"] * 6 + ["Divorced", "Widowed"] + ([] if children else ["Single"] * 3)
    return Family(status=random.choice(statuses), children=children)


def _children_text(style: st.ChildrenStyle, n: int) -> str:
    noun = style.noun if n != 1 else {"children": "child", "kids": "kid", "daughters": "daughter", "sons": "son"}[style.noun]
    if style.label_first:
        label = noun[:1].upper() + noun[1:] if style.capitalized else noun
        count = "None" if n == 0 else str(n) if style.number == "digit" else st.NUMBER_WORDS[n] if n <= 10 else str(n)
        return f"{label}{':' if style.colon else ''} {count}"
    if n == 0:
        count = "no"  # "no children" reads naturally where "0 children" does not
    elif style.number == "digit":
        count = str(n)
    else:
        count = st.NUMBER_WORDS[n] if n <= 10 else str(n)
    text = f"{count} {noun if n != 0 else style.noun}"
    return text[:1].upper() + text[1:] if style.capitalized else text


def fake_marital_status(detail=None, identity=None) -> str:
    """Status and/or children in the original's shape; ``identity`` (a Family) keeps a document's boxes consistent."""
    family = identity if isinstance(identity, Family) else fake_family()
    status = family.status if detail is None or getattr(detail, "capitalized", True) else family.status.lower()
    if isinstance(detail, st.DigitTemplate):
        # Coded entries: the children code ("c. 2.") carries the family's count; other codes get new digits.
        def code(m: re.Match) -> str:
            letter, rest = m.group(1), m.group(2)
            numbers = re.findall(r"\d+", rest)
            if letter.lower() == "c" and len(numbers) == 1:
                return letter + rest.replace(numbers[0], str(family.children), 1)
            return letter + _fill_digits(rest)
        return re.sub(r"([A-Za-z])(\.?\s*\d[\d,\s]*)", code, detail.template)
    if isinstance(detail, st.MaritalStyle) and detail.short:
        return family.status[0] + ("." if detail.dot else "")
    if isinstance(detail, st.StatusStyle):
        return status
    if isinstance(detail, st.ChildrenStyle):
        return _children_text(detail, family.children)
    if isinstance(detail, st.CombinedStyle):
        status = family.status if detail.status.capitalized else family.status.lower()
        return f"{status}{detail.separator}{_children_text(detail.children, family.children)}"
    if family.children:
        return f"{family.status}, {family.children} {'child' if family.children == 1 else 'children'}"
    return family.status


# Street and city names are drawn from real US naming, not invented: Faker's cities ("Wallacemouth")
# and streets ("Kertzmann Fords") read as obviously fake on a scanned form.
STREET_SUFFIXES = ["St", "St.", "Street", "Ave", "Ave.", "Avenue", "Rd", "Rd.", "Road", "Dr", "Dr.", "Drive",
                   "Ln", "Lane", "Blvd", "Blvd.", "Boulevard", "Ct", "Court", "Pl", "Place", "Way", "Circle",
                   "Terrace", "Parkway", "Highway", "Trail"]
STREET_WORDS = [
    "Maple", "Oak", "Elm", "Cedar", "Pine", "Walnut", "Chestnut", "Willow", "Spruce", "Birch", "Poplar",
    "Laurel", "Magnolia", "Hickory", "Sycamore", "Dogwood", "Juniper", "Aspen", "Cypress",
    "Washington", "Lincoln", "Jefferson", "Jackson", "Madison", "Monroe", "Adams", "Franklin", "Grant",
    "Harrison", "Wilson", "Taylor", "Garfield", "Hayes", "Cleveland", "Kennedy", "Roosevelt",
    "Hill", "Lake", "River", "Valley", "Ridge", "Meadow", "Forest", "Spring", "Summit", "Highland",
    "Fairview", "Riverside", "Lakeview", "Sunset", "Woodland", "Orchard", "Prairie", "Brook", "Creek",
    "Church", "School", "Market", "Union", "Center", "Park", "Main", "Broad", "Liberty", "Mill",
    "Depot", "College", "Bridge", "Garden", "Grove", "Clark", "Morgan", "Bennett", "Sherman", "Bradley",
]
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
            "14th", "17th", "21st", "23rd", "34th", "42nd", "56th", "63rd", "72nd", "81st", "96th"]
DIRECTIONALS = ["N", "S", "E", "W", "N.", "S.", "E.", "W.", "North", "South", "East", "West"]
UNITS = ["Apt {n}", "Apt. {n}", "# {n}", "Unit {n}", "Suite {n}", "Rm {n}"]


@cache
def _cities_up_to(max_length: int) -> tuple[str, ...]:
    return tuple(c for c in us_cities() if len(c) <= max_length) or us_cities()


def _real_city(max_length: int) -> str:
    """A real US city name, short enough for the field."""
    return random.choice(_cities_up_to(max_length))


@cache
def _place_street_names() -> tuple[str, ...]:
    """Single-word real place names, which US streets are commonly named after ("Bristol Rd")."""
    return tuple(c for c in us_cities() if " " not in c and 4 <= len(c) <= 11)


def _street_name() -> str:
    """One street name. The pool is wide on purpose: a corpus has thousands of addresses in it,
    and a short list would put the same few names on page after page."""
    r = random.random()
    if r < 0.25:
        return random.choice(STREET_WORDS)  # trees, presidents, features: the classic names
    if r < 0.60:
        return _faker.last_name()  # surnames are the other classic source ("Bennett Ave")
    if r < 0.92:
        # Place names widen the pool; a few read oddly as a street, so they stay a minority.
        return random.choice(_place_street_names())
    return random.choice(ORDINALS)


def _street_line() -> str:
    """A street line in common US form: number, optional direction, name, suffix, optional unit."""
    number = random.choice([random.randint(1, 99), random.randint(100, 999), random.randint(1000, 9999)])
    name = _street_name()
    parts = [str(number)]
    if random.randint(0, 5) == 0:
        parts.append(random.choice(DIRECTIONALS))
    parts += [name, random.choice(STREET_SUFFIXES)]
    if random.randint(0, 9) == 0:
        parts.append(random.choice(UNITS).format(n=random.randint(1, 40)))
    return " ".join(parts)


def fake_address(detail=None, max_line_length: int = 25, identity=None) -> str:
    """US address: "street\\ncity, ST 12345", or just the street or city line.

    ``identity`` (an addresses.FakeAddress) supplies components shared with other
    boxes on the page; missing components are generated.
    """
    if not isinstance(detail, st.AddressStyle):
        detail = st.AddressStyle(part="full", has_zip=True, state="postal")

    street = identity.street if identity and identity.street else None
    while street is None or len(street) > max_line_length:
        street = _street_line()
    if detail.part == "street":
        return street

    city = identity.city if identity and identity.city else fake_city()
    code = identity.state if identity and identity.state else _state_code(detail.state or "postal")
    city_line = f"{city}, {st.format_state(code, detail.state or 'postal')}"
    if detail.has_zip:
        city_line += f" {identity.zip[:5] if identity and identity.zip else zip_for_state(code)}"
    if detail.part == "city":
        return city_line
    return f"{street}\n{city_line}"


def fake_date(detail=None, identity=None) -> str:
    style = detail if isinstance(detail, (st.NumericDate, st.TextDate)) else st.random_date_style()
    if identity is not None:
        return st.format_date(style, identity.year, identity.month, identity.day)
    year = random.randint(1930, 1970)
    month = random.randint(1, 12)
    day = random.randint(1, calendar.monthrange(year, month)[1])
    return st.format_date(style, year, month, day)


def random_state_code() -> str:
    """One of the 50 states, as a postal code (Faker's state_abbr can return territories)."""
    return random.choice(sorted(US_STATES))


def zip_for_state(code: str) -> str:
    """A ZIP from the state's own range, so the city line doesn't read wrong to a local."""
    ranges = ZIP3_RANGES.get(code)
    if not ranges:
        return f"{random.randint(1000, 99999):05d}"
    lo, hi = random.choice(ranges)
    return f"{random.randint(lo, hi):03d}{random.randint(0, 99):02d}"


@cache
def _codes_for_form(form: str) -> tuple[str, ...]:
    """States that read back as ``form``. A traditional abbreviation of dotted initials ("N.H.")
    is indistinguishable from the postal_dotted style, so those states sit out that form."""
    codes = tuple(c for c in sorted(US_STATES) if st.classify_state(st.format_state(c, form)) == form)
    return codes or tuple(sorted(US_STATES))


def _state_code(form: str) -> str:
    return random.choice(_codes_for_form(form))


def _state(form: str) -> str:
    return st.format_state(_state_code(form), form)


def fake_birthplace(detail=None, max_city_length: int = 15, identity=None) -> str:
    style = detail if isinstance(detail, st.PlaceStyle) else st.random_place_style()
    if identity is not None:
        if style.state == "none":
            return identity.city
        return f"{identity.city}, {st.format_state(identity.state, style.state)}"
    city = _real_city(max_city_length)
    if style.state == "none":
        return city
    return f"{city}, {_state(style.state)}"


@_shared
def fake_nationality(detail=None) -> str:
    return random.choice(NATIONALITIES)


def fake_street(detail=None, identity=None) -> str:
    return fake_address(st.AddressStyle(part="street", has_zip=False), identity=identity)


def fake_city(detail=None, max_length: int = 15, identity=None) -> str:
    if identity and identity.city:
        return identity.city
    return _real_city(max_length)


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
    return zip_for_state(random_state_code())


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
        if tok.kind in ("title", "salutation"):
            part = tok.text  # "Dear", "Mr." -- not PII; titles keep gender agreement with the first name
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
    "other_id": fake_other_id,
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
IDENTITY_CATEGORIES = {
    "home_address", "street", "city", "state", "zip",  # addresses.link
    "ssn", "home_phone", "birth_date", "birth_place", "age", "other_id",  # values.link
    "citizenship_nationality", "religion", "marital_status",
}
