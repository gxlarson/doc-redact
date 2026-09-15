"""Keep one fake address per real address within a document.

Linking works on components rather than whole addresses: within a page, the
same street always becomes the same fake street, the same city (and state)
the same fake city and state, the same zip the same fake zip. That covers an
address written twice in different formats, and separate street / city /
state / zip boxes that repeat parts of a full address, with one mechanism.
Each box still renders in its own format (state as "PA", "Pa." or
"Pennsylvania"; with or without the zip).

The original components are only compared in memory; nothing about them is kept.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from rvlcdip_redact import fakedata, geometry, style
from rvlcdip_redact.annotations import Annotation
from rvlcdip_redact.categories import canonical
from rvlcdip_redact.ocr import Word
from rvlcdip_redact.people import similar_names
from rvlcdip_redact.resources import US_STATES

ADDRESS_CATEGORIES = {"home_address", "street", "city", "state", "zip"}

_STREET_ABBREVIATIONS = {
    "street": "st", "avenue": "ave", "road": "rd", "drive": "dr", "lane": "ln", "boulevard": "blvd",
    "place": "pl", "court": "ct", "route": "rt", "rte": "rt", "apartment": "apt", "suite": "ste",
    "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "circle": "cir",
    "north": "n", "south": "s", "east": "e", "west": "w",
}


@dataclass(frozen=True)
class FakeAddress:
    """Replacement components for one box; None where the box had no such component."""

    street: str | None = None
    city: str | None = None
    state: str | None = None  # postal code
    zip: str | None = None  # 9 digits; boxes use the first 5 unless they showed ZIP+4


def street_key(street: str) -> tuple[str, str]:
    """(numbers, normalized words): "12 Maple Avenue" and "12 maple ave." give the same key."""
    tokens = re.findall(r"[A-Za-z]+|\d+", street.lower())
    if tokens[:3] == ["p", "o", "box"]:
        tokens = ["pobox"] + tokens[3:]
    numbers = " ".join(t for t in tokens if t.isdigit())
    words = " ".join(_STREET_ABBREVIATIONS.get(t, t) for t in tokens if not t.isdigit())
    return numbers, words


def same_street(a: str, b: str) -> bool:
    (numbers_a, words_a), (numbers_b, words_b) = street_key(a), street_key(b)
    return bool(numbers_a) and numbers_a == numbers_b and (words_a == words_b or similar_names(words_a, words_b))


class AddressBook:
    """Hands out one fake value per real street, city/state and zip on a page."""

    def __init__(self) -> None:
        self._streets: list[tuple[str, str | None, str]] = []  # (real street, real zip, fake street)
        self._cities: list[tuple[str, str | None, str, str]] = []  # (real city, real state, fake city, fake state)
        self._states: dict[str, str] = {}
        self._zips: dict[str, str] = {}

    def street(self, real: str, real_zip: str | None = None) -> str:
        for known, known_zip, fake in self._streets:
            if same_street(known, real):
                return fake
            # OCR can mangle a street name beyond one slip (typed vs handwritten copies);
            # the same house number in the same zip is still the same street.
            if real_zip and known_zip and real_zip[:5] == known_zip[:5] and street_key(known)[0] \
                    and street_key(known)[0] == street_key(real)[0]:
                return fake
        if style._PO_BOX.search(real):
            # Keep the P.O. box format, change the number.
            fake = re.sub(r"\d", lambda _: str(random.randint(0, 9)), real)
            fake = re.sub(r"\d", lambda _: str(random.randint(1, 9)), fake, count=1)
        else:
            fake = fakedata.fake_street()
        self._streets.append((real, real_zip, fake))
        return fake

    def state(self, real: str) -> str:
        if real not in self._states:
            self._states[real] = random.choice(sorted(US_STATES))
        return self._states[real]

    def city(self, real: str, real_state: str | None) -> tuple[str, str]:
        for known, known_state, fake_city, fake_state in self._cities:
            if similar_names(known, real) and (known_state is None or real_state is None or known_state == real_state):
                return fake_city, fake_state
        fake_state = self.state(real_state) if real_state else random.choice(sorted(US_STATES))
        fake_city = fakedata.fake_city()
        self._cities.append((real, real_state, fake_city, fake_state))
        return fake_city, fake_state

    def zip(self, real: str) -> str:
        key = real[:5]
        if key not in self._zips:
            self._zips[key] = f"{random.randint(10000, 99999)}{random.randint(0, 9999):04d}"
        return self._zips[key]


def parse_box(category: str, text: str) -> style.ParsedAddress | None:
    category = canonical(category)
    text = text.strip()
    if not text:
        return None
    if category == "home_address":
        return style.parse_address(text)
    if category == "street":
        return style.ParsedAddress(street=text)
    if category == "city":
        city, code = style._split_city_state(text.rstrip(","))
        return style.ParsedAddress(city=city, state=code)
    if category == "state":
        code = style.state_code(text.rstrip(","))
        return style.ParsedAddress(state=code) if code else None
    if category == "zip":
        digits = re.sub(r"\D", "", text)
        return style.ParsedAddress(zip=digits) if len(digits) in (5, 9) else None
    return None


def link(annotations: list[Annotation], words: list[Word]) -> dict[Annotation, FakeAddress]:
    """A FakeAddress per address-like box, sharing fake components wherever the real ones repeat."""
    book = AddressBook()
    parsed = []
    for annotation in annotations:
        if canonical(annotation.category) not in ADDRESS_CATEGORIES:
            continue
        text = geometry.reading_text(geometry.group_lines(geometry.words_inside(annotation.bbox, words)))
        address = parse_box(annotation.category, text)
        if address is not None:
            parsed.append((annotation, address))

    # Full addresses first, so a city's state is known before a bare city box asks for it.
    parsed.sort(key=lambda item: canonical(item[0].category) != "home_address")
    linked: dict[Annotation, FakeAddress] = {}
    city_for_zip: dict[str, tuple[str, str]] = {}  # a zip seen with a city keeps that fake city/state
    for annotation, address in parsed:
        street = book.street(address.street, address.zip) if address.street else None
        city = state = None
        zip5 = address.zip[:5] if address.zip else None
        if address.city and zip5 in city_for_zip:
            # The same zip is the same place, even when OCR garbled the city or state in this copy.
            city, state = city_for_zip[zip5]
        elif address.city:
            city, state = book.city(address.city, address.state)
            if zip5:
                city_for_zip[zip5] = (city, state)
        elif address.state:
            state = book.state(address.state)
        zip_code = book.zip(address.zip) if address.zip else None
        linked[annotation] = FakeAddress(street=street, city=city, state=state, zip=zip_code)
    return linked
