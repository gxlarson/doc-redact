"""Keep one fake value per real value within a document, for SSNs, phones, dates and similar fields.

The same real value appearing in several boxes on a page (an SSN on two
forms, a birth date written "4/12/50" and "April 12, 1950") gets one fake
value, rendered in each box's own format. Values that appear once are left
to the normal generators. Names and addresses have their own linkers
(people.py, addresses.py).

The original values are only compared in memory; nothing about them is kept.
"""

from __future__ import annotations

import calendar
import random
import re
from dataclasses import dataclass

from doc_redact import fakedata, geometry, style
from doc_redact.annotations import Annotation
from doc_redact.categories import canonical
from doc_redact.ocr import Word
from doc_redact.people import similar_names
from doc_redact.resources import US_STATES

# marital_status is not here: a document's marital boxes share one fakedata.Family instead
# (status and children are often annotated as separate boxes, e.g. "Married" and "2 children").
TEXT_CATEGORIES = {"citizenship_nationality", "religion"}
VALUE_CATEGORIES = {"ssn", "home_phone", "birth_date", "birth_place", "age"} | TEXT_CATEGORIES


@dataclass(frozen=True)
class FakeDigits:
    digits: str


@dataclass(frozen=True)
class FakeDate:
    year: int
    month: int
    day: int


@dataclass(frozen=True)
class FakePlace:
    city: str
    state: str  # postal code


@dataclass
class SharedText:
    """Filled by the first box that renders it; later boxes reuse the same text."""

    value: str | None = None


def _key(category: str, text: str):
    """A hashable key identifying the real value, or None if this box's value can't be read."""
    digits = re.sub(r"\D", "", text)
    if category == "ssn":
        return digits if len(digits) == 9 else None
    if category == "home_phone":
        return digits[-7:] if len(digits) in (7, 10, 11) else None
    if category == "age":
        return digits if 1 <= len(digits) <= 3 else None
    if category == "birth_date":
        parsed = style.parse_date(text.rstrip(",;:. "))
        return parsed[1:] if parsed else None
    if category == "birth_place":
        flat = " ".join(text.split()).rstrip(",;:.")
        if "," in flat:
            city, state = flat.rsplit(",", 1)
            code = style.state_code(state)
            if code:
                return (city.strip(), code)
        return (flat, None) if flat else None
    letters = re.sub(r"[^a-z]", "", text.lower())
    return letters or None


def _same(category: str, a, b) -> bool:
    if category == "birth_place":
        return similar_names(a[0], b[0]) and (a[1] is None or b[1] is None or a[1] == b[1])
    return a == b


def _new_value(category: str):
    if category == "ssn":
        return FakeDigits(re.sub(r"\D", "", fakedata._faker.ssn()))
    if category == "home_phone":
        return FakeDigits(f"{random.randint(2, 9)}{random.randint(0, 99):02d}{random.randint(2, 9)}"
                          f"{random.randint(0, 999999):06d}")
    if category == "age":
        return FakeDigits(str(random.randint(29, 59)))
    if category == "birth_date":
        year, month = random.randint(1930, 1970), random.randint(1, 12)
        return FakeDate(year, month, random.randint(1, calendar.monthrange(year, month)[1]))
    if category == "birth_place":
        return FakePlace(fakedata.fake_city(), random.choice(sorted(US_STATES)))
    return SharedText()


def link(annotations: list[Annotation], words: list[Word]) -> dict[Annotation, object]:
    """Shared fake values for boxes whose real value appears more than once on the page."""
    groups: list[tuple[str, object, list[Annotation]]] = []  # (category, key, boxes)
    for annotation in annotations:
        category = canonical(annotation.category)
        if category not in VALUE_CATEGORIES:
            continue
        text = geometry.reading_text(geometry.group_lines(geometry.words_inside(annotation.bbox, words)))
        key = _key(category, text) if text.strip() else None
        if key is None:
            continue
        for group_category, group_key, boxes in groups:
            if group_category == category and _same(category, group_key, key):
                boxes.append(annotation)
                break
        else:
            groups.append((category, key, [annotation]))

    linked: dict[Annotation, object] = {}
    for category, _, boxes in groups:
        if len(boxes) < 2:
            continue
        value = _new_value(category)
        for annotation in boxes:
            linked[annotation] = value
    return linked
