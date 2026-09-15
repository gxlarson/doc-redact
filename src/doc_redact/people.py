"""Keep one fake identity per real person within a document.

A page often repeats a name in different forms ("Last, First M." in one field,
"First M. Last" in another). Name boxes are grouped into people, and each
person gets one generated identity that every box fills in its own format.
People who share a surname (a family) also share a fake surname.

The original names are only compared in memory; nothing about them is kept.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from doc_redact import fakedata, geometry, style
from doc_redact.annotations import Annotation, BBox
from doc_redact.categories import canonical
from doc_redact.ocr import Word


@dataclass(frozen=True)
class Person:
    first: str
    middle: str
    last: str
    gender: str


def title_left_of(bbox: BBox, words: list[Word]) -> str | None:
    """A title ("Mr.", "Mrs.") printed just left of a name box on the same line, if any."""
    row = [w for w in geometry.words_in_row(bbox, words) if w.bbox.right <= bbox.left + 3]
    if not row:
        return None
    nearest = max(row, key=lambda w: w.bbox.right)
    if bbox.left - nearest.bbox.right > 2 * bbox.height:
        return None
    return nearest.text if nearest.text.rstrip(".").lower() in style.TITLES else None


def _letters(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _one_edit_apart(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def similar_names(a: str, b: str) -> bool:
    """Equal ignoring case and punctuation, or one OCR slip apart for longer names."""
    a, b = _letters(a), _letters(b)
    return a == b or (min(len(a), len(b)) >= 5 and _one_edit_apart(a, b))


def similar_given_names(a: str, b: str) -> bool:
    """Like similar_names, but short given names may also differ by one added or dropped letter.

    That covers variant spellings with an extra final letter, while two
    different short names that differ by a substituted letter stay distinct.
    """
    if similar_names(a, b):
        return True
    a, b = _letters(a), _letters(b)
    return min(len(a), len(b)) >= 4 and abs(len(a) - len(b)) == 1 and _one_edit_apart(a, b)


def same_person(a: style.ParsedName, b: style.ParsedName) -> bool:
    """Surnames match and the given names agree; without first-name evidence on both sides, no match.

    A bare "Mrs. Surname" next to "First Surname" could be a spouse, so it isn't merged.
    """
    if not (a.last and b.last and similar_names(a.last, b.last)):
        return False
    if not (a.first_letter and b.first_letter) or a.first_letter.lower() != b.first_letter.lower():
        return False
    if a.first and b.first and not similar_given_names(a.first, b.first):
        return False
    if a.middle_letter and b.middle_letter and a.middle_letter.lower() != b.middle_letter.lower():
        return False
    return True


def link(annotations: list[Annotation], words: list[Word]) -> dict[Annotation, Person]:
    """One Person per group of name boxes that refer to the same person."""
    parsed: list[tuple[Annotation, style.ParsedName, str | None]] = []
    for annotation in annotations:
        if canonical(annotation.category) != "name":
            continue
        text = geometry.reading_text(geometry.group_lines(geometry.words_inside(annotation.bbox, words)))
        name = style.parse_name(text)
        if name is not None:
            parsed.append((annotation, name, title_left_of(annotation.bbox, words)))

    # Union-find over the name boxes.
    parent = list(range(len(parsed)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            if same_person(parsed[i][1], parsed[j][1]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(parsed)):
        groups.setdefault(find(i), []).append(i)

    surnames: list[tuple[str, str]] = []  # (original surname, fake surname) shared across a family
    people: dict[Annotation, Person] = {}
    for members in groups.values():
        names = [parsed[i][1] for i in members]
        titles = [t.text for n in names for t in n.style.tokens if t.kind == "title"]
        titles += [parsed[i][2] for i in members if parsed[i][2]]
        genders = {style.title_gender(t) for t in titles} - {None}
        gender = genders.pop() if len(genders) == 1 else random.choice(["male", "female"])

        original_last = next((n.last for n in names if n.last), None)
        fake_last = None
        if original_last:
            fake_last = next((fake for real, fake in surnames if similar_names(real, original_last)), None)
        if fake_last is None:
            fake_last = fakedata._faker.last_name()
            if any(t.hyphenated for n in names for t in n.style.tokens if t.kind == "last"):
                fake_last = f"{fake_last}-{fakedata._faker.last_name()}"
            if original_last:
                surnames.append((original_last, fake_last))

        given = fakedata._faker.first_name_male if gender == "male" else fakedata._faker.first_name_female
        person = Person(first=given(), middle=given(), last=fake_last, gender=gender)
        for i in members:
            people[parsed[i][0]] = person
    return people
