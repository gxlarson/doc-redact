"""Annotation category names.

Different annotation rounds used different labels for the same kind of PII
(e.g. ``address`` and ``home_address``). Everything that looks a category up
(generators, style detection, checks, layout) goes through :func:`canonical`.
"""

from __future__ import annotations

ALIASES = {
    "date_of_birth": "birth_date",
    "place_of_birth": "birth_place",
    "address": "home_address",
    "phone": "home_phone",
    "ssn_like": "ssn",
    "citizenship": "citizenship_nationality",
    "religious_affiliation": "religion",
}


def canonical(category: str) -> str:
    return ALIASES.get(category, category)
