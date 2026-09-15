"""Word lists bundled in ``doc_redact/data``."""

from __future__ import annotations

from functools import cache
from importlib.resources import files


@cache
def word_list(name: str) -> tuple[str, ...]:
    text = files("doc_redact").joinpath("data", name).read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def us_cities() -> tuple[str, ...]:
    return word_list("us_cities.txt")


def state_abbreviations() -> tuple[str, ...]:
    """Traditional multi-letter abbreviations such as "Tex." and "Calif."."""
    return word_list("state_abbrev_multiletter.txt")


# Traditional (pre-postal-code) abbreviations, e.g. for "Springfield, Pa."; states without one use "Xx.".
TRADITIONAL_ABBREVIATIONS = {
    "AL": "Ala.", "AZ": "Ariz.", "AR": "Ark.", "CA": "Calif.", "CO": "Colo.", "CT": "Conn.", "DE": "Del.",
    "FL": "Fla.", "GA": "Ga.", "IL": "Ill.", "IN": "Ind.", "KS": "Kans.", "KY": "Ky.", "LA": "La.", "MD": "Md.",
    "MA": "Mass.", "MI": "Mich.", "MN": "Minn.", "MS": "Miss.", "MO": "Mo.", "MT": "Mont.", "NE": "Nebr.",
    "NV": "Nev.", "NH": "N.H.", "NJ": "N.J.", "NM": "N.Mex.", "NY": "N.Y.", "NC": "N.C.", "ND": "N.Dak.",
    "OK": "Okla.", "OR": "Oreg.", "PA": "Pa.", "RI": "R.I.", "SC": "S.C.", "SD": "S.Dak.", "TN": "Tenn.",
    "TX": "Tex.", "VT": "Vt.", "VA": "Va.", "WA": "Wash.", "WV": "W.Va.", "WI": "Wis.", "WY": "Wyo.",
}

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
