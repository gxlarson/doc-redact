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

# Leading three digits of the ZIP codes each state uses, as published by the USPS. A fake ZIP drawn from
# its state's range keeps "city, ST 12345" from reading wrong to anyone who knows the area.
ZIP3_RANGES = {
    "AL": [(350, 369)], "AK": [(995, 999)], "AZ": [(850, 865)], "AR": [(716, 729)], "CA": [(900, 961)],
    "CO": [(800, 816)], "CT": [(60, 69)], "DE": [(197, 199)], "FL": [(320, 349)], "GA": [(300, 319), (398, 399)],
    "HI": [(967, 968)], "ID": [(832, 838)], "IL": [(600, 629)], "IN": [(460, 479)], "IA": [(500, 528)],
    "KS": [(660, 679)], "KY": [(400, 427)], "LA": [(700, 714)], "ME": [(39, 49)], "MD": [(206, 219)],
    "MA": [(10, 27)], "MI": [(480, 499)], "MN": [(550, 567)], "MS": [(386, 397)], "MO": [(630, 658)],
    "MT": [(590, 599)], "NE": [(680, 693)], "NV": [(889, 898)], "NH": [(30, 38)], "NJ": [(70, 89)],
    "NM": [(870, 884)], "NY": [(100, 149)], "NC": [(270, 289)], "ND": [(580, 588)], "OH": [(430, 459)],
    "OK": [(730, 749)], "OR": [(970, 979)], "PA": [(150, 196)], "RI": [(28, 29)], "SC": [(290, 299)],
    "SD": [(570, 577)], "TN": [(370, 385)], "TX": [(750, 799)], "UT": [(840, 847)], "VT": [(50, 59)],
    "VA": [(220, 246)], "WA": [(980, 994)], "WV": [(247, 268)], "WI": [(530, 549)], "WY": [(820, 831)],
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
