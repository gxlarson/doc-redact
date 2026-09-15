"""Detect the surface format of original PII text so replacements can mimic it.

A style captures *how* a value is written (separators, abbreviations, casing,
digit layout) but never the value itself. Detectors return ``None`` when the
text isn't recognized, in which case generators pick a random style.
"""

from __future__ import annotations

import calendar
import dataclasses
import random
import re
from dataclasses import dataclass
from typing import Callable, Union

from doc_redact.categories import canonical
from doc_redact.resources import TRADITIONAL_ABBREVIATIONS, US_STATES, state_abbreviations

MONTHS = [calendar.month_name[i] for i in range(1, 13)]
MONTH_ABBREVIATIONS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]


# --- casing ---------------------------------------------------------------

def detect_case(text: str) -> str | None:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return None
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    return None


def apply_case(text: str, case: str | None) -> str:
    if case == "upper":
        return text.upper()
    if case == "lower":
        return text.lower()
    return text


# --- dates ----------------------------------------------------------------

@dataclass(frozen=True)
class NumericDate:
    sep: str  # "-", "/" or "."
    year_digits: int  # 2 or 4
    pad: bool  # zero-padded day/month
    day_first: bool


@dataclass(frozen=True)
class TextDate:
    month: str  # "full", "abbr" or "abbr_dot"
    day_first: bool
    comma: bool
    year_digits: int


DateStyle = Union[NumericDate, TextDate]

_NUMERIC_DATE = re.compile(r"(\d{1,2})\s*([-/.])\s*(\d{1,2})\s*\2\s*(\d{4}|\d{2})")
_MONTH_TOKEN = r"(?P<month>[A-Za-z]{3,9})(?P<dot>\.)?"
_MONTH_FIRST = re.compile(_MONTH_TOKEN + r"\s*(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*(?P<comma>,)?\s*(?P<year>\d{4}|\d{2})")
_DAY_FIRST = re.compile(r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_TOKEN + r"\s*(?P<comma>,)?\s*(?P<year>\d{4}|\d{2})")


def _month_form(token: str, dot: bool) -> str | None:
    lower = token.lower()
    if lower in (m.lower() for m in MONTHS):
        return "full"
    if any(m.lower().startswith(lower) for m in MONTHS):
        return "abbr_dot" if dot else "abbr"
    return None


def detect_date(text: str) -> DateStyle | None:
    parsed = parse_date(text)
    return parsed[0] if parsed else None


def _full_year(year: str) -> int:
    return int(year) if len(year) == 4 else 1900 + int(year)


def parse_date(text: str) -> tuple[DateStyle, int, int, int] | None:
    """(style, year, month, day). Two-digit years are read as 19xx; an ambiguous numeric date as month first."""
    t = " ".join(text.split())
    if m := _NUMERIC_DATE.fullmatch(t):
        first, _, second, year = m.groups()
        pad = (len(first) == 2 and first[0] == "0") or (len(second) == 2 and second[0] == "0")
        day_first = int(first) > 12
        month, day = (int(second), int(first)) if day_first else (int(first), int(second))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return NumericDate(sep=m[2], year_digits=len(year), pad=pad, day_first=day_first), _full_year(year), month, day
    for pattern, day_first in ((_MONTH_FIRST, False), (_DAY_FIRST, True)):
        if (m := pattern.fullmatch(t)) and (form := _month_form(m["month"], bool(m["dot"]))):
            token = m["month"].lower()
            month = next(i for i, name in enumerate(MONTHS, start=1) if name.lower().startswith(token))
            style = TextDate(month=form, day_first=day_first, comma=bool(m["comma"]), year_digits=len(m["year"]))
            return style, _full_year(m["year"]), month, int(m["day"])
    return None


def random_date_style() -> DateStyle:
    if random.randint(0, 2) == 0:
        return NumericDate(sep=random.choice("-/"), year_digits=4, pad=False, day_first=False)
    if random.randint(0, 3) == 0:
        month = "abbr_dot" if random.randint(0, 2) > 0 else "abbr"
    else:
        month = "full"
    day_first = random.randint(0, 5) == 0
    return TextDate(month=month, day_first=day_first, comma=not day_first, year_digits=4)


def format_date(style: DateStyle, year: int, month: int, day: int) -> str:
    y = f"{year % 100:02d}" if style.year_digits == 2 else str(year)
    if isinstance(style, NumericDate):
        fmt = "{:02d}" if style.pad else "{}"
        parts = [fmt.format(day), fmt.format(month)] if style.day_first else [fmt.format(month), fmt.format(day)]
        return style.sep.join([*parts, y])
    if style.month == "full":
        name = MONTHS[month - 1]
    else:
        name = MONTH_ABBREVIATIONS[month - 1]
        if style.month == "abbr_dot" and name != "May":
            name += "."
    comma = "," if style.comma else ""
    if style.day_first:
        return f"{day} {name}{comma} {y}"
    return f"{name} {day}{comma} {y}"


# --- places ---------------------------------------------------------------

@dataclass(frozen=True)
class PlaceStyle:
    # "postal" (TX), "postal_dotted" (T.X.), "title_dot" (Tx.),
    # "abbrev" (Tex.), "full" (Texas) or "none" (city only)
    state: str


def classify_state(state: str) -> str | None:
    """How a US state is written: a PlaceStyle.state form, or None if it isn't a recognizable state."""
    state = state.strip()
    compact = state.replace(" ", "")
    if compact.lower() in (a.lower() for a in state_abbreviations()):
        return "abbrev"
    if re.fullmatch(r"[A-Z]{2}", state):
        return "postal"
    if re.fullmatch(r"[A-Z]\.[A-Z]\.", compact):
        return "postal_dotted"
    if re.fullmatch(r"[A-Z][a-z]\.", state):
        return "title_dot"
    if state.lower() in (s.lower() for s in US_STATES.values()):
        return "full"
    return None


_EXTRA_ABBREVIATIONS = {"col.": "CO", "kan.": "KS", "neb.": "NE", "ore.": "OR", "wisc.": "WI", "tenn.": "TN"}


def state_code(state: str) -> str | None:
    """The postal code for a state written in any recognized form ("TX", "T.X.", "Tx.", "Tex.", "Texas")."""
    s = state.strip()
    compact = s.replace(" ", "").lower()
    if re.fullmatch(r"[A-Za-z]{2}", s) and s.upper() in US_STATES:
        return s.upper()
    if re.fullmatch(r"[a-z]\.[a-z]\.", compact) and compact.replace(".", "").upper() in US_STATES:
        return compact.replace(".", "").upper()
    for code, abbreviation in TRADITIONAL_ABBREVIATIONS.items():
        if compact == abbreviation.replace(" ", "").lower():
            return code
    if compact in _EXTRA_ABBREVIATIONS:
        return _EXTRA_ABBREVIATIONS[compact]
    if re.fullmatch(r"[a-z]{2}\.", compact) and compact[:2].upper() in US_STATES:
        return compact[:2].upper()
    for code, name in US_STATES.items():
        if s.lower() == name.lower():
            return code
    return None


def format_state(code: str, form: str) -> str:
    """Write a state's postal code in a PlaceStyle.state form."""
    if form == "full":
        return US_STATES[code]
    if form == "abbrev":
        return TRADITIONAL_ABBREVIATIONS.get(code, code.title() + ".")
    if form == "title_dot":
        return code.title() + "."
    if form == "postal_dotted":
        return f"{code[0]}.{code[1]}."
    return code


def detect_place(text: str) -> PlaceStyle | None:
    t = " ".join(text.split())
    if "," not in t:
        return PlaceStyle("none") if re.fullmatch(r"[A-Za-z .'-]+", t) else None
    form = classify_state(t.rsplit(",", 1)[1])
    return PlaceStyle(form) if form else None


def detect_state(text: str) -> PlaceStyle | None:
    """A box holding just a state ("TX", "Tex.", "Texas")."""
    form = classify_state(" ".join(text.split()))
    return PlaceStyle(form) if form else None


def random_place_style() -> PlaceStyle:
    kind = random.randint(0, 2)
    if kind == 0:
        r = random.randint(0, 3)
        return PlaceStyle("title_dot" if r == 0 else "postal_dotted" if r == 1 else "postal")
    return PlaceStyle("abbrev" if kind == 1 else "full")


# --- digit layouts (SSN, phone, age) --------------------------------------

@dataclass(frozen=True)
class DigitTemplate:
    template: str  # original text; each digit is replaced, everything else kept


def detect_digits(text: str, allowed_counts: set[int]) -> DigitTemplate | None:
    t = " ".join(text.split())
    if len(re.findall(r"\d", t)) in allowed_counts:
        return DigitTemplate(t)
    return None


# --- marital status / addresses -------------------------------------------

@dataclass(frozen=True)
class MaritalStyle:
    short: bool  # a single letter such as "M" or "S."
    dot: bool


# "m. 51;" (married '51), "c. 2." (2 children), or several codes in one box: "m. 51; c. 2."
_MARITAL_CODED = re.compile(r"(?:[A-Za-z]\.?\s*\d[\d,\s]*[.;,]?\s*)+")

STATUS_WORDS = ["married", "single", "divorced", "widowed", "separated", "unmarried", "engaged"]
NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_CHILD_NOUNS = r"children|child|kids|kid|daughters|daughter|sons|son"
_NUMBER = r"\d{1,2}|" + "|".join(NUMBER_WORDS) + r"|none"
_CHILDREN_COUNT_FIRST = re.compile(rf"(?P<num>{_NUMBER})\s*(?P<noun>{_CHILD_NOUNS})", re.I)  # "2 children", "3children"
_CHILDREN_LABEL_FIRST = re.compile(rf"(?P<noun>{_CHILD_NOUNS})\s*(?P<colon>:)?\s*(?P<num>{_NUMBER})", re.I)  # "Children: 2"
_STATUS = re.compile(rf"(?P<status>{'|'.join(STATUS_WORDS)})", re.I)
_COMBINED_SEPARATOR = re.compile(r"\s*(,|;|-|with)\s*", re.I)


@dataclass(frozen=True)
class ChildrenStyle:
    """How a number of children is written: "2 children", "Three children", "Children: None"."""

    number: str  # "digit" or "word"
    noun: str  # "children", "daughters", "sons" or "kids" (singular chosen by count)
    label_first: bool  # "Children: 2" rather than "2 children"
    colon: bool = False
    capitalized: bool = False  # the first letter was a capital ("Three children")


@dataclass(frozen=True)
class StatusStyle:
    """Just a status word ("Married", "single")."""

    capitalized: bool = True


@dataclass(frozen=True)
class CombinedStyle:
    """Status followed by children: "Married, 2 children", "Married - Three Children"."""

    status: StatusStyle
    separator: str  # the text between the two parts, e.g. ", " or " - " or " with "
    children: ChildrenStyle


def _children_style(t: str) -> ChildrenStyle | None:
    for pattern, label_first in ((_CHILDREN_COUNT_FIRST, False), (_CHILDREN_LABEL_FIRST, True)):
        if m := pattern.fullmatch(t):
            noun = m["noun"].lower()
            plural = {"child": "children", "kid": "kids", "daughter": "daughters", "son": "sons"}.get(noun, noun)
            return ChildrenStyle(
                number="digit" if m["num"].isdigit() else "word", noun=plural, label_first=label_first,
                colon=bool(label_first and m["colon"]), capitalized=t[:1].isupper(),
            )
    return None


def detect_marital(text: str) -> MaritalStyle | DigitTemplate | StatusStyle | ChildrenStyle | CombinedStyle | None:
    t = " ".join(text.split())
    if _MARITAL_CODED.fullmatch(t):
        return DigitTemplate(t)
    if re.fullmatch(r"[A-Za-z]\.?", t):
        return MaritalStyle(short=True, dot=t.endswith("."))
    if _STATUS.fullmatch(t):
        return StatusStyle(capitalized=t[:1].isupper())
    if children := _children_style(t):
        return children
    if (m := _STATUS.match(t)) and (sep := _COMBINED_SEPARATOR.match(t, m.end())):
        if children := _children_style(t[sep.end():]):
            separator = t[m.end():sep.end()]
            return CombinedStyle(StatusStyle(capitalized=t[:1].isupper()), separator, children)
    if re.fullmatch(r"[A-Za-z].*", t) and t[-1] not in ",;:.":  # trailing punctuation is peeled off by detect()
        return MaritalStyle(short=False, dot=False)
    return None


@dataclass(frozen=True)
class AddressStyle:
    part: str  # "full", "street" (12 Maple Avenue) or "city" (Springfield, PA 19064)
    has_zip: bool
    state: str | None = None  # a PlaceStyle.state form, when recognized


_ZIP = re.compile(r"\s*\b\d{5}(-\d{4})?$")


def detect_address(text: str) -> AddressStyle | None:
    lines = [" ".join(line.split()) for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    flat = ", ".join(lines)
    has_zip = bool(_ZIP.search(flat))
    core = _ZIP.sub("", flat).rstrip(", ")
    if len(lines) == 1 and "," not in core and not has_zip:
        part = "street"
    elif len(lines) == 1 and not core[:1].isdigit() and core.count(",") == 1:
        part = "city"
    else:
        part = "full"
    place = detect_place(", ".join(core.split(",")[-2:])) if "," in core else None
    state = place.state if place and place.state != "none" else None
    return AddressStyle(part=part, has_zip=has_zip, state=state)


@dataclass(frozen=True)
class ParsedAddress:
    """The parts of an original address, used only to link repeats of it on a page."""

    street: str | None = None
    city: str | None = None
    state: str | None = None  # postal code
    zip: str | None = None  # digits only


_PO_BOX = re.compile(r"\bp\.?\s*o\.?\s*box\b", re.I)
_STREET_WORDS = re.compile(
    r"\b(st|street|ave|avenue|rd|road|dr|drive|ln|lane|blvd|boulevard|way|pl|place|ct|court|rt|rte|route|"
    r"box|apt|suite|ste|hwy|highway|pkwy|parkway|terrace|ter|circle|cir)\b\.?", re.I,
)


def _split_city_state(segment: str) -> tuple[str | None, str | None]:
    """"Springfield PA" / "Springfield" -> (city, state code)."""
    words = segment.split()
    for n in (2, 1):  # a two-word state name ("New Jersey") or a one-token form ("PA", "Pa.")
        if len(words) >= n:
            code = state_code(" ".join(words[-n:]))
            if code:
                return (" ".join(words[:-n]) or None), code
    return (segment or None), None


def parse_address(text: str) -> ParsedAddress | None:
    """Street, city, state and zip of an address written on one or more lines (heuristic)."""
    lines = [" ".join(line.split()).strip(" ,") for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    flat = ", ".join(lines)
    zip_match = re.search(r"\b(\d{5})(?:-(\d{4}))?\s*$", flat)
    zip_code = zip_match.group(0).replace("-", "").strip() if zip_match else None
    core = flat[:zip_match.start()].strip(" ,") if zip_match else flat
    segments = [s.strip() for s in core.split(",") if s.strip()]
    street_segments, rest = [], []
    for n, s in enumerate(segments):
        # Street-like segments can follow other text (e.g. a name line at the top of the box),
        # but the final segment is the city/state whenever there is more than one.
        is_last = n == len(segments) - 1 and len(segments) > 1
        # Any digit counts: OCR often misreads the words ("Box", "Rt.") but numbers survive, and city
        # names rarely contain digits.
        looks_like_street = re.search(r"\d", s) or _PO_BOX.search(s) or _STREET_WORDS.search(s)
        (street_segments if looks_like_street and not is_last else rest).append(s)
    city = state = None
    if rest:
        if len(rest) >= 2 and state_code(rest[-1]):
            city, state = rest[-2], state_code(rest[-1])
        else:
            city, state = _split_city_state(rest[-1])
            if state is None and len(rest) >= 2 and len(rest[-1].replace(".", "").strip()) <= 3:
                city = rest[-2]  # a short unreadable last segment is a misread state, not the city
    street = ", ".join(street_segments) or None
    if not (street or city or state or zip_code):
        return None
    return ParsedAddress(street=street, city=city, state=state, zip=zip_code)


# --- personal names -------------------------------------------------------

TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "rev"}
FEMALE_TITLES = {"mrs", "ms", "miss"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "md", "phd"}


@dataclass(frozen=True)
class NameToken:
    kind: str  # "title", "first", "middle", "last", "initial", "suffix" or "literal"
    text: str = ""  # kept verbatim for titles and literals
    dot: bool = False  # initials and suffixes written with a period
    hyphenated: bool = False  # "Smith-Jones"
    upper: bool = False  # this word written in capitals ("HARTLEY, Jonas")
    of: str = ""  # for initials: "first" or "middle"


def title_gender(title: str) -> str | None:
    bare = title.rstrip(".").lower()
    return "female" if bare in FEMALE_TITLES else "male" if bare == "mr" else None


@dataclass(frozen=True)
class NameStyle:
    tokens: tuple[NameToken, ...]
    last_first: bool  # "Last, First ..." rather than "First ... Last"
    title_outside: str | None = None  # a title printed just left of the box ("Mr." before the field)
    identity: object | None = None  # a people.Person to fill this shape with, so repeated names match

    @property
    def gender(self) -> str | None:
        titles = [t.text for t in self.tokens if t.kind == "title"]
        if titles:
            return title_gender(titles[0])
        return title_gender(self.title_outside) if self.title_outside else None


@dataclass(frozen=True)
class ParsedName:
    """The parts of an original name, used only to decide which boxes are the same person."""

    style: NameStyle
    first: str | None
    first_letter: str | None  # from the first name or its initial
    middle_letter: str | None
    last: str | None


def detect_name(text: str) -> NameStyle | None:
    """The shape of a personal name: order, initials, title, suffix. None if it doesn't look like a name."""
    parsed = parse_name(text)
    return parsed.style if parsed else None


def parse_name(text: str) -> ParsedName | None:
    t = " ".join(text.split())
    if not t or not re.search(r"[A-Za-z]", t) or re.search(r"\d", t):
        return None
    last_first = bool(re.match(r"^[A-Za-z][\w'.-]*,\s", t))
    raw = [tok for tok in re.split(r"[\s,]+", t) if tok]

    kinds: list[NameToken] = []
    words: list[int] = []  # positions of full name words, assigned first/middle/last below
    for i, tok in enumerate(raw):
        bare = tok.rstrip(".").lower()
        if bare in TITLES and i == 0:
            kinds.append(NameToken("title", text=tok))
        elif bare in SUFFIXES and i == len(raw) - 1 and i > 0:
            kinds.append(NameToken("suffix", dot=tok.endswith(".")))
        elif re.fullmatch(r"[A-Za-z]\.?", tok):
            kinds.append(NameToken("initial", dot=tok.endswith(".")))
        elif re.fullmatch(r"[A-Za-z][A-Za-z'-]*", tok):
            letters = re.sub(r"[^A-Za-z]", "", tok)
            kinds.append(NameToken("word", hyphenated="-" in tok.strip("-"), upper=len(letters) > 1 and letters.isupper()))
            words.append(i)
        else:
            kinds.append(NameToken("literal", text=tok))
    if not words and not any(k.kind == "initial" for k in kinds):
        return None

    has_title = any(k.kind == "title" for k in kinds)
    roles: dict[int, str] = {}
    if last_first and words:
        roles[words[0]] = "last"
        for n, pos in enumerate(words[1:]):
            roles[pos] = "first" if n == 0 else "middle"
    elif len(words) == 1:
        roles[words[0]] = "last" if has_title or any(k.kind == "initial" for k in kinds) else "first"
    elif words:
        roles[words[0]] = "first"
        roles[words[-1]] = "last"
        for pos in words[1:-1]:
            roles[pos] = "middle"
    tokens: list[NameToken] = []
    first = first_letter = middle_letter = last = None
    for i, k in enumerate(kinds):
        if k.kind == "word":
            role = roles[i]
            tokens.append(NameToken(role, hyphenated=k.hyphenated, upper=k.upper))
            word = raw[i]
            if role == "first":
                first, first_letter = word, word[0]
            elif role == "middle":
                middle_letter = middle_letter or word[0]
            elif role == "last":
                last = word
        elif k.kind == "initial":
            # An initial names the first name until one is seen, then the middle name.
            of = "middle" if first_letter else "first"
            tokens.append(dataclasses.replace(k, of=of))
            if of == "first":
                first_letter = raw[i][0]
            elif middle_letter is None:
                middle_letter = raw[i][0]
        else:
            tokens.append(k)
    style = NameStyle(tokens=tuple(tokens), last_first=last_first)
    return ParsedName(style=style, first=first, first_letter=first_letter, middle_letter=middle_letter, last=last)


# --- per-category registry ------------------------------------------------

@dataclass(frozen=True)
class Style:
    case: str | None = None
    detail: object | None = None
    suffix: str = ""  # trailing punctuation to re-append, e.g. ";" in "February 3, 1941;"
    identity: object | None = None  # shared components for repeated values on a page (addresses.FakeAddress)


DETECTORS: dict[str, Callable[[str], object | None]] = {  # keyed by canonical category
    "birth_date": detect_date,
    "birth_place": detect_place,
    "ssn": lambda t: detect_digits(t, {9}),
    "home_phone": lambda t: detect_digits(t, {7, 10, 11}),
    "age": lambda t: detect_digits(t, {1, 2, 3}),
    "marital_status": detect_marital,
    "home_address": detect_address,
    "state": detect_state,
    "zip": lambda t: detect_digits(t, {5, 9}),
    "name": detect_name,
}


def detector_for(category: str) -> Callable[[str], object | None] | None:
    return DETECTORS.get(canonical(category))


def _punctuation_variants(text: str) -> list[tuple[str, str]]:
    """(core, suffix) splits to try, least stripped first; "." is meaningful in "N.Y." so it goes last."""
    variants = [(text, "")]
    core = text.rstrip(",;:")
    suffix = text[len(core):]
    if suffix:
        variants.append((core, suffix))
    if core.endswith("."):
        variants.append((core[:-1], "." + suffix))
    return variants


def detect(category: str, original: str) -> Style:
    # Normalize spacing within lines but keep line breaks; address detection needs them.
    text = "\n".join(" ".join(line.split()) for line in original.strip().splitlines() if line.strip())
    case = detect_case(text)
    variants = _punctuation_variants(text)
    detector = detector_for(category)
    if detector:
        for core, suffix in variants:
            if (detail := detector(core)) is not None:
                return Style(case=case, detail=detail, suffix=suffix)
    core = text.rstrip(",;:")
    return Style(case=case, suffix=text[len(core):])
