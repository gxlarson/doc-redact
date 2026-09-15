"""Audit annotations against the OCR text underneath them."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rvlcdip_redact import geometry, style
from rvlcdip_redact.annotations import Annotation
from rvlcdip_redact.ocr import Word

DATE_CATEGORIES = {"birth_date", "date_of_birth"}
DIGIT_CATEGORIES = DATE_CATEGORIES | {"ssn", "home_phone", "age"}
TEXT_CATEGORIES = {"birth_place", "place_of_birth", "marital_status", "religion", "citizenship_nationality"}
_SSN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")


@dataclass
class BoxReport:
    index: int
    category: str
    text: str
    lines: int
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_annotation(index: int, annotation: Annotation, page_words: list[Word]) -> BoxReport:
    lines = geometry.group_lines(geometry.words_inside(annotation.bbox, page_words))
    text = geometry.reading_text(lines)
    category = annotation.category
    report = BoxReport(index=index, category=category, text=text, lines=len(lines))
    flags = report.flags

    if not lines:
        flags.append("no_ocr_words")
        return report
    flat = " ".join(text.split())
    if category not in DATE_CATEGORIES and category != "home_address" and style.detect_date(flat):
        flags.append("looks_like_date")
    if category != "ssn" and _SSN.search(flat):
        flags.append("looks_like_ssn")
    if category in DIGIT_CATEGORIES and not re.search(r"\d", flat):
        flags.append("no_digits")
    if category in TEXT_CATEGORIES and re.fullmatch(r"[\d\W]+", flat):
        flags.append("only_digits")
    if category in style.DETECTORS and style.detect(category, flat).detail is None:
        flags.append("style_not_detected")
    return report


def check_document(annotations: list[Annotation], page_words: list[Word]) -> list[BoxReport]:
    return [check_annotation(i, a, page_words) for i, a in enumerate(annotations)]


# Flags that suggest a labeling mistake, as opposed to informational ones.
PROBLEM_FLAGS = {"no_ocr_words", "looks_like_date", "looks_like_ssn", "no_digits", "only_digits"}
