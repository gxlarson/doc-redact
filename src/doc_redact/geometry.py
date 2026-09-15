"""Relate an annotated region to the OCR words on the page."""

from __future__ import annotations

from statistics import median

from doc_redact.annotations import BBox
from doc_redact.ocr import Word


def words_inside(region: BBox, words: list[Word], min_coverage: float = 0.7) -> list[Word]:
    """Words whose area lies at least ``min_coverage`` inside ``region``."""
    return [
        w for w in words
        if w.bbox.area > 0 and region.intersection_area(w.bbox) / w.bbox.area > min_coverage
    ]


def words_in_row(region: BBox, words: list[Word]) -> list[Word]:
    """Words anywhere on the page whose vertical center falls within ``region``'s rows."""
    return [w for w in words if region.top <= w.bbox.top + w.bbox.height / 2 <= region.bottom]


def group_lines(words: list[Word]) -> list[list[Word]]:
    """Cluster words into text lines, top to bottom, each sorted left to right."""
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: w.bbox.top + w.bbox.height / 2):
        center = word.bbox.top + word.bbox.height / 2
        if lines:
            line = lines[-1]
            line_center = median(w.bbox.top + w.bbox.height / 2 for w in line)
            line_height = median(w.bbox.height for w in line)
            if abs(center - line_center) <= line_height / 2:
                line.append(word)
                continue
        lines.append([word])
    return [sorted(line, key=lambda w: w.bbox.left) for line in lines]


def reading_text(lines: list[list[Word]]) -> str:
    return "\n".join(" ".join(w.text for w in line) for line in lines)


def extent(words: list[Word]) -> BBox | None:
    if not words:
        return None
    left = min(w.bbox.left for w in words)
    top = min(w.bbox.top for w in words)
    right = max(w.bbox.right for w in words)
    bottom = max(w.bbox.bottom for w in words)
    return BBox(left=left, top=top, width=right - left, height=bottom - top)


def line_pitch(lines: list[list[Word]]) -> float | None:
    """Median distance between the tops of consecutive lines."""
    if len(lines) < 2:
        return None
    tops = [min(w.bbox.top for w in line) for line in lines]
    return median(b - a for a, b in zip(tops, tops[1:]))
