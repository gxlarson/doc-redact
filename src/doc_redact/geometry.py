"""Relate an annotated region to the OCR words on the page."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from doc_redact.annotations import BBox
from doc_redact.ocr import Word


def words_inside(region: BBox, words: list[Word], min_coverage: float = 0.7) -> list[Word]:
    """Words whose area lies at least ``min_coverage`` inside ``region``."""
    return [
        w for w in words
        if w.bbox.area > 0 and region.intersection_area(w.bbox) / w.bbox.area > min_coverage
    ]


def erase_extent(region: BBox, words: list[Word], max_grow: float | None = None) -> BBox:
    """``region`` widened sideways to cover the words it mostly contains, so an annotation drawn a little
    tight doesn't leave the ends of a word behind. Growth per side is capped at ``max_grow`` (default: the
    region's height); rows are never extended, which would reach into the lines above and below."""
    inside = words_inside(region, words)
    if not inside:
        return region
    cap = region.height if max_grow is None else max_grow
    left = max(region.left - cap, min(region.left, min(w.bbox.left for w in inside)))
    right = min(region.right + cap, max(region.right, max(w.bbox.right for w in inside)))
    return BBox(left, region.top, right - left, region.height)


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


@dataclass(frozen=True)
class Region:
    """The part of a box its replacement may use: the box, with any overlap with a neighbor split between them."""

    bbox: BBox
    cut_top: bool = False
    cut_bottom: bool = False
    cut_left: bool = False
    cut_right: bool = False

    @property
    def is_cut(self) -> bool:
        return self.cut_top or self.cut_bottom or self.cut_left or self.cut_right

    def limits(self, margin_x: float, margin_y: float) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) the replacement's ink may reach: exact on cut sides, with a margin elsewhere
        (so descenders and swashes on free sides aren't clipped)."""
        b = self.bbox
        return (b.left - (0 if self.cut_left else margin_x), b.top - (0 if self.cut_top else margin_y),
                b.right + (0 if self.cut_right else margin_x), b.bottom + (0 if self.cut_bottom else margin_y))


def near_duplicates(boxes: list[BBox], categories: list[str], min_iou: float = 0.5) -> set[int]:
    """Indices of boxes that repeat an earlier box of the same category over nearly the same area."""
    duplicates = set()
    for j, b in enumerate(boxes):
        for i in range(j):
            if i in duplicates or categories[i] != categories[j]:
                continue
            inter = boxes[i].intersection_area(b)
            union = boxes[i].area + b.area - inter
            if union > 0 and inter / union >= min_iou:
                duplicates.add(j)
                break
    return duplicates


def exclusive_regions(boxes: list[BBox]) -> list[Region]:
    """Split every pairwise overlap at its middle, along the axis where the boxes overlap least."""
    edges = [[b.left, b.top, b.right, b.bottom] for b in boxes]
    cuts = [[False, False, False, False] for _ in boxes]  # left, top, right, bottom
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            dx = min(a.right, b.right) - max(a.left, b.left)
            dy = min(a.bottom, b.bottom) - max(a.top, b.top)
            if dx <= 0 or dy <= 0:
                continue
            if dy / min(a.height, b.height) <= dx / min(a.width, b.width):  # stacked: split horizontally
                upper, lower = (i, j) if a.top + a.height / 2 <= b.top + b.height / 2 else (j, i)
                middle = (boxes[lower].top + boxes[upper].bottom) / 2
                edges[upper][3] = min(edges[upper][3], middle)
                edges[lower][1] = max(edges[lower][1], middle)
                cuts[upper][3] = cuts[lower][1] = True
            else:  # side by side: split vertically
                left, right = (i, j) if a.left + a.width / 2 <= b.left + b.width / 2 else (j, i)
                middle = (boxes[right].left + boxes[left].right) / 2
                edges[left][2] = min(edges[left][2], middle)
                edges[right][0] = max(edges[right][0], middle)
                cuts[left][2] = cuts[right][0] = True
    return [
        Region(BBox(left=l, top=t, width=max(1.0, r - l), height=max(1.0, btm - t)),
               cut_left=c[0], cut_top=c[1], cut_right=c[2], cut_bottom=c[3])
        for (l, t, r, btm), c in zip(edges, cuts)
    ]


def line_pitch(lines: list[list[Word]]) -> float | None:
    """Median distance between the tops of consecutive lines."""
    if len(lines) < 2:
        return None
    tops = [min(w.bbox.top for w in line) for line in lines]
    return median(b - a for a, b in zip(tops, tops[1:]))
