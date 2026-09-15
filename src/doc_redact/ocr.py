"""Helpers for AWS Textract OCR output (the JSON stored next to each image)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from doc_redact.annotations import BBox


@dataclass(frozen=True)
class Word:
    text: str
    bbox: BBox
    handwritten: bool = False  # Textract TextType == "HANDWRITING"


def words(ocr: dict[str, Any], image_size: tuple[int, int]) -> list[Word]:
    """WORD blocks with their normalized geometry scaled to pixels."""
    (w, h) = image_size
    out = []
    for block in ocr["Blocks"]:
        if block["BlockType"] != "WORD":
            continue
        g = block["Geometry"]["BoundingBox"]
        out.append(
            Word(
                text=block["Text"],
                bbox=BBox(left=g["Left"] * w, top=g["Top"] * h, width=g["Width"] * w, height=g["Height"] * h),
                handwritten=block.get("TextType") == "HANDWRITING",
            )
        )
    return out


def mostly_handwritten(words: list[Word]) -> bool:
    return bool(words) and sum(w.handwritten for w in words) * 2 > len(words)
