"""PII annotation records and the Label Studio export parser.

The on-disk "parsed" format (shared by every dataset) is a JSON list of:

    {"url": "...", "image": "...", "annotations": [
        {"category": "ssn", "bbox": {"left": 1, "top": 2, "width": 3, "height": 4}}
    ]}

Bounding boxes are in pixels, with the origin at the top-left of the image.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBox:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection_area(self, other: BBox) -> float:
        dx = min(self.right, other.right) - max(self.left, other.left)
        dy = min(self.bottom, other.bottom) - max(self.top, other.top)
        return dx * dy if dx > 0 and dy > 0 else 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BBox:
        return cls(left=d["left"], top=d["top"], width=d["width"], height=d["height"])

    def to_dict(self) -> dict[str, float]:
        return {"top": self.top, "left": self.left, "height": self.height, "width": self.width}


@dataclass(frozen=True)
class RelativeBBox:
    """A box in percent of the image size, as Label Studio stores it: {"x", "y", "width", "height"}."""

    x: float
    y: float
    width: float
    height: float

    def to_pixels(self, image_size: tuple[int, int]) -> BBox:
        w, h = image_size
        return BBox(
            left=int(round(self.x / 100.0 * w)),
            top=int(round(self.y / 100.0 * h)),
            width=int(round(self.width / 100.0 * w)),
            height=int(round(self.height / 100.0 * h)),
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def bbox_from_dict(d: dict[str, Any]) -> BBox | RelativeBBox:
    """Pixel boxes use left/top; unconverted Label Studio boxes use x/y percentages."""
    if "left" in d:
        return BBox.from_dict(d)
    if "x" in d:
        return RelativeBBox(x=d["x"], y=d["y"], width=d["width"], height=d["height"])
    raise ValueError(f"unrecognized bbox format: {sorted(d)}")


@dataclass(frozen=True)
class Annotation:
    category: str
    bbox: BBox | RelativeBBox


@dataclass
class Document:
    url: str
    annotations: list[Annotation] = field(default_factory=list)
    image: str | None = None  # Label Studio upload path, kept for provenance

    @property
    def filename(self) -> str:
        return posixpath.basename(self.url)

    @property
    def stem(self) -> str:
        return posixpath.splitext(self.filename)[0]

    def in_pixels(self, image_size: tuple[int, int]) -> Document:
        """This document with every percent-of-image box converted to pixels."""
        return Document(
            url=self.url,
            image=self.image,
            annotations=[
                Annotation(a.category, a.bbox.to_pixels(image_size) if isinstance(a.bbox, RelativeBBox) else a.bbox)
                for a in self.annotations
            ],
        )


def load_documents(path: str | Path) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Document(
            url=item["url"],
            image=item.get("image"),
            annotations=[
                Annotation(category=a["category"], bbox=bbox_from_dict(a["bbox"]))
                for a in item["annotations"]
            ],
        )
        for item in raw
    ]


def save_documents(docs: list[Document], path: str | Path) -> None:
    out = []
    for doc in docs:
        item: dict[str, Any] = {}
        if doc.image is not None:
            item["image"] = doc.image
        item["url"] = doc.url
        item["annotations"] = [
            {"bbox": a.bbox.to_dict(), "category": a.category} for a in doc.annotations
        ]
        out.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


def parse_labelstudio(export: list[dict[str, Any]], base_url: str) -> list[Document]:
    """Convert a Label Studio rectangle-label export into Documents.

    Label Studio stores boxes as percentages of the image size and prefixes
    uploaded filenames with a random id ("954fb992-<original name>"); the
    prefix is stripped and the remainder appended to ``base_url``.
    """
    if not base_url.endswith("/"):
        base_url += "/"
    docs = []
    for task in export:
        annotations = []
        for anno in task["annotations"]:
            for result in anno["result"]:
                v = result["value"]
                relative = RelativeBBox(x=v["x"], y=v["y"], width=v["width"], height=v["height"])
                size = (result["original_width"], result["original_height"])
                annotations.append(Annotation(category=v["rectanglelabels"][0], bbox=relative.to_pixels(size)))
        upload_path = task["data"]["image"]
        original_name = posixpath.basename(upload_path).split("-", 1)[1]
        docs.append(Document(url=base_url + original_name, image=upload_path, annotations=annotations))
    return docs
