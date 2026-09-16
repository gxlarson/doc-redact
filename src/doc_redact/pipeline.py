"""Redact or audit whole documents and batches of documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from doc_redact import addresses, baseline, checks, geometry, ocr, people, render, typeface, values
from doc_redact.annotations import Document, merge_documents
from doc_redact.datasets import Dataset
from doc_redact import fakedata
from doc_redact.categories import canonical
from doc_redact.fakedata import has_generator
from doc_redact.remote import Fetcher

log = logging.getLogger(__name__)


class Mode(str, Enum):
    WHITE = "white"
    PSEUDO = "pseudo"


@dataclass
class Options:
    mode: Mode
    text: render.TextSettings


def redact_document(image: Image.Image, doc: Document, page_ocr: dict | None, options: Options) -> Image.Image:
    doc = doc.in_pixels(image.size)
    if options.mode is Mode.WHITE:
        for annotation in doc.annotations:
            image = render.white_out(image, annotation.bbox)
        return image

    # Measure weight and typeface on the untouched page so earlier replacements can't skew later boxes.
    gray = np.asarray(image.convert("L"))
    words = ocr.words(page_ocr, image.size)
    print_matcher = handwriting_matcher = None
    if options.text.match_font:
        print_matcher = typeface.TypefaceMatcher(gray, typeface.candidates(options.text.system_fonts))
        if options.text.handwriting and any(w.handwritten for w in words):
            handwriting_matcher = typeface.TypefaceMatcher(gray, typeface.candidates(handwriting=True))
    page = render.Page(
        words=words, gray=gray, typefaces=print_matcher, handwriting_typefaces=handwriting_matcher,
        skew=baseline.page_skew(gray, words), rgb=np.asarray(image.convert("RGB")),
    )
    page.identities = people.link(doc.annotations, words)
    page.addresses = addresses.link(doc.annotations, words)
    if page.addresses:
        streets = {a.street for a in page.addresses.values() if a.street}
        log.debug("%s: %d address box(es) share %d fake street(s)", doc.stem, len(page.addresses), len(streets))
    if any(canonical(a.category) == "marital_status" for a in doc.annotations):
        page.family = fakedata.fake_family()
    page.shared_values = values.link(doc.annotations, words)
    if page.shared_values:
        log.debug("%s: %d box(es) share %d repeated value(s)", doc.stem, len(page.shared_values),
                  len({id(v) for v in page.shared_values.values()}))
    if page.identities:
        log.debug("%s: %d name box(es) linked to %d person(s)", doc.stem, len(page.identities),
                  len({id(p) for p in page.identities.values()}))
    if handwriting_matcher is not None:
        page.handwriting_face = render.document_handwriting_face(words, handwriting_matcher)
        if page.handwriting_face is not None:
            log.debug("%s: handwriting typeface for the document: %s", doc.stem, page.handwriting_face.name)
    # Boxes that get replacement text. Near-duplicates (the same field annotated twice) are drawn once.
    writable = []
    for annotation in doc.annotations:
        if has_generator(annotation.category) or annotation.category in options.text.overrides:
            writable.append(annotation)
        else:
            log.warning("%s: no generator for %r; region left blank", doc.stem, annotation.category)
    duplicates = geometry.near_duplicates([a.bbox for a in writable], [canonical(a.category) for a in writable])
    writable = [a for i, a in enumerate(writable) if i not in duplicates]
    if duplicates:
        log.debug("%s: %d near-duplicate box(es) erased but not drawn twice", doc.stem, len(duplicates))
    regions = geometry.exclusive_regions([a.bbox for a in writable])
    squeezed = sum(r.is_cut for r in regions)
    if squeezed:
        log.debug("%s: %d box(es) share overlaps with neighbors", doc.stem, squeezed)

    # Plan everything on the untouched page, erase every box, then draw: an erase can never cut into a
    # replacement already drawn for an overlapping box, and each replacement stays in its own region.
    plans = [render.plan_text(a, page, options.text, region) for a, region in zip(writable, regions)]
    for annotation in doc.annotations:
        image = render.erase(image, geometry.erase_extent(annotation.bbox, words), page, options.text)
    for plan in plans:
        image = render.draw_plan(image, plan)
    return image


SUFFIX_FOR_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "GIF": ".gif", "BMP": ".bmp"}


def output_paths(out_dir: Path, dataset: Dataset, doc: Document, image_format: str,
                 source_format: str | None = None) -> list[Path]:
    """Where a redacted page is written. "same" keeps the source's own extension; "both" writes the
    page twice from one render, as a lossless PNG and a compact JPEG of identical pixels.

    Some URLs carry no extension at all. For those, "same" falls back to ``source_format`` (what the
    server actually sent), because a path with no suffix gives the encoder nothing to go on.
    """
    name = dataset.output_name(doc)
    stem = Path(name).stem
    if image_format == "png":
        return [out_dir / f"{stem}.png"]
    if image_format == "jpeg":
        return [out_dir / f"{stem}.jpg"]
    if image_format == "both":
        return [out_dir / "png" / f"{stem}.png", out_dir / "jpg" / f"{stem}.jpg"]
    if not Path(name).suffix:
        name = stem + SUFFIX_FOR_FORMAT.get(source_format or "", ".png")
    return [out_dir / name]


def save_page(image: Image.Image, path: Path, source_mode: str, jpeg_quality: int) -> None:
    """Save a redacted page, keeping a grayscale scan grayscale.

    Pages are edited in RGB, but writing a single-channel scan back as three channels triples the
    pixel data and lets JPEG chroma subsampling touch pixels we never painted. Bilevel sources come
    back as grayscale, not 1-bit: the replacement text is antialiased, and rethresholding it would
    look nothing like the scan around it.
    """
    if source_mode in ("L", "1") and image.mode != "L":
        image = image.convert("L")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        # subsampling=0 keeps full chroma resolution on the pages that really are colour.
        image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        image.save(path)


def run(
    docs: list[Document],
    dataset: Dataset,
    fetcher: Fetcher,
    out_dir: Path,
    options: Options,
    keep_unannotated: bool = False,
    image_format: str = "same",
    jpeg_quality: int = 95,
) -> list[tuple[Document, Exception]]:
    """Redact each document into ``out_dir``. Returns the documents that failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Records can differ in URL yet land on one output file (a URL listed with and without its
    # extension); merge on the output name too, or the last one written would win.
    docs = merge_documents(docs, key=dataset.output_name)
    failures = []
    for i, doc in enumerate(docs, 1):
        if not doc.annotations and not keep_unannotated:
            log.debug("skipping %s: no annotations", doc.stem)
            continue
        image_url = dataset.image_url(doc)
        log.info("[%d/%d] %s", i, len(docs), image_url)
        try:
            source_mode, source_format = fetcher.image_kind(image_url)
            image = fetcher.image(image_url)
            page_ocr = fetcher.json(dataset.ocr_url(doc)) if options.mode is Mode.PSEUDO else None
            redacted = redact_document(image, doc, page_ocr, options)
            for path in output_paths(out_dir, dataset, doc, image_format, source_format):
                save_page(redacted, path, source_mode, jpeg_quality)
        except Exception as exc:  # keep going; one bad page shouldn't sink a batch
            log.error("failed on %s: %r", doc.stem, exc, exc_info=log.isEnabledFor(logging.DEBUG))
            failures.append((doc, exc))
    return failures


def check(docs: list[Document], dataset: Dataset, fetcher: Fetcher) -> Iterator[tuple[Document, list[checks.BoxReport] | Exception]]:
    """Yield each annotated document with its box reports, or the error that prevented checking it."""
    for doc in docs:
        if not doc.annotations:
            continue
        try:
            size = fetcher.image_size(dataset.image_url(doc))
            page_words = ocr.words(fetcher.json(dataset.ocr_url(doc)), size)
            yield doc, checks.check_document(doc.in_pixels(size).annotations, page_words)
        except Exception as exc:
            yield doc, exc
