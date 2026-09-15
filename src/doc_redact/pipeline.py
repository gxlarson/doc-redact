"""Redact or audit whole documents and batches of documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from doc_redact import addresses, baseline, checks, ocr, people, render, typeface, values
from doc_redact.annotations import Document
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
    for annotation in doc.annotations:
        if not has_generator(annotation.category) and annotation.category not in options.text.overrides:
            log.warning("%s: no generator for %r; region left blank", doc.stem, annotation.category)
            image = render.erase(image, annotation.bbox, page, options.text)
            continue
        image = render.pseudonymize(image, annotation, page, options.text)
    return image


def run(
    docs: list[Document],
    dataset: Dataset,
    fetcher: Fetcher,
    out_dir: Path,
    options: Options,
    keep_unannotated: bool = False,
) -> list[tuple[Document, Exception]]:
    """Redact each document into ``out_dir``. Returns the documents that failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for i, doc in enumerate(docs, 1):
        if not doc.annotations and not keep_unannotated:
            log.debug("skipping %s: no annotations", doc.stem)
            continue
        image_url = dataset.image_url(doc)
        log.info("[%d/%d] %s", i, len(docs), image_url)
        try:
            image = fetcher.image(image_url)
            page_ocr = fetcher.json(dataset.ocr_url(doc)) if options.mode is Mode.PSEUDO else None
            redacted = redact_document(image, doc, page_ocr, options)
            redacted.save(out_dir / dataset.output_name(doc))
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
