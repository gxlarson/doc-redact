"""Per-dataset conventions: where images and OCR live, and how outputs are named.

Every dataset's annotation file lists a ``url``; these rules derive the actual
image URL, the Textract JSON URL, and the output filename from it. A "URL" may
also be a local path, which is read from disk instead of downloaded.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from doc_redact.annotations import Document


def _swap_ext(url: str, ext: str) -> str:
    return posixpath.splitext(url)[0] + ext


def _basename(url: str) -> str:
    return posixpath.basename(url.replace("\\", "/"))


def _with_ext(url: str, ext: str) -> str:
    """``url`` with ``ext`` appended when it carries no extension of its own.

    Some RVL-CDIP annotations lost the ".jpg" from their url; the file on the server still has it,
    so the bare url 404s and the page would be dropped from the run.
    """
    return url if posixpath.splitext(url)[1] else url + ext


@dataclass(frozen=True)
class Dataset:
    name: str
    image_url: Callable[[Document], str]
    ocr_url: Callable[[Document], str]
    output_name: Callable[[Document], str]
    font: str
    font_size: int


def in_folders(dataset: Dataset, images: Path | None = None, ocr: Path | None = None) -> Dataset:
    """``dataset`` with its images and/or OCR read from local folders instead of where its URLs point.

    The file names are the ones the dataset's own rules give, so an annotation file written against
    a server works unchanged on a downloaded copy of it.
    """
    image_url, ocr_url = dataset.image_url, dataset.ocr_url
    if images is not None:
        image_url = lambda doc: str(images / _basename(dataset.image_url(doc)))  # noqa: E731
    if ocr is not None:
        ocr_url = lambda doc: str(ocr / _basename(dataset.ocr_url(doc)))  # noqa: E731
    return replace(dataset, image_url=image_url, ocr_url=ocr_url)


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        # Any other corpus: the image is at `url` (a URL or a local path) and its OCR beside it,
        # under the same name with ".json".
        Dataset(
            name="generic",
            image_url=lambda doc: doc.url,
            ocr_url=lambda doc: _swap_ext(doc.url, ".json"),
            output_name=lambda doc: doc.filename,
            font="LiberationSerif-Regular.ttf",
            font_size=24,
        ),
        # Annotations point at FUNSD/<id>.jpg, but the originals are PNGs, and
        # OCR lives in a sibling FUNSD_ocr/ folder.
        Dataset(
            name="funsd",
            image_url=lambda doc: _swap_ext(doc.url, ".png"),
            ocr_url=lambda doc: _swap_ext(doc.url.replace("/FUNSD/", "/FUNSD_ocr/"), ".json"),
            output_name=lambda doc: doc.stem + ".png",
            font="Inconsolata-Variable.ttf",
            font_size=14,
        ),
        Dataset(
            name="tobacco3482",
            image_url=lambda doc: doc.url,
            ocr_url=lambda doc: _swap_ext(doc.url, ".json"),
            output_name=lambda doc: doc.filename,
            font="LiberationSerif-Regular.ttf",
            font_size=34,
        ),
        Dataset(
            name="tobacco800",
            image_url=lambda doc: doc.url,
            ocr_url=lambda doc: _swap_ext(doc.url, ".json"),
            output_name=lambda doc: doc.filename,
            font="CourierPrime-Regular.ttf",
            font_size=21,
        ),
        Dataset(
            name="rvlcdip",
            image_url=lambda doc: _with_ext(doc.url, ".jpg"),
            ocr_url=lambda doc: _swap_ext(doc.url.replace("/rvlcdip/", "/rvlcdip_ocr/"), ".json"),
            output_name=lambda doc: _with_ext(doc.filename, ".jpg"),
            font="Inconsolata-Variable.ttf",
            font_size=13,
        ),
    ]
}
