"""Saving redacted pages: the source's colour mode and extension are respected. Values are invented."""
import dataclasses

import numpy as np
import pytest
from PIL import Image

from doc_redact import pipeline
from doc_redact.annotations import Annotation, BBox, Document
from doc_redact.datasets import DATASETS

DOC = Document(url="https://host/rvlcdip/train_letter_abc.jpg", annotations=[])
DATASET = DATASETS["rvlcdip"]


NO_EXTENSION = Document(url="https://host/rvlcdip/test_resume_abc", annotations=[])


def _names(tmp_path, image_format, doc=DOC, source_format=None):
    return [str(p.relative_to(tmp_path)).replace("\\", "/")
            for p in pipeline.output_paths(tmp_path, DATASET, doc, image_format, source_format)]


def test_output_path_follows_the_requested_format(tmp_path):
    assert _names(tmp_path, "same") == ["train_letter_abc.jpg"]
    assert _names(tmp_path, "png") == ["train_letter_abc.png"]
    assert _names(tmp_path, "jpeg") == ["train_letter_abc.jpg"]
    assert _names(tmp_path, "both") == ["png/train_letter_abc.png", "jpg/train_letter_abc.jpg"]


def test_rvlcdip_urls_missing_their_extension_get_it_back(tmp_path):
    """100 RVL-CDIP annotations lost the ".jpg"; the file on the server still has it."""
    assert DATASET.image_url(NO_EXTENSION).endswith("/test_resume_abc.jpg")
    assert DATASET.ocr_url(NO_EXTENSION).endswith("/rvlcdip_ocr/test_resume_abc.json")
    assert _names(tmp_path, "same", NO_EXTENSION, "JPEG") == ["test_resume_abc.jpg"]


def test_a_bare_output_name_takes_the_format_the_server_sent(tmp_path):
    """Datasets whose output name has no suffix: a suffixless path gives the encoder nothing to go on."""
    bare = dataclasses.replace(DATASET, output_name=lambda doc: "page")
    assert pipeline.output_paths(tmp_path, bare, DOC, "same", "JPEG")[0].name == "page.jpg"
    assert pipeline.output_paths(tmp_path, bare, DOC, "same", "PNG")[0].name == "page.png"
    assert pipeline.output_paths(tmp_path, bare, DOC, "same", None)[0].name == "page.png"
    assert pipeline.output_paths(tmp_path, bare, DOC, "png")[0].name == "page.png"  # explicit format needs no source


def test_a_grayscale_scan_is_saved_grayscale(tmp_path):
    page = Image.new("RGB", (40, 30), "white")  # pages are edited in RGB whatever the source was
    out = tmp_path / "page.jpg"
    pipeline.save_page(page, out, source_mode="L", jpeg_quality=95)
    with Image.open(out) as saved:
        assert saved.mode == "L"


def test_a_bilevel_scan_comes_back_as_grayscale_not_rethresholded(tmp_path):
    page = Image.new("RGB", (40, 30), "white")
    out = tmp_path / "page.png"
    pipeline.save_page(page, out, source_mode="1", jpeg_quality=95)
    with Image.open(out) as saved:
        assert saved.mode == "L"  # antialiased replacement text would not survive 1-bit


def test_a_colour_scan_stays_colour(tmp_path):
    page = Image.new("RGB", (40, 30), (200, 40, 40))
    out = tmp_path / "page.jpg"
    pipeline.save_page(page, out, source_mode="RGB", jpeg_quality=95)
    with Image.open(out) as saved:
        assert saved.mode == "RGB"
        assert saved.getpixel((5, 5))[0] > saved.getpixel((5, 5))[1]


def test_both_formats_hold_the_same_page(tmp_path):
    rng = np.random.default_rng(1)
    page = Image.fromarray(rng.integers(0, 255, (50, 40), dtype=np.uint8), mode="L").convert("RGB")
    png, jpg = pipeline.output_paths(tmp_path, DATASET, DOC, "both")
    for path in (png, jpg):
        pipeline.save_page(page, path, source_mode="L", jpeg_quality=95)
    assert png.exists() and jpg.exists()
    a, b = (np.asarray(Image.open(p).convert("L")).astype(int) for p in (png, jpg))
    assert a.shape == b.shape
    assert np.abs(a - b).mean() < 6  # the same render, one lossless and one JPEG-encoded


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_quality_applies_to_jpeg_only_and_png_is_lossless(tmp_path, suffix):
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (60, 60), dtype=np.uint8), mode="L").convert("RGB")
    low, high = tmp_path / f"low{suffix}", tmp_path / f"high{suffix}"
    pipeline.save_page(noisy, low, "L", jpeg_quality=30)
    pipeline.save_page(noisy, high, "L", jpeg_quality=95)
    if suffix == ".jpg":
        assert low.stat().st_size < high.stat().st_size
    else:
        assert low.stat().st_size == high.stat().st_size
        assert np.array_equal(np.asarray(Image.open(low)), np.asarray(noisy.convert("L")))


def test_a_page_listed_with_and_without_its_extension_is_redacted_once_with_every_box(tmp_path, monkeypatch):
    """Both records write test_resume_abc.jpg; redacted separately, the last write would win."""
    ssn = Annotation("ssn", BBox(left=10, top=10, width=50, height=12))
    name = Annotation("name", BBox(left=0, top=40, width=30, height=10))
    docs = [dataclasses.replace(NO_EXTENSION, annotations=[ssn]),
            Document(url=NO_EXTENSION.url + ".jpg", annotations=[ssn, name])]

    class Fetcher:
        def image_kind(self, url): return "L", "JPEG"
        def image(self, url): return Image.new("RGB", (100, 100), "white")
        def json(self, url): return {"Blocks": []}

    redacted = []
    monkeypatch.setattr(pipeline, "redact_document", lambda image, doc, ocr, options: redacted.append(doc) or image)
    monkeypatch.setattr(pipeline, "save_page", lambda *args: None)
    pipeline.run(docs, DATASET, Fetcher(), tmp_path, pipeline.Options(mode=pipeline.Mode.WHITE, text=None), image_format="both")

    assert len(redacted) == 1
    assert redacted[0].annotations == [ssn, name]
