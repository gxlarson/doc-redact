"""Saving redacted pages: the source's colour mode and extension are respected. Values are invented."""
import numpy as np
import pytest
from PIL import Image

from doc_redact import pipeline
from doc_redact.annotations import Document
from doc_redact.datasets import DATASETS

DOC = Document(url="https://host/rvlcdip/train_letter_abc.jpg", annotations=[])
DATASET = DATASETS["rvlcdip"]


def test_output_path_follows_the_requested_format(tmp_path):
    assert pipeline.output_path(tmp_path, DATASET, DOC, "same").name == "train_letter_abc.jpg"
    assert pipeline.output_path(tmp_path, DATASET, DOC, "png").name == "train_letter_abc.png"
    assert pipeline.output_path(tmp_path, DATASET, DOC, "jpeg").name == "train_letter_abc.jpg"


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
