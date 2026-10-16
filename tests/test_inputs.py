"""Where pages and OCR come from: local paths, local folders, URLs, or no OCR at all."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from doc_redact import cli
from doc_redact.annotations import Document
from doc_redact.datasets import DATASETS, in_folders
from doc_redact.remote import Fetcher, local_path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("url, expected", [
    ("https://host/a.jpg", None),
    ("s3://bucket/a.jpg", None),
    ("page.png", Path("base") / "page.png"),
    ("pages/page.png", Path("base") / "pages" / "page.png"),
])
def test_paths_are_read_from_disk_and_urls_are_downloaded(url, expected):
    assert local_path(url, "base") == expected


def test_absolute_and_file_urls_ignore_the_base(tmp_path):
    page = tmp_path / "page.png"
    assert local_path(str(page), "elsewhere") == page
    assert local_path(page.as_uri(), "elsewhere") == page


def test_fetcher_reads_a_relative_path_from_the_annotation_folder(tmp_path):
    (tmp_path / "page.json").write_text('{"Blocks": []}')
    assert Fetcher(base_dir=tmp_path).json("page.json") == {"Blocks": []}


def test_cache_folder_for_a_url_with_a_port_has_no_colon(tmp_path):
    path = Fetcher(tmp_path)._cache_path("http://localhost:8000/data/page.png")
    assert path == tmp_path / "localhost_8000" / "data" / "page.png"


def test_folders_keep_each_datasets_own_file_names(tmp_path):
    doc = Document(url="https://host/stefan/rvlcdip/train_letter_abc", annotations=[])
    local = in_folders(DATASETS["rvlcdip"], images=tmp_path / "img", ocr=tmp_path / "ocr")
    assert local.image_url(doc) == str(tmp_path / "img" / "train_letter_abc.jpg")
    assert local.ocr_url(doc) == str(tmp_path / "ocr" / "train_letter_abc.json")
    only_images = in_folders(DATASETS["rvlcdip"], images=tmp_path)
    assert only_images.ocr_url(doc) == DATASETS["rvlcdip"].ocr_url(doc)  # OCR still where the url says


def test_generic_dataset_finds_ocr_beside_the_image():
    doc = Document(url="scans/page 1.tif", annotations=[])
    generic = DATASETS["generic"]
    assert (generic.image_url(doc), generic.ocr_url(doc), generic.output_name(doc)) == (
        "scans/page 1.tif", "scans/page 1.json", "page 1.tif")


def _changed(out: Path) -> float:
    before = np.asarray(Image.open(EXAMPLES / "page.png").convert("L")).astype(int)
    return float(np.abs(np.asarray(Image.open(out).convert("L")).astype(int) - before).mean())


def test_the_bundled_example_runs_from_disk(tmp_path):
    assert cli.main([
        "redact", str(EXAMPLES / "annotations.json"), "-o", str(tmp_path), "--seed", "0", "--no-match-font",
    ]) == 0
    assert _changed(tmp_path / "page.png") > 0.5


def test_folders_override_a_hosted_url_and_no_ocr_still_redacts(tmp_path):
    shutil.copy(EXAMPLES / "page.png", tmp_path / "page.png")
    hosted = json.loads((EXAMPLES / "annotations_hosted.json").read_text(encoding="utf-8"))
    assert hosted[0]["url"].startswith("http://")  # never fetched: --images points at the local copy
    assert cli.main([
        "redact", str(EXAMPLES / "annotations_hosted.json"), "--images", str(tmp_path),
        "--no-ocr", "-o", str(tmp_path / "out"), "--seed", "0",
    ]) == 0
    assert _changed(tmp_path / "out" / "page.png") > 0.5


def test_a_missing_ocr_file_fails_the_page_rather_than_redacting_blind(tmp_path):
    shutil.copy(EXAMPLES / "page.png", tmp_path / "page.png")
    shutil.copy(EXAMPLES / "annotations.json", tmp_path / "annotations.json")
    assert cli.main(["redact", str(tmp_path / "annotations.json"), "-o", str(tmp_path / "out")]) == 1
