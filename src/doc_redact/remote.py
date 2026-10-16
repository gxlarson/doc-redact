"""Read images and OCR JSON from disk, or download them, optionally caching downloads on disk."""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from PIL import Image

TIMEOUT_SECONDS = 60


def local_path(url: str, base_dir: str | Path | None = None) -> Path | None:
    """The file ``url`` names, if it is a path or a file:// URL rather than something to download.

    Relative paths are taken from ``base_dir`` (the annotation file's folder), so an annotation file
    can sit next to its pages and say just "page.png".
    """
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if parsed.scheme and len(parsed.scheme) > 1:  # http, https, s3, ...; one letter is a Windows drive
        return None
    path = Path(url)
    return path if path.is_absolute() or base_dir is None else Path(base_dir) / path


class Fetcher:
    def __init__(self, cache_dir: str | Path | None = None, base_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.base_dir = Path(base_dir) if base_dir else None

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        parsed = urlparse(url)
        # Mirror the URL path so the cache doubles as a readable local copy of the data.
        parts = [p for p in posixpath.dirname(parsed.path).split("/") if p]
        name = posixpath.basename(parsed.path) or hashlib.sha1(url.encode()).hexdigest()
        # A port ("localhost:8000") would put a colon in the folder name, which Windows refuses.
        return self.cache_dir.joinpath(parsed.netloc.replace(":", "_"), *parts, name)

    def fetch_bytes(self, url: str) -> bytes:
        local = local_path(url, self.base_dir)
        if local is not None:
            return local.read_bytes()  # already on disk: nothing to download or cache
        path = self._cache_path(url)
        if path is not None and path.is_file():
            return path.read_bytes()
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            data = response.read()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return data

    def image(self, url: str) -> Image.Image:
        return Image.open(io.BytesIO(self.fetch_bytes(url))).convert("RGB")

    def image_kind(self, url: str) -> tuple[str, str | None]:
        """The stored mode ("L" for a grayscale scan) and format ("JPEG"), before ``image`` converts it."""
        with Image.open(io.BytesIO(self.fetch_bytes(url))) as im:
            return im.mode, im.format

    def image_size(self, url: str) -> tuple[int, int]:
        # Image.open only parses the header, so this avoids decoding the pixels.
        with Image.open(io.BytesIO(self.fetch_bytes(url))) as im:
            return im.size

    def json(self, url: str) -> Any:
        return json.loads(self.fetch_bytes(url))
