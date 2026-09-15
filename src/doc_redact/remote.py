"""Download images and OCR JSON, optionally caching them on disk."""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

TIMEOUT_SECONDS = 60


class Fetcher:
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        parsed = urlparse(url)
        # Mirror the URL path so the cache doubles as a readable local copy of the data.
        parts = [p for p in posixpath.dirname(parsed.path).split("/") if p]
        name = posixpath.basename(parsed.path) or hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir.joinpath(parsed.netloc, *parts, name)

    def fetch_bytes(self, url: str) -> bytes:
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

    def image_size(self, url: str) -> tuple[int, int]:
        # Image.open only parses the header, so this avoids decoding the pixels.
        with Image.open(io.BytesIO(self.fetch_bytes(url))) as im:
            return im.size

    def json(self, url: str) -> Any:
        return json.loads(self.fetch_bytes(url))
