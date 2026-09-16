"""Download images and OCR JSON, optionally caching them on disk."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import posixpath
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

TIMEOUT_SECONDS = 60
RETRIES = 4
BACKOFF_SECONDS = 2.0

log = logging.getLogger(__name__)


def _is_transient(error: Exception) -> bool:
    """A dropped connection or a 5xx is worth retrying; a 404 never is."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    return True


def _urlopen_with_retries(url: str) -> bytes:
    """Fetch ``url``, retrying transient failures with exponential backoff.

    Parallel shards hit the hosting server hard enough that it occasionally drops a
    connection; without this, one drop permanently skips a page mid-run.
    """
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - re-raised below when not retryable
            if attempt == RETRIES or not _is_transient(error):
                raise
            delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
            log.warning("fetch %s failed (%s); retrying in %.0fs [%d/%d]", url, error, delay, attempt, RETRIES)
            time.sleep(delay)
    raise AssertionError("unreachable")


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
        data = _urlopen_with_retries(url)
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
