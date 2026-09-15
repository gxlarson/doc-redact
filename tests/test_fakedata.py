import re

import pytest

from doc_redact import fakedata


@pytest.mark.parametrize("category", sorted(fakedata.GENERATORS))
def test_every_generator_returns_nonempty_text(category):
    fakedata.seed(0)
    for _ in range(50):
        text = fakedata.fake_text(category)
        assert isinstance(text, str) and text


def test_address_is_two_short_civilian_lines():
    fakedata.seed(1)
    for _ in range(100):
        lines = fakedata.fake_address().split("\n")
        assert len(lines) == 2
        assert not re.search(r"\b[FAD]PO\b", lines[1])
        assert len(lines[0]) <= 25


def test_seed_is_reproducible():
    fakedata.seed(42)
    first = [fakedata.fake_text(c) for c in sorted(fakedata.GENERATORS)]
    fakedata.seed(42)
    assert [fakedata.fake_text(c) for c in sorted(fakedata.GENERATORS)] == first


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        fakedata.fake_text("favorite_color")
