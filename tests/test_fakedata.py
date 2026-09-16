import re
from collections import Counter

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


def test_ssns_have_the_shape_the_ssa_issues_with_no_corpus_wide_tell():
    """Ordinary-looking numbers: no reserved area, group or serial that would mark every page alike."""
    fakedata.seed(2)
    from doc_redact.style import DigitTemplate
    texts = [fakedata.fake_ssn() for _ in range(300)] + [fakedata.fake_ssn(DigitTemplate("123 45 6789")) for _ in range(300)]
    for text in texts:
        assert re.fullmatch(r"\d{3}[- ]\d{2}[- ]\d{4}", text)
        area, group, serial = (int(p) for p in re.findall(r"\d+", text))
        assert 1 <= area <= 899 and area != 666
        assert 1 <= group <= 99 and 1 <= serial <= 9999
    assert len({t[0] for t in texts}) > 5
    assert max(Counter(t[0] for t in texts).values()) < 0.3 * len(texts)


def test_cities_are_real_places_and_zips_match_their_state():
    from doc_redact.resources import ZIP3_RANGES, us_cities
    real = set(us_cities())
    fakedata.seed(4)
    for _ in range(200):
        street, city_line = fakedata.fake_address().split("\n")
        city, rest = city_line.split(",")
        state, zip_code = rest.split()
        assert city in real, city  # no invented city names
        assert any(lo <= int(zip_code[:3]) <= hi for lo, hi in ZIP3_RANGES[state])
        assert re.match(r"^\d+ ", street)


def test_street_names_do_not_repeat_across_a_corpus_worth_of_addresses():
    fakedata.seed(5)
    names = [fakedata._street_name() for _ in range(4000)]
    assert len(set(names)) > 1500


def test_seed_is_reproducible():
    fakedata.seed(42)
    first = [fakedata.fake_text(c) for c in sorted(fakedata.GENERATORS)]
    fakedata.seed(42)
    assert [fakedata.fake_text(c) for c in sorted(fakedata.GENERATORS)] == first


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        fakedata.fake_text("favorite_color")
