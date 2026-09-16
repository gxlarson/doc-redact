"""Linking repeated names to one fake identity. All names here are invented."""
import pytest

from doc_redact import fakedata, people, render, style
from doc_redact.annotations import Annotation, BBox
from doc_redact.ocr import Word


def _parsed(text):
    return style.parse_name(text)


@pytest.mark.parametrize("a, b", [
    ("Quimby, Ansel R.", "Ansel R. Quimby"),
    ("QUIMBY, Ansel", "Ansel Quimby"),
    ("A. R. Quimby", "Ansel Quimby"),
    ("Ansel R. Quimby", "Ansel Quimbv"),  # one OCR slip in a long surname
    ("Mr. Ansel Quimby Jr.", "Quimby, A."),
    ("Nella Quimby", "Nell R. Quimby"),   # short given name, one letter dropped
])
def test_same_person(a, b):
    assert people.same_person(_parsed(a), _parsed(b))


@pytest.mark.parametrize("a, b", [
    ("Ansel Quimby", "Bertha Quimby"),   # same family, different people
    ("Ansel R. Quimby", "Ansel T. Quimby"),  # conflicting middle initials
    ("Mrs. Quimby", "Ansel Quimby"),     # no first name on one side: could be a spouse
    ("Ansel Quimby", "Ansel Wendover"),
    ("Ansel Fell", "Ansel Bell"),        # short surnames must match exactly
    ("Nora Quimby", "Nola Quimby"),      # short given names differing by a substituted letter
])
def test_different_people(a, b):
    assert not people.same_person(_parsed(a), _parsed(b))


def test_initials_know_whether_they_are_first_or_middle():
    assert [t.of for t in _parsed("A. R. Quimby").style.tokens if t.kind == "initial"] == ["first", "middle"]
    assert [t.of for t in _parsed("Ansel R. Quimby").style.tokens if t.kind == "initial"] == ["middle"]
    assert [t.of for t in _parsed("Quimby, A. R.").style.tokens if t.kind == "initial"] == ["first", "middle"]


def _page_with_names(entries):
    """entries: [(text, left, top)] -> OCR words and one name annotation per entry."""
    words, annotations = [], []
    for text, left, top in entries:
        x = left
        for token in text.split():
            words.append(Word(token, BBox(x, top, 12 * len(token), 20)))
            x += 12 * len(token) + 10
        annotations.append(Annotation("name", BBox(left - 3, top - 4, x - left + 6, 28)))
    return words, annotations


def test_repeated_name_gets_one_replacement_in_each_format():
    words, annotations = _page_with_names([
        ("Quimby, Ansel R.", 100, 100),
        ("Ansel R. Quimby", 100, 300),
        ("Bertha Quimby", 100, 500),
        ("Delia Wendover", 100, 700),
    ])
    fakedata.seed(4)
    identities = people.link(annotations, words)
    assert identities[annotations[0]] is identities[annotations[1]]
    assert identities[annotations[2]] is not identities[annotations[0]]
    assert identities[annotations[2]].last == identities[annotations[0]].last  # family keeps one surname
    assert identities[annotations[3]].last != identities[annotations[0]].last

    page = render.Page(words=words, identities=identities)
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=20, auto_font_size=False, fit_width=False)
    first_box = render.plan_text(annotations[0], page, settings).text
    second_box = render.plan_text(annotations[1], page, settings).text
    person = identities[annotations[0]]
    assert first_box == f"{person.last}, {person.first} {person.middle[0]}."
    assert second_box == f"{person.first} {person.middle[0]}. {person.last}"
    assert "Quimby" not in first_box + second_box


def test_salutation_with_surname_only_parses_and_links_to_the_one_matching_person():
    parsed = style.parse_name("Dear Mr. Quimby:")
    assert [t.kind for t in parsed.style.tokens] == ["salutation", "title", "last"]
    assert parsed.last == "Quimby" and parsed.first_letter is None

    words, annotations = _page_with_names([
        ("Quimby, Ansel R.", 100, 100), ("Ansel R. Quimby", 100, 300), ("Dear Mr. Quimby:", 100, 500),
    ])
    fakedata.seed(1)
    identities = people.link(annotations, words)
    person = identities[annotations[0]]
    assert identities[annotations[1]] is person and identities[annotations[2]] is person
    assert person.gender == "male"  # from the salutation's title

    page = render.Page(words=words, identities=identities)
    settings = render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=20, auto_font_size=False, fit_width=False)
    assert render.plan_text(annotations[2], page, settings).text == f"Dear Mr. {person.last}:"


def test_surname_only_stays_separate_when_it_is_ambiguous_or_contradicts_gender():
    words, annotations = _page_with_names([
        ("Ansel Quimby", 100, 100), ("Bertha Quimby", 100, 300), ("Mr. Quimby", 100, 500),
    ])
    fakedata.seed(1)
    identities = people.link(annotations, words)
    assert len({id(p) for p in identities.values()}) == 3  # two Quimbys: can't tell which one
    assert len({p.last for p in identities.values()}) == 1  # but the family surname is shared

    words, annotations = _page_with_names([("Mrs. Delia Wendover", 100, 100), ("Mr. Wendover", 100, 300)])
    identities = people.link(annotations, words)
    assert identities[annotations[0]] is not identities[annotations[1]]


def test_title_left_of_the_box_sets_the_linked_persons_gender():
    words, annotations = _page_with_names([("Ansel Quimby", 150, 100), ("Quimby, Ansel", 150, 300)])
    words.append(Word("Mrs.", BBox(100, 300, 40, 20)))
    fakedata.seed(0)
    identities = people.link(annotations, words)
    assert identities[annotations[0]].gender == "female"
