"""Build a synthetic form whose "original" values are invented, then redact it.

The stand-in figures (scripts/paper_figures.py) show real scanner output, but they can only cover
what boilerplate can stand in for: printed dates and reference numbers. They cannot show handwriting,
because handwriting is never boilerplate, and they cannot show a name or an address in a filled-in
field. This builds such a page from nothing: every value on it comes from the generators, so the
"before" crop is publishable, and the page is degraded so the pipeline sees something scan-like.

The page carries its own Textract-style OCR, written from the boxes we drew, so the real pipeline
runs on it unchanged - the same typeface matching, baseline fitting, linking and region logic.

    python scripts/synthetic_page.py -o figures/synthetic --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc_redact import augment, background, fakedata, geometry, handwriting, layout, ocr, pipeline, render  # noqa: E402
from doc_redact import style as st  # noqa: E402
from doc_redact.annotations import Annotation, BBox, Document  # noqa: E402

log = logging.getLogger("synthetic_page")

WIDTH, HEIGHT = 900, 820
HAND_SIZE = 30

# Each variant is a different office: what the form was printed with, what it was filled in with, and
# whose hand signed it. The pipeline has to match whatever it is handed, so the figures should differ.
VARIANTS = {
    "typewriter": dict(label="LiberationSans-Regular.ttf", value="CourierPrime-Regular.ttf",
                       hand="Caveat-Variable.ttf", title="EMPLOYEE RECORD"),
    "serif": dict(label="texgyreschola-regular.otf", value="LiberationSerif-Regular.ttf",
                  hand="HomemadeApple-Regular.ttf", title="PERSONNEL FILE"),
    "elite": dict(label="Carlito-Regular.ttf", value="SpecialElite-Regular.ttf",
                  hand="ArchitectsDaughter-Regular.ttf", title="APPLICATION FOR EMPLOYMENT"),
    # Colour: a form printed in dark blue, filled in on a typewriter, signed in blue ballpoint.
    "color": dict(label="Carlito-Regular.ttf", value="CourierPrime-Regular.ttf",
                  hand="Kalam-Regular.ttf", title="MEMBERSHIP RECORD",
                  label_ink=(24, 56, 120), value_ink=(28, 28, 32), hand_ink=(26, 52, 160),
                  rule_ink=(24, 56, 120)),
}


@dataclass
class Field:
    label: str
    category: str
    value: str
    handwritten: bool = False
    box: BBox | None = None  # the value's box, filled in as it is drawn
    words: list[tuple[str, BBox]] = field(default_factory=list)


def compose(seed: int) -> tuple[list[Field], str]:
    """The fields of one invented personnel form. Every value comes from the generators."""
    fakedata.seed(seed)
    random.seed(seed)
    # Ask for a plain "First Last" so the parts are unambiguous; fake_name can also write "Last, First".
    plain = st.NameStyle((st.NameToken("first"), st.NameToken("last")), last_first=False)
    person = fakedata.fake_name(plain)
    first, surname = person.split()[0], person.split()[-1]
    address = fakedata.fake_address()
    street, city_line = address.split("\n")
    fields = [
        Field("NAME OF EMPLOYEE", "name", person),
        Field("SOCIAL SECURITY NO.", "ssn", fakedata.fake_ssn()),
        Field("DATE OF BIRTH", "birth_date", fakedata.fake_date()),
        Field("PLACE OF BIRTH", "birth_place", fakedata.fake_birthplace()),
        Field("HOME ADDRESS", "street", street),
        # The whole city line, so the style detector sees "City, ST 12345" rather than a bare city.
        Field("CITY, STATE, ZIP", "home_address", city_line),
        Field("HOME TELEPHONE", "home_phone", fakedata.fake_phone()),
        Field("MARITAL STATUS", "marital_status", fakedata.fake_marital_status()),
        # Handwritten entries: the same person signing in, and a second mention of the surname.
        Field("SIGNED BY (PRINT)", "name", f"{first} {surname}", handwritten=True),
        Field("SPOUSE NAME", "name", f"{fakedata.fake_name(plain).split()[0]} {surname}", handwritten=True),
        Field("DATE SIGNED", "birth_date", fakedata.fake_date(), handwritten=True),
    ]
    return fields, person


def draw_hand_word(page: Image.Image, xy: tuple[float, float], word: str, font: str, size: int,
                   ink: tuple[int, int, int] = (0, 0, 0)) -> BBox:
    """Write one handwritten word onto ``page`` and return the box its ink actually occupies.

    handwriting.draw jitters each letter's size and advance, so it runs wider than the font metrics
    predict. Trusting the metrics here gave boxes too narrow for the words beneath them, and the
    redaction then erased only part of the original -- leaving its tail behind under the replacement.
    """
    layer = Image.new("RGB", page.size, "white")
    handwriting.draw(layer, xy, word, font, size, ink=ink)
    ink = np.asarray(layer.convert("L")) < 250
    if not ink.any():
        return BBox(xy[0], xy[1], layout.text_width(layout.load_font(font, size), word), size)
    rows, cols = np.nonzero(ink)
    page.paste(layer, (0, 0), Image.fromarray((ink * 255).astype(np.uint8), mode="L"))
    left, top = float(cols.min()), float(rows.min())
    return BBox(left, top, float(cols.max()) - left + 1, float(rows.max()) - top + 1)


def draw_page(fields: list[Field], seed: int, variant: dict) -> Image.Image:
    """Render the form, then degrade it so it looks like a scan rather than a screenshot."""
    rng = np.random.default_rng(seed)
    paper = background.generate_paper(background.Paper(color=(246, 244, 238), noise_std=3.2,
                                                       speck_density=0.00035, speck_gray=90.0), HEIGHT, WIDTH, rng)
    page = Image.fromarray(paper)
    draw = ImageDraw.Draw(page)
    label_font = layout.load_font(variant["label"], 15)
    value_font = layout.load_font(variant["value"], 19)
    hand_font = variant["hand"]

    label_ink = variant.get("label_ink", (35, 35, 35))
    value_ink = variant.get("value_ink", (25, 25, 25))
    hand_ink = variant.get("hand_ink", (25, 25, 35))
    rule_ink = variant.get("rule_ink", (60, 60, 60))

    draw.text((60, 52), variant["title"], font=layout.load_font(variant["label"], 26), fill=label_ink)
    draw.line([(60, 92), (WIDTH - 60, 92)], fill=rule_ink, width=2)

    y = 140
    for f in fields:
        draw.text((60, y), f.label, font=label_font, fill=label_ink)
        x = 330
        baseline = y + 20
        cursor = x
        if f.handwritten:
            for word in f.value.split():
                box = draw_hand_word(page, (cursor, y - 4), word, hand_font, HAND_SIZE, hand_ink)
                f.words.append((word, box))
                cursor = box.right + 0.35 * HAND_SIZE  # a written space
            f.box = geometry.extent([ocr.Word(w, b) for w, b in f.words])
        else:
            draw.text((x, y), f.value, font=value_font, fill=value_ink)
            f.box = BBox(x, y - 2, layout.text_width(value_font, f.value), 26)
            for word in f.value.split():
                f.words.append((word, BBox(cursor, f.box.top, layout.text_width(value_font, word), f.box.height)))
                cursor += layout.text_width(value_font, word + " ")
        draw.line([(325, y + 27), (WIDTH - 80, y + 27)], fill=rule_ink, width=1)  # field rule
        y += 52 if not f.handwritten else 58

    degraded = augment.degrade(page.convert("RGB"), rotate=True)
    return degraded


def ocr_json(fields: list[Field]) -> dict:
    """Textract-shaped OCR built from the boxes we drew, so the pipeline runs on this page unchanged."""
    blocks = []
    for f in fields:
        for text, box in f.words:
            blocks.append({
                "BlockType": "WORD",
                "Text": text,
                "TextType": "HANDWRITING" if f.handwritten else "PRINTED",
                "Geometry": {"BoundingBox": {"Left": box.left / WIDTH, "Top": box.top / HEIGHT,
                                             "Width": box.width / WIDTH, "Height": box.height / HEIGHT}},
            })
    return {"Blocks": blocks}


def annotation_record(url: str, fields: list[Field], rounded: bool = False) -> dict:
    """One page's entry in the annotations format that ``doc-redact`` reads."""
    def edge(v: float) -> float:
        return round(v) if rounded else v
    return {"url": url, "annotations": [
        {"category": f.category, "bbox": {"left": edge(f.box.left), "top": edge(f.box.top),
                                          "width": edge(f.box.width), "height": edge(f.box.height)}}
        for f in fields]}


def write_example(out: Path, variant: dict, seed: int) -> None:
    """A ready-to-run input for ``doc-redact redact``: a page, its Textract JSON and its annotations.

    ``annotations.json`` names the page by a path relative to itself, so it runs from disk as it is;
    ``annotations_hosted.json`` names the same page by URL, for trying the download path against a
    local web server.
    """
    fields, _ = compose(seed)
    page = draw_page(fields, seed, variant)
    out.mkdir(parents=True, exist_ok=True)
    page.save(out / "page.png")
    (out / "page.json").write_text(json.dumps(ocr_json(fields), indent=1), encoding="utf-8")
    for name, url in (("annotations.json", "page.png"), ("annotations_hosted.json", "http://localhost:8000/page.png")):
        (out / name).write_text(json.dumps([annotation_record(url, fields, rounded=True)], indent=2),
                                encoding="utf-8")
    log.info("wrote %s: page.png, page.json (OCR), annotations.json, annotations_hosted.json", out)


def build(out: Path, name: str, variant: dict, seed: int, crop_scale: int) -> None:
    """One synthetic page and its redaction, written as <name>_before.png / _after.png."""
    fields, _ = compose(seed)
    page = draw_page(fields, seed, variant)
    doc = Document(url=f"https://example.invalid/synthetic/{name}.png",
                   annotations=[Annotation(f.category, f.box) for f in fields])
    options = pipeline.Options(pipeline.Mode.PSEUDO, render.TextSettings(font=variant["value"], font_size=19))
    fakedata.seed(seed + 1)  # different values than the page was built with
    np.random.seed(seed + 1)
    redacted = pipeline.redact_document(page, doc, ocr_json(fields), options)

    out.mkdir(parents=True, exist_ok=True)
    page.save(out / f"{name}_before.png")
    redacted.save(out / f"{name}_after.png")
    (out / f"{name}_annotations.json").write_text(
        json.dumps([annotation_record(doc.url, fields)], indent=2), encoding="utf-8")

    # A crop around the handwritten block, which the stand-in figures cannot show at all.
    hand = [f.box for f in fields if f.handwritten]
    window = (int(min(b.left for b in hand)) - 280, int(min(b.top for b in hand)) - 16,
              int(max(b.right for b in hand)) + 60, int(max(b.bottom for b in hand)) + 16)
    for label, im in (("before", page), ("after", redacted)):
        crop = im.crop(window)
        crop.resize((crop.width * crop_scale, crop.height * crop_scale), Image.LANCZOS).save(
            out / f"{name}_handwriting_{label}.png")
    log.info("wrote %s_*.png (%s printed in %s, signed in %s, seed %d)",
             name, variant["title"].lower(), variant["value"], variant["hand"], seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-scale", type=int, default=2)
    parser.add_argument("--variants", nargs="*", choices=sorted(VARIANTS), default=sorted(VARIANTS),
                        help="which office to imitate (default: all of them)")
    parser.add_argument("--example", action="store_true",
                        help="write unredacted input for doc-redact (page, OCR, annotations) instead of figures; "
                             "uses the first of --variants")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.example:
        write_example(args.out, VARIANTS[args.variants[0]], args.seed)
        return 0
    for i, name in enumerate(args.variants):
        build(args.out, f"synthetic_{name}", VARIANTS[name], args.seed + 10 * i, args.crop_scale)
    log.info("every value on these pages is generated; nothing on them belongs to a person")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
