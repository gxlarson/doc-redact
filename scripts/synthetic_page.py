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

from doc_redact import augment, background, fakedata, handwriting, layout, pipeline, render  # noqa: E402
from doc_redact import style as st  # noqa: E402
from doc_redact.annotations import Annotation, BBox, Document  # noqa: E402

log = logging.getLogger("synthetic_page")

WIDTH, HEIGHT = 900, 820
LABEL_FONT, VALUE_FONT = "LiberationSans-Regular.ttf", "CourierPrime-Regular.ttf"
HAND_FONT, HAND_SIZE = "Caveat-Variable.ttf", 30


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


def draw_page(fields: list[Field], seed: int) -> Image.Image:
    """Render the form, then degrade it so it looks like a scan rather than a screenshot."""
    rng = np.random.default_rng(seed)
    paper = background.generate_paper(background.Paper(color=(246, 244, 238), noise_std=3.2,
                                                       speck_density=0.00035, speck_gray=90.0), HEIGHT, WIDTH, rng)
    page = Image.fromarray(paper)
    draw = ImageDraw.Draw(page)
    label_font = layout.load_font(LABEL_FONT, 15)
    value_font = layout.load_font(VALUE_FONT, 19)

    draw.text((60, 52), "EMPLOYEE RECORD", font=layout.load_font(LABEL_FONT, 26), fill=(20, 20, 20))
    draw.line([(60, 92), (WIDTH - 60, 92)], fill=(40, 40, 40), width=2)

    y = 140
    for f in fields:
        draw.text((60, y), f.label, font=label_font, fill=(35, 35, 35))
        x = 330
        baseline = y + 20
        if f.handwritten:
            handwriting.draw(page, (x, y - 4), f.value, HAND_FONT, HAND_SIZE)
            f.box = BBox(x, y - 4, layout.text_width(layout.load_font(HAND_FONT, HAND_SIZE), f.value), 32)
        else:
            draw.text((x, y), f.value, font=value_font, fill=(25, 25, 25))
            f.box = BBox(x, y - 2, layout.text_width(value_font, f.value), 26)
        # Word boxes, for the OCR we hand the pipeline.
        cursor = x
        for word in f.value.split():
            font = layout.load_font(HAND_FONT, HAND_SIZE) if f.handwritten else value_font
            w = layout.text_width(font, word + " ")
            f.words.append((word, BBox(cursor, f.box.top, layout.text_width(font, word), f.box.height)))
            cursor += w
        draw.line([(325, y + 27), (WIDTH - 80, y + 27)], fill=(60, 60, 60), width=1)  # field rule
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-scale", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fields, person = compose(args.seed)
    page = draw_page(fields, args.seed)
    doc = Document(url="https://example.invalid/synthetic/employee_record.png",
                   annotations=[Annotation(f.category, f.box) for f in fields])
    options = pipeline.Options(pipeline.Mode.PSEUDO,
                               render.TextSettings(font=VALUE_FONT, font_size=19))
    fakedata.seed(args.seed + 1)  # different values than the page was built with
    np.random.seed(args.seed + 1)
    redacted = pipeline.redact_document(page, doc, ocr_json(fields), options)

    args.out.mkdir(parents=True, exist_ok=True)
    page.save(args.out / "synthetic_before.png")
    redacted.save(args.out / "synthetic_after.png")
    (args.out / "synthetic_annotations.json").write_text(
        json.dumps([{"url": doc.url, "annotations": [
            {"category": a.category, "bbox": {"left": a.bbox.left, "top": a.bbox.top,
                                              "width": a.bbox.width, "height": a.bbox.height}}
            for a in doc.annotations]}], indent=2), encoding="utf-8")

    # A crop around the handwritten block, which the stand-in figures cannot show at all.
    hand = [f.box for f in fields if f.handwritten]
    window = (int(min(b.left for b in hand)) - 280, int(min(b.top for b in hand)) - 16,
              int(max(b.right for b in hand)) + 60, int(max(b.bottom for b in hand)) + 16)
    for name, im in (("before", page), ("after", redacted)):
        crop = im.crop(window)
        crop = crop.resize((crop.width * args.crop_scale, crop.height * args.crop_scale), Image.LANCZOS)
        crop.save(args.out / f"synthetic_handwriting_{name}.png")
    log.info("wrote %s (page %dx%d, %d fields, seed %d)", args.out, page.width, page.height, len(fields), args.seed)
    log.info("the page's own values are generated; nothing on it belongs to a person")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
