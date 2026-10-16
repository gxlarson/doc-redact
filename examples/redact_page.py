"""Redact one page from Python, step by step.

The page is examples/page.png, a form filled in with generated values, and its OCR is
examples/page.json. The script replaces six fields on it, first with white boxes and then
with fake text, and writes the results and a side-by-side comparison to examples/output/.

    pip install -e .
    python examples/redact_page.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from doc_redact import fakedata, pipeline, render
from doc_redact.annotations import Annotation, BBox, Document

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# 1. Load the page image and its OCR. The OCR is the Textract JSON for the page; the
#    replacement text is matched to the words in it (typeface, size, format, skew).
page = Image.open(HERE / "page.png").convert("RGB")
ocr = json.loads((HERE / "page.json").read_text(encoding="utf-8"))

# 2. Describe the boxes to replace, in pixels from the top-left corner of the image. Each
#    box has a category, which determines the kind of fake text written into it. The
#    typed name and the handwritten name belong to the same person, so they receive the
#    same fake name, and the spouse receives the same fake surname.
boxes = [
    Annotation("name", BBox(left=330, top=138, width=121, height=26)),        # typed name
    Annotation("ssn", BBox(left=330, top=190, width=121, height=26)),
    Annotation("birth_date", BBox(left=330, top=242, width=154, height=26)),
    Annotation("name", BBox(left=333, top=558, width=143, height=31)),        # handwritten name
    Annotation("name", BBox(left=328, top=618, width=167, height=28)),        # handwritten spouse
    Annotation("birth_date", BBox(left=334, top=677, width=126, height=21)),  # handwritten date
]
doc = Document(url="page.png", annotations=boxes)

# 3. White boxes: each box is filled with white and nothing else is changed.
white = pipeline.redact_document(page, doc, ocr, pipeline.Options(
    mode=pipeline.Mode.WHITE, text=render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=24)))

# 4. Fake text: each box is erased with paper texture sampled from around it and filled
#    with generated text in the matched typeface. The seed makes the generated values
#    reproducible.
fakedata.seed(0)
np.random.seed(0)
pseudo = pipeline.redact_document(page, doc, ocr, pipeline.Options(
    mode=pipeline.Mode.PSEUDO, text=render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=24)))

# 5. Save the two redacted pages, and a comparison of the filled-in part of the form.
OUT.mkdir(exist_ok=True)
white.save(OUT / "page_white.png")
pseudo.save(OUT / "page_pseudo.png")

crop = (40, 120, 560, 710)
panels = [("original", page), ("white boxes", white), ("fake text", pseudo)]
width, height = crop[2] - crop[0], crop[3] - crop[1]
sheet = Image.new("RGB", (3 * width + 40, height + 40), "white")
draw = ImageDraw.Draw(sheet)
for i, (label, image) in enumerate(panels):
    x = 10 + i * (width + 10)
    draw.text((x, 10), label, fill="black")
    sheet.paste(image.crop(crop), (x, 30))
    for box in boxes:  # outline the replaced boxes in red
        b = box.bbox
        left, top = x + b.left - crop[0], 30 + b.top - crop[1]
        draw.rectangle([left - 2, top - 2, left + b.width + 2, top + b.height + 2], outline=(220, 40, 40))
sheet.save(OUT / "comparison.png")

print(f"wrote {OUT / 'page_white.png'}")
print(f"wrote {OUT / 'page_pseudo.png'}")
print(f"wrote {OUT / 'comparison.png'}")
