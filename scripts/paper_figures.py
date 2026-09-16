"""Before/after figures for a paper, without publishing anyone's PII.

A figure needs a "before" crop, and on this corpus every annotated region is real personal data. So
instead we redact a *stand-in*: a region of a real scan that is demonstrably not personal, but is
shaped like the category it stands in for -- a document date standing in for a birth date, a printed
reference number for an SSN. The "before" crop is then genuine scanner output (real paper, real
typeface, real skew, real rules) and was never anyone's data. Say so in the caption.

Two rules keep it honest, both enforced here rather than left to the eye:

* a stand-in must lie clear of every annotated PII box, and each of its words must appear on at least
  ``--min-df`` distinct pages -- preprinted boilerplate, not something someone wrote;
* the crop that goes in the paper must not overlap any annotated PII box, or it would publish real
  data from elsewhere on the page. Crops that would are dropped, with a note.

    python scripts/paper_figures.py find  --annotations ann.json --cache .cache -o figures/
    python scripts/paper_figures.py render --annotations ann.json --cache .cache -o figures/ \
        --page train_invoice_09998110-857a --box 250,430,120,18 --category birth_date --seed 0

``find`` writes a contact sheet to choose from; ``render`` writes one publication-resolution pair.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc_redact import background, fakedata, geometry, ocr, pipeline, render, style  # noqa: E402
from doc_redact.annotations import Annotation, BBox, Document, bbox_from_dict  # noqa: E402
from doc_redact.datasets import DATASETS  # noqa: E402
from doc_redact.remote import Fetcher  # noqa: E402

log = logging.getLogger("paper_figures")

# Words that mark a form's own label rather than a filled-in value; redacting one reads as a bug.
LABELS = {"name", "address", "date", "social", "security", "number", "no", "mailing", "birth", "other",
          "city", "state", "zip", "phone", "sex", "age", "spouse", "employer", "signature", "title"}


def category_for(text: str) -> str | None:
    """The PII category this non-personal text is shaped like, if any.

    Only dates and digit runs qualify. Place names were tried and dropped: too many ordinary words
    ("Institute", "Basic", "Mark") are also US place names, so the finder kept picking words out of
    the middle of headings.
    """
    flat = " ".join(text.split()).strip(".,;:")
    if not flat or any(t.strip(".,;:").lower() in LABELS for t in flat.split()):
        return None
    if style.detect_date(flat):
        return "birth_date"
    digits = re.sub(r"\D", "", flat)
    if re.fullmatch(r"[\d][\d\-/ ]{5,}[\d]", flat) and len(digits) in (7, 9, 10):
        return "ssn" if len(digits) == 9 else "home_phone"
    return None


def load_pages(fetcher: Fetcher, dataset, annotations: Path, cache: Path):
    """Cached pages that appear in the annotation file, with their OCR words and document frequencies."""
    annotated = {d["url"]: d for d in json.load(open(annotations, encoding="utf-8"))}
    ocr_dir = cache / "cumberland.isis.vanderbilt.edu" / "stefan" / "rvlcdip_ocr"
    df: collections.Counter[str] = collections.Counter()
    loaded = {}
    for path in sorted(ocr_dir.glob("*.json")):
        url = f"https://cumberland.isis.vanderbilt.edu/stefan/rvlcdip/{path.stem}.jpg"
        if url not in annotated:
            continue
        try:
            size = fetcher.image_size(dataset.image_url(Document(url=url, annotations=[])))
        except Exception as exc:
            log.debug("skipping %s: %r", path.stem, exc)
            continue
        words = ocr.words(json.load(open(path, encoding="utf-8")), size)
        loaded[url] = (words, size)
        df.update({w.text.strip(".,:;()").lower() for w in words if w.text.strip()})
    return annotated, loaded, df


def pii_boxes(annotated: dict, url: str, size: tuple[int, int]) -> list[BBox]:
    doc = Document(url=url, annotations=[Annotation(a["category"], bbox_from_dict(a["bbox"]))
                                         for a in annotated[url]["annotations"]])
    return [a.bbox for a in doc.in_pixels(size).annotations]


def find_candidates(fetcher, dataset, annotated, loaded, df, min_df: int):
    """Stand-in regions per phenomenon: {phenomenon: [(url, [(bbox, category)], text)]}."""
    found = collections.defaultdict(list)
    for url, (words, size) in loaded.items():
        boxes = pii_boxes(annotated, url, size)
        runs = []
        for line in geometry.group_lines(words):
            for n in (1, 2, 3, 4):
                for i in range(len(line) - n + 1):
                    run = line[i:i + n]
                    region = geometry.extent(run)
                    if region is None or any(b.intersection_area(region) > 0 for b in boxes):
                        continue
                    if any(w.handwritten for w in run):
                        continue
                    if not all(df[w.text.strip(".,:;()").lower()] >= min_df for w in run):
                        continue
                    category = category_for(" ".join(w.text for w in run))
                    if category:
                        runs.append((region, category, " ".join(" ".join(w.text for w in run).split())))
        if not runs:
            continue
        runs.sort(key=lambda r: -r[0].width)  # prefer the longest match at each spot
        kept = []
        for region, category, text in runs:
            if not any(k[0].intersection_area(region) > 0 for k in kept):
                kept.append((region, category, text))
        repeated = collections.defaultdict(list)
        for region, category, text in kept:
            repeated[text].append((region, category))
        for text, items in repeated.items():
            if len(items) >= 2:
                found["repeated value, one replacement"].append((url, items[:3], text))
                break
        gray = None
        for region, category, text in kept:
            if gray is None:
                gray = np.asarray(fetcher.image(dataset.image_url(Document(url=url, annotations=[]))).convert("L"))
            phenomenon = "rule under the text" if background.detect_rules(gray, region, words) else "style matched"
            found[phenomenon].append((url, [(region, category)], text))
    return found


def redact_standin(fetcher, dataset, url: str, regions, seed: int):
    """The page with only the stand-in regions replaced (every real PII box is left alone)."""
    fakedata.seed(seed)
    np.random.seed(seed)
    doc = Document(url=url, annotations=[Annotation(category, box) for box, category in regions])
    image = fetcher.image(dataset.image_url(doc))
    page_ocr = fetcher.json(dataset.ocr_url(doc))
    options = pipeline.Options(pipeline.Mode.PSEUDO, render.TextSettings(font=dataset.font, font_size=dataset.font_size))
    return image, pipeline.redact_document(image, doc, page_ocr, options)


def crop_window(regions, size, margin=(40, 12)) -> BBox:
    left = min(b.left for b, _ in regions) - margin[0]
    top = min(b.top for b, _ in regions) - margin[1]
    right = max(b.right for b, _ in regions) + margin[0]
    bottom = max(b.bottom for b, _ in regions) + margin[1]
    left, top = max(0, left), max(0, top)
    return BBox(left, top, min(size[0], right) - left, min(size[1], bottom) - top)


def crop_pair(image, redacted, window: BBox, scale: int):
    box = (int(window.left), int(window.top), int(window.right), int(window.bottom))
    out = []
    for im in (image, redacted):
        c = im.convert("RGB").crop(box)
        out.append(c.resize((c.width * scale, c.height * scale), Image.LANCZOS) if scale > 1 else c)
    return out


def cmd_find(args) -> int:
    fetcher = Fetcher(args.cache)
    annotated, loaded, df = load_pages(fetcher, dataset, args.annotations, args.cache)
    log.info("%d cached pages carry annotations", len(loaded))
    found = find_candidates(fetcher, dataset, annotated, loaded, df, args.min_df)
    log.info("candidates: %s", {k: len(v) for k, v in found.items()})

    rows, skipped = [], 0
    for phenomenon, items in sorted(found.items()):
        for url, regions, text in items[:args.per_phenomenon]:
            image, redacted = redact_standin(fetcher, dataset, url, regions, args.seed)
            window = crop_window(regions, image.size)
            if any(window.intersection_area(b) > 0 for b in pii_boxes(annotated, url, image.size)):
                skipped += 1  # would publish real PII from elsewhere on the page
                continue
            scale = max(1, min(4, int(340 / max(1, window.width))))
            rows.append((phenomenon, regions[0][1], url, crop_pair(image, redacted, window, scale), regions))
    if skipped:
        log.info("dropped %d crop(s) that would have shown an annotated PII box", skipped)
    if not rows:
        log.error("no candidates survived; try a lower --min-df or a fuller cache")
        return 1

    pad = 12
    width = max(sum(t.width for t in row[3]) + 3 * pad for row in rows) + 250
    height = sum(max(t.height for t in row[3]) + pad + 16 for row in rows) + 30
    sheet = Image.new("RGB", (width, height), (170, 170, 170))
    draw = ImageDraw.Draw(sheet)
    draw.text((255, 8), "before (real scan, non-personal region)", fill=(0, 0, 0))
    draw.text((255 + rows[0][3][0].width + pad, 8), "after (redacted as if it were PII)", fill=(0, 0, 0))
    y = 26
    index = []
    for phenomenon, category, url, tiles, regions in rows:
        draw.text((6, y + 4), phenomenon[:36], fill=(0, 0, 0))
        draw.text((6, y + 18), f"as {category}", fill=(60, 60, 60))
        draw.text((6, y + 32), url.rsplit("/", 1)[-1][:26], fill=(90, 90, 90))
        x = 250
        for tile in tiles:
            sheet.paste(tile, (x, y))
            x += tile.width + pad
        y += max(t.height for t in tiles) + pad + 16
        # Everything `render` needs to redraw this pair at publication resolution.
        index.append({
            "phenomenon": phenomenon, "category": category, "url": url, "seed": args.seed,
            "page": url.rsplit("/", 1)[-1].rsplit(".", 1)[0],
            "boxes": [f"{b.left:.0f},{b.top:.0f},{b.width:.0f},{b.height:.0f}" for b, _ in regions],
        })
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out / "candidates.png")
    (args.out / "candidates.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows)", args.out / "candidates.png", len(rows))
    return 0


def cmd_render(args) -> int:
    fetcher = Fetcher(args.cache)
    annotated = {d["url"]: d for d in json.load(open(args.annotations, encoding="utf-8"))}
    url = next((u for u in annotated if args.page in u), None)
    if url is None:
        log.error("no annotated page matches %r", args.page)
        return 1
    regions = [(BBox(*(float(v) for v in spec.split(","))), args.category) for spec in args.box]
    image, redacted = redact_standin(fetcher, dataset, url, regions, args.seed)
    window = crop_window(regions, image.size, margin=(args.margin, args.margin // 3 or 1))
    overlap = [b for b in pii_boxes(annotated, url, image.size) if window.intersection_area(b) > 0]
    if overlap:
        log.error("refusing: the crop would show %d annotated PII box(es); move it or shrink --margin", len(overlap))
        return 1
    before, after = crop_pair(image, redacted, window, args.scale)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = url.rsplit("/", 1)[-1].rsplit(".", 1)[0][:40]
    before.save(args.out / f"{stem}_before.png")
    after.save(args.out / f"{stem}_after.png")
    log.info("wrote %s_before.png and _after.png (%dx%d, seed %d)", stem, before.width, before.height, args.seed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--annotations", type=Path, required=True)
    common.add_argument("--cache", type=Path, required=True, help="the --cache directory from a redaction run")
    common.add_argument("-o", "--out", type=Path, required=True)
    common.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("find", parents=[common], help="contact sheet of candidate stand-in figures")
    f.add_argument("--min-df", type=int, default=8, help="a stand-in's words must appear on this many pages")
    f.add_argument("--per-phenomenon", type=int, default=2)
    f.set_defaults(func=cmd_find)

    r = sub.add_parser("render", parents=[common], help="render one figure at publication resolution")
    r.add_argument("--page", required=True, help="substring of the page url")
    r.add_argument("--box", action="append", required=True, metavar="LEFT,TOP,WIDTH,HEIGHT")
    r.add_argument("--category", required=True)
    r.add_argument("--margin", type=int, default=40)
    r.add_argument("--scale", type=int, default=4)
    r.set_defaults(func=cmd_render)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    return args.func(args)


dataset = DATASETS["rvlcdip"]

if __name__ == "__main__":
    raise SystemExit(main())
