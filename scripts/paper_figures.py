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


def load_pages(fetcher: Fetcher, dataset, annotations: Path, cache: Path, fetch_classes=None, max_pages=0):
    """Pages from the annotation file with their OCR words, plus how often each token occurs across them.

    Cached pages are free. ``fetch_classes`` additionally downloads whole RVL-CDIP classes that are not
    cached yet; a class is downloaded entire (subject to ``max_pages``) rather than page by page,
    because the token counts are what tell boilerplate apart from someone's data, and counts taken over
    a handful of pages would call everything rare.
    """
    annotated = {d["url"]: d for d in json.load(open(annotations, encoding="utf-8"))}
    ocr_dir = cache / "cumberland.isis.vanderbilt.edu" / "stefan" / "rvlcdip_ocr"
    urls = {f"https://cumberland.isis.vanderbilt.edu/stefan/rvlcdip/{p.stem}.jpg" for p in ocr_dir.glob("*.json")}
    urls &= set(annotated)
    if fetch_classes:
        wanted = [u for u in annotated if any(f"_{c}_" in u for c in fetch_classes)]
        if max_pages:
            wanted = wanted[:max_pages]
        log.info("fetching OCR for %d page(s) of %s", len(wanted), ", ".join(fetch_classes))
        urls |= set(wanted)

    df: collections.Counter[str] = collections.Counter()
    loaded = {}
    for url in sorted(urls):
        doc = Document(url=url, annotations=[])
        try:
            size = fetcher.image_size(dataset.image_url(doc))
            page_ocr = fetcher.json(dataset.ocr_url(doc))
        except Exception as exc:
            log.debug("skipping %s: %r", url.rsplit("/", 1)[-1][:28], exc)
            continue
        words = ocr.words(page_ocr, size)
        loaded[url] = (words, size)
        df.update({w.text.strip(".,:;()").lower() for w in words if w.text.strip()})
    return annotated, loaded, df


def pii_boxes(annotated: dict, url: str, size: tuple[int, int]) -> list[BBox]:
    doc = Document(url=url, annotations=[Annotation(a["category"], bbox_from_dict(a["bbox"]))
                                         for a in annotated[url]["annotations"]])
    return [a.bbox for a in doc.in_pixels(size).annotations]


def find_candidates(fetcher, dataset, annotated, loaded, df, min_df: int, handwritten: bool = False):
    """Stand-in regions per phenomenon: {phenomenon: [(url, [(bbox, category)], text)]}.

    ``handwritten`` also accepts entries somebody wrote by hand. Document frequency cannot vouch for
    those -- handwriting is never boilerplate -- so they are held to a narrower rule instead: dates
    only, on the institutional classes, where a handwritten date is the date the document was stamped,
    received or approved rather than anything about a person. Everything the crop shows is still
    screened, and these especially want an eye on them before publishing.
    """
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
                    hand = [w.handwritten for w in run]
                    if any(hand) and not (handwritten and all(hand)):
                        continue  # mixed runs are an OCR seam, not a field
                    if not all(hand) and not all(df[w.text.strip(".,:;()").lower()] >= min_df for w in run):
                        continue
                    category = category_for(" ".join(w.text for w in run))
                    if all(hand) and category != "birth_date":
                        continue  # a handwritten number could be an identifier nobody annotated
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
            inside = geometry.words_inside(region, words)
            if inside and all(w.handwritten for w in inside):
                phenomenon = "handwritten entry"
            elif background.detect_rules(gray, region, words):
                phenomenon = "rule under the text"
            else:
                phenomenon = "style matched"
            found[phenomenon].append((url, [(region, category)], text))
    return found


def redact_standin(fetcher, dataset, url: str, regions, seed: int):
    """The page with only the stand-in regions replaced (every real PII box is left alone).

    Also reports the typeface the matcher settled on, so a figure set can be chosen to show more than
    one of them rather than three examples of the same Courier.
    """
    fakedata.seed(seed)
    np.random.seed(seed)
    doc = Document(url=url, annotations=[Annotation(category, box) for box, category in regions])
    image = fetcher.image(dataset.image_url(doc))
    page_ocr = fetcher.json(dataset.ocr_url(doc))
    options = pipeline.Options(pipeline.Mode.PSEUDO, render.TextSettings(font=dataset.font, font_size=dataset.font_size))

    fonts: list[str] = []
    original_plan = render.plan_text

    def recording_plan(*a, **kw):
        plan = original_plan(*a, **kw)
        fonts.append(Path(plan.font).stem)
        return plan

    render.plan_text = recording_plan
    try:
        redacted = pipeline.redact_document(image, doc, page_ocr, options)
    finally:
        render.plan_text = original_plan
    return image, redacted, fonts


# Content that identifies a person without naming them, or that is sensitive on its own. A career
# history ("Senior Analyst, Regional Water Authority of Northfield, 1981-83") names nobody and still narrows the
# subject to a handful of people; a diagnosis is sensitive whether or not anyone is named.
SENSITIVE = {
    "diagnosis", "patient", "autopsy", "deceased", "death", "cremation", "mortuary", "burial",
    "psychiatric", "surgical", "inpatient", "outpatient", "carcinoma", "cancer", "tumor", "illness",
    "injury", "treatment", "internship", "residency", "curriculum", "vitae", "employment",
}

HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "rev", "hon", "sgt", "capt", "lt", "col", "gen"}
CREDENTIALS = {"m.d.", "md", "ph.d.", "phd", "d.d.s.", "r.n.", "rn", "esq", "jr", "sr", "r.r.a."}
_INITIALS = re.compile(r"[A-Z]\.")
_TITLE_CASE = re.compile(r"[A-Z][a-z]{2,}(?:[-'][A-Z]?[a-z]+)*")  # "Anne-Marie", "O'Brien"
_UPPER_WORD = re.compile(r"[A-Z]{3,}(?:[-'][A-Z]+)*")  # memo headers shout surnames: "WYATT", "MILLER"


def _name_shaped(text: str) -> bool:
    return bool(_TITLE_CASE.fullmatch(text) or _UPPER_WORD.fullmatch(text))


def names_in_crop(window: BBox, words, df) -> list[str]:
    """Person-name evidence inside a crop, as shapes, not the names themselves.

    The annotation file marks only the subject's own PII, so a wider crop can pull in a third party --
    a signing officer, a physician, an agency contact -- that no box covers. Rarity does the work: a
    capitalised word that appears on almost no other page is somebody's name, while the words that make
    up a form's printed furniture appear everywhere.
    """
    # Any overlap counts, not half: a crop edge that clips an addressee line still shows part of the name.
    inside = [w for w in words if window.intersection_area(w.bbox) > 0]
    hits = []
    for i, word in enumerate(inside):
        bare = word.text.strip(".,:;()").lower()
        if bare in SENSITIVE:
            hits.append(f"sensitive content ({bare})")
            continue
        if bare in HONORIFICS or word.text.strip(",:;()").lower() in CREDENTIALS:
            hits.append("honorific/credential")
            continue
        if _INITIALS.fullmatch(word.text) and i + 1 < len(inside) and _name_shaped(inside[i + 1].text.strip(",")):
            hits.append("initial + capitalised word")
            continue
        if _name_shaped(word.text) and df[bare] <= 2:
            nxt = inside[i + 1].text.strip(",") if i + 1 < len(inside) else ""
            # A rare capitalised word beside another is a name; alone in a memo header it is one too.
            if _name_shaped(nxt) and df[nxt.strip(".,:;()").lower()] <= 2:
                hits.append("two rare capitalised words")
            elif _UPPER_WORD.fullmatch(word.text) and df[bare] <= 1:
                hits.append("a rare shouted word, as memo headers write surnames")
    return hits


def safe_margin(regions, pii: list[BBox], size, wanted: int, floor: int = 8) -> int:
    """The largest margin up to ``wanted`` whose crop still touches no annotated PII box.

    A figure reads better with the surrounding form visible, but every pixel of that context comes off
    a real page, so how wide we can go is decided by where the real PII sits, not by taste.
    """
    margin = wanted
    while margin > floor:
        window = crop_window(regions, size, (margin, max(1, margin // 3)))
        if not any(window.intersection_area(b) > 0 for b in pii):
            return margin
        margin = int(margin * 0.8)
    return 0


def crop_window(regions, size, margin=(40, 12)) -> BBox:
    """``margin`` is (x, y), or (left, top, right, bottom) when the page is only safe on some sides:
    a letterhead may be worth showing above the date while the addressee sits just below it."""
    l, t, r, b = margin if len(margin) == 4 else (margin[0], margin[1], margin[0], margin[1])
    left = min(box.left for box, _ in regions) - l
    top = min(box.top for box, _ in regions) - t
    right = max(box.right for box, _ in regions) + r
    bottom = max(box.bottom for box, _ in regions) + b
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
    annotated, loaded, df = load_pages(fetcher, dataset, args.annotations, args.cache,
                                       fetch_classes=args.fetch, max_pages=args.max_pages)
    log.info("%d annotated pages available", len(loaded))
    found = find_candidates(fetcher, dataset, annotated, loaded, df, args.min_df, args.handwritten)
    log.info("candidates: %s", {k: len(v) for k, v in found.items()})

    rows, skipped, unnamed = [], 0, 0
    for phenomenon, items in sorted(found.items()):
        shown = 0
        for url, regions, text in items:
            if shown >= args.per_phenomenon:
                break
            if args.classes and not any(f"_{c}_" in url for c in args.classes):
                continue
            if any(f"_{c}_" in url for c in args.exclude_classes or []):
                continue
            image, redacted, fonts = redact_standin(fetcher, dataset, url, regions, args.seed)
            pii = pii_boxes(annotated, url, image.size)
            margin = safe_margin(regions, pii, image.size, args.context)
            if margin < args.min_context:
                skipped += 1  # real PII sits too close to show any context around this one
                continue
            window = crop_window(regions, image.size, (margin, max(1, margin // 3)))
            evidence = names_in_crop(window, loaded[url][0], df)
            if evidence:
                unnamed += 1
                log.debug("skipping %s: crop shows %s", url.rsplit("/", 1)[-1][:24], sorted(set(evidence)))
                continue
            scale = max(1, min(4, int(340 / max(1, window.width))))
            rows.append((phenomenon, regions[0][1], url, crop_pair(image, redacted, window, scale), regions,
                         margin, fonts[0] if fonts else "?"))
            shown += 1
    if skipped:
        log.info("dropped %d candidate(s): real PII sits within %dpx of the stand-in", skipped, args.min_context)
    if unnamed:
        log.info("dropped %d candidate(s): the crop showed a person's name that no annotation covers", unnamed)
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
    for phenomenon, category, url, tiles, regions, margin, font in rows:
        draw.text((6, y + 4), phenomenon[:36], fill=(0, 0, 0))
        draw.text((6, y + 18), f"as {category}, {margin}px, {font[:22]}", fill=(60, 60, 60))
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
            "margin": margin,  # the widest context this page allows without touching real PII
            "font": font,  # the typeface the matcher chose for this box
        })
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out / "candidates.png")
    (args.out / "candidates.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows)", args.out / "candidates.png", len(rows))
    return 0


def cmd_render(args) -> int:
    fetcher = Fetcher(args.cache)
    annotated, loaded, df = load_pages(fetcher, dataset, args.annotations, args.cache,
                                       fetch_classes=args.fetch, max_pages=args.max_pages)
    url = next((u for u in loaded if args.page in u), None) or next((u for u in annotated if args.page in u), None)
    if url is None:
        log.error("no annotated page matches %r", args.page)
        return 1
    regions = [(BBox(*(float(v) for v in spec.split(","))), args.category) for spec in args.box]
    image, redacted, fonts = redact_standin(fetcher, dataset, url, regions, args.seed)
    pii = pii_boxes(annotated, url, image.size)
    if args.margin == "auto":
        margin = safe_margin(regions, pii, image.size, args.max_margin)
        if not margin:
            log.error("refusing: real PII sits right beside this region, so no context can be shown")
            return 1
        log.info("widest context this page allows: %dpx", margin)
    else:
        margin = int(args.margin)
    pad = tuple(int(v) for v in args.pad.split(",")) if args.pad else (margin, margin // 3 or 1)
    window = crop_window(regions, image.size, margin=pad)
    overlap = [b for b in pii if window.intersection_area(b) > 0]
    if overlap:
        log.error("refusing: the crop would show %d annotated PII box(es); move it or shrink --margin", len(overlap))
        return 1
    if url in loaded:
        evidence = sorted(set(names_in_crop(window, loaded[url][0], df)))
        if evidence:
            log.error("refusing: the crop shows %s, which no annotation covers; shrink --margin", ", ".join(evidence))
            return 1
    else:
        log.warning("this page's OCR is not cached, so the crop was not screened for unannotated names")
    before, after = crop_pair(image, redacted, window, args.scale)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = url.rsplit("/", 1)[-1].rsplit(".", 1)[0][:40]
    before.save(args.out / f"{stem}_before.png")
    after.save(args.out / f"{stem}_after.png")
    log.info("wrote %s_before.png and _after.png (%dx%d, seed %d, matched %s)",
             stem, before.width, before.height, args.seed, ", ".join(dict.fromkeys(fonts)) or "?")
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
    f.add_argument("--context", type=int, default=260, help="px of surrounding page to show, if the page allows it")
    f.add_argument("--min-context", type=int, default=60, help="skip candidates that cannot show this much")
    f.add_argument("--classes", nargs="*", help="only these RVL-CDIP classes, e.g. invoice specification")
    f.add_argument("--fetch", nargs="*", help="download these classes if they are not cached yet")
    f.add_argument("--handwritten", action="store_true",
                   help="also accept handwritten entries as stand-ins (dates only; see find_candidates)")
    f.add_argument("--max-pages", type=int, default=0, help="cap on pages downloaded by --fetch")
    f.add_argument("--exclude-classes", nargs="*", default=["resume", "questionnaire", "handwritten"],
                   help="classes to skip (default: the personal-record classes, whose context identifies "
                        "the subject through career history and the like even with every name removed)")
    f.set_defaults(func=cmd_find)

    r = sub.add_parser("render", parents=[common], help="render one figure at publication resolution")
    r.add_argument("--page", required=True, help="substring of the page url")
    r.add_argument("--box", action="append", required=True, metavar="LEFT,TOP,WIDTH,HEIGHT")
    r.add_argument("--category", required=True)
    r.add_argument("--margin", default="auto", help="px of context, or 'auto' for the widest this page allows")
    r.add_argument("--max-margin", type=int, default=300, help="cap for --margin auto")
    r.add_argument("--pad", help="per-side context as LEFT,TOP,RIGHT,BOTTOM; overrides --margin")
    r.add_argument("--fetch", nargs="*", help="download these classes if they are not cached yet")
    r.add_argument("--max-pages", type=int, default=0, help="cap on pages downloaded by --fetch")
    r.add_argument("--scale", type=int, default=4)
    r.set_defaults(func=cmd_render)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    return args.func(args)


dataset = DATASETS["rvlcdip"]

if __name__ == "__main__":
    raise SystemExit(main())
