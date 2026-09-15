# doc-redact

Redact PII regions in scanned document images, either by whiting them out or by
replacing them with realistic fake text that is degraded to look printed.

Supports RVL-CDIP, FUNSD, Tobacco3482 and Tobacco800. Images and their AWS
Textract OCR are downloaded from the hosting server named in each annotation's `url`.

## Install

```sh
pip install -e ".[dev]"
```

## Usage

```sh
# White boxes over every annotated region
doc-redact redact annotations/redactions_FUNSD_parsed.json \
    --dataset funsd --mode white -o output/FUNSD_white

# Fake replacement text (default mode), reproducible, with a local download cache
doc-redact redact annotations/redactions_tobacco3482_parsed.json \
    --dataset tobacco3482 -o output/tobacco3482_pseudo --seed 0 --cache .cache

# Redo a single page with fixed text for some categories
doc-redact redact annotations/redactions_tobacco3482_parsed.json \
    --dataset tobacco3482 -o output/redo --only 50536499-6500 \
    --text birth_date="Oct.8,14" --text birth_place="Orance, CA"

# Old behaviour: fixed font size, random formats, no width fitting
doc-redact redact annotations/redactions_tobacco3482_parsed.json \
    --dataset tobacco3482 -o output/plain --font-size 34 --no-fit-width --no-match-style

# Audit annotations: show the OCR text under each box and flag likely label errors
doc-redact check annotations/redactions_tobacco3482_parsed.json \
    --dataset tobacco3482 --cache .cache --problems-only --json check.json

# Convert a Label Studio export into the annotations format
doc-redact parse-labelstudio redactions_FUNSD.json redactions_FUNSD_parsed.json \
    --base-url https://cumberland.isis.vanderbilt.edu/stefan/FUNSD/
```

`python -m doc_redact ...` works without installing (with `src` on `PYTHONPATH`).

Options for `redact`:

| Option | Meaning |
|---|---|
| `--dataset` | `funsd`, `tobacco3482`, `tobacco800`, `rvlcdip`; sets URL rules and default font |
| `--mode` | `pseudo` (default) or `white` |
| `--[no-]match-font` | pick, per box, the bundled typeface that best matches the scanned words (default on) |
| `--system-fonts` | also consider installed fonts (Times New Roman, Courier New, Georgia, ...); results then depend on the machine |
| `--font` | use this one font everywhere (path, bundled file name, or system font name); turns off `--match-font` |
| `--font-size N` | fixed size (turns off auto sizing); also the fallback when auto sizing has no words to measure |
| `--[no-]auto-font-size` | size text so the OCR words under the box, re-rendered in `--font`, span their original width (default on) |
| `--[no-]fit-width` | pick the candidate closest to the original text width, and shrink if it would overflow the box (default on) |
| `--[no-]match-weight` | thicken strokes to match the scanned text under the box (default on) |
| `--[no-]match-style` | mimic the original's format: date layout, state form, digit separators, address part, casing, trailing punctuation (default on) |
| `--only ID` | process only these documents (repeatable) |
| `--text CATEGORY=TEXT` | fixed replacement text for a category |
| `--seed N` | reproducible fake data and augmentation |
| `--cache DIR` | keep downloaded images/OCR on disk |
| `--keep-unannotated` | also save pages that have no annotations |

Failed pages are logged and skipped; the command exits non-zero if any failed.

## Dataset defaults

| Dataset | Image URL | OCR URL | Fallback font |
|---|---|---|---|
| `funsd` | `url` with `.png` | `FUNSD_ocr/<id>.json` | Inconsolata 14 |
| `tobacco3482` | `url` | `url` with `.json` | Liberation Serif 34 |
| `tobacco800` | `url` | `url` with `.json` | Courier Prime 21 |
| `rvlcdip` | `url` | `rvlcdip_ocr/<id>.json` | Inconsolata 13 |

The fallback font and size are used only when a box has no measurable OCR words nearby.

## Annotation format

```json
[{"url": "https://.../50650671-0672.jpg",
  "annotations": [{"category": "birth_date",
                   "bbox": {"left": 1273, "top": 327, "width": 121, "height": 31}}]}]
```

Supported categories: `age`, `birth_date`/`date_of_birth`, `birth_place`/`place_of_birth`,
`citizenship_nationality`, `home_address`, `home_phone`, `marital_status`, `religion`, `ssn`.
Other categories (e.g. `sex_gender`, `other`) are whited out and logged.

## How replacement text is placed

For each box, the OCR words at least 70% inside it are grouped into lines. That text drives:

- **Typeface** (`--match-font`): up to three words in the box (or, if none are usable, the nearest words on
  its line) are rendered in every candidate font at the scanned width. Scan and renderings are cleaned of
  specks, reduced to 1-px skeletons and compared by chamfer distance; words vote by rank.

- **Font size**: estimated from word widths (Textract pads boxes vertically, so heights overestimate).
  Falls back to words in the same row, then the dataset default.
- **Weight**: stroke thickness is measured on the scanned words (ink area / half the boundary length) and on the
  same words re-rendered in `--font`; text is drawn at 4x with the outline that best matches, then downsampled,
  so weight changes in 0.5 px steps. Fonts heavier than the scan are not thinned.
- **Position**: glyph tops align with the original words' left/top; empty boxes are centered vertically.
- **Line count**: addresses stay two lines only if the original spans two lines.
- **Style** (`--match-style`): e.g. `4/12/50` → `9/3/61`, `Marfa, Tex.` → `Olney, Ark.`,
  `(555) 014-2233` → `(804) 311-9072`, `Springfield, PA 19064` → city line only, `MARRIED` → upper case.
  Only the format is copied, never the value. It does reveal the original's format and approximate
  width, which matters if that is sensitive for your use.

`check` flags: `no_ocr_words`, `looks_like_date`, `looks_like_ssn`, `no_digits`, `only_digits`
(likely label problems) and `style_not_detected` (informational; a random format will be used).

## Fonts

19 open-license fonts are bundled (5.2 MB) so matching works the same on every machine:

| Kind | Bundled | Stands in for |
|---|---|---|
| Serif | Liberation Serif (+ Bold), Gelasio, TeX Gyre Pagella / Schola / Bonum | Times, Georgia, Palatino, Century, Bookman |
| Sans | Liberation Sans (+ Bold), Carlito, DejaVu Sans | Arial, Calibri, Verdana |
| Typewriter, serifed | Courier Prime (+ Bold), Courier 10 Pitch (+ Bold), Cutive Mono, Special Elite | Courier and worn typewriter faces |
| Typewriter, sans | Liberation Mono, Inconsolata, DejaVu Sans Mono | Consolas, Lucida Console |

Licenses (SIL OFL, Apache 2.0, GUST, Bitstream, DejaVu) are listed in `src/doc_redact/fonts/FONTS.txt`
and ship next to the fonts.

## Layout

```
src/doc_redact/
  cli.py          argument parsing, subcommands
  pipeline.py     per-document and batch redaction
  datasets.py     URL/output/font conventions per dataset
  remote.py       downloads with optional disk cache
  annotations.py  BBox/Document types, JSON I/O, Label Studio parser
  ocr.py          Textract WORD blocks -> pixel boxes
  geometry.py     word coverage and line-height heuristics
  checks.py       annotation audit rules
  render.py       white-out, replacement planning and drawing
  layout.py       font loading, size estimation, width fitting, supersampled drawing
  weight.py       stroke thickness measurement and matching
  typeface.py     candidate fonts and per-box typeface matching
  fonts/          bundled open-license fonts (see fonts/FONTS.txt)
  style.py        format detection (dates, places, digits, addresses, casing)
  augment.py      scan-like degradation and compositing
  fakedata.py     replacement text generators
  resources.py    bundled word lists and US state names
```

## Tests

```sh
pytest
```
