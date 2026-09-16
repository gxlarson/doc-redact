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

## Running a whole corpus

Defaults are what you want for a full run: pseudonymized text, matched typeface, size,
weight, skew and format, paper fill and preserved rules. Only `-o`, `--seed` and
`--cache` need choosing.

```sh
doc-redact redact rvlcdip_pii_annotated.json \
    --dataset rvlcdip -o output/rvlcdip_pseudo --seed 0 --cache .cache
```

Cost is dominated by typeface matching, and it is charged **per box**, not per page:
about 1.6 s per box with `--match-font`, 0.3 s without (measured on 1080p-ish RVL-CDIP
scans; handwritten pages cost roughly double, since a second matcher runs over the
handwriting faces). For the 15,956 annotated RVL-CDIP pages (39,435 boxes) that is
roughly **19 core-hours**, or about 4.5 with `--no-match-font`.

Downloads are ~780 KB per page (image plus Textract JSON), so a full RVL-CDIP run pulls
about 12 GB. With `--cache` it is pulled once; pre-fetching before a parallel run keeps
the workers from hammering the server at once:

```sh
doc-redact check rvlcdip_pii_annotated.json --dataset rvlcdip --cache .cache >/dev/null
```

### Across cores

Pages are independent, so split the annotation file and run one process per core.
Output file names come from each document's URL, so every shard can write to the same
`-o` directory.

```sh
# Split into one file per core
python - <<'PY'
import json, math, pathlib
docs = json.load(open("rvlcdip_pii_annotated.json"))
n = 12
size = math.ceil(len(docs) / n)
pathlib.Path("shards").mkdir(exist_ok=True)
for i in range(n):
    json.dump(docs[i * size:(i + 1) * size], open(f"shards/{i:02d}.json", "w"))
PY

# One process per shard, each with its own seed
seq -w 0 11 | xargs -P 12 -I{} \
    doc-redact redact shards/{}.json --dataset rvlcdip \
        -o output/rvlcdip_pseudo --cache .cache --seed {}
```

Give every shard a **different `--seed`**: equal seeds make different pages share the
same generated people, addresses and numbers. Scaling is close to linear (the work is
CPU-bound in numpy/scipy), so 12 cores brings the run above to roughly 1.5 h and 32 to
about 35 min. A shared `--cache` directory is safe to read from concurrently; pre-fetch
it first as above rather than having 12 workers download the same corpus.

Failures are logged per page and skipped, and each process exits non-zero if any of its
pages failed, so check the exit codes rather than assuming a silent run was complete.

### Output files

One file per page, written flat into `--out` (created if missing), named after the
source: RVL-CDIP and Tobacco pages keep their `.jpg` name, FUNSD its `.png`. A grayscale
scan is saved grayscale, even though pages are edited in RGB.

JPEG sources are re-encoded, which touches the whole page and not only the replaced
regions. Measured on RVL-CDIP scans, against the original pixels outside every
annotation:

| output | size/page | drift outside the replaced regions |
|---|---|---|
| `--format same` (JPEG, quality 95) | ~180 KB | 0.31 gray levels |
| `--format png` | ~190 KB | 0.003 (lossless) |
| `--format same --jpeg-quality 60` | ~86 KB | 1.7 gray levels |

If the redacted pages will be compared against unmodified originals, or used to train or
evaluate a model, prefer `--format png`: re-encoding artifacts spread across the page
are a difference between the two sets that has nothing to do with redaction.

`--format both` writes each page twice from the same render — a lossless PNG under
`png/` and a JPEG under `jpg/`, same pixels, same file name. The page is fetched,
matched and drawn once, so the second copy costs only its encode (a few percent of the
run) and the two are guaranteed to hold the same replacements. Useful when you want an
archival copy and a compact one for distribution, without redacting twice and getting two
different sets of fake people.

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
| `--[no-]match-skew` | rotate the replacement to the scanned line's baseline slope, falling back to the page skew (default on) |
| `--[no-]match-color` | draw in the original ink's colour, sampled from the strokes; grey ink stays black (default on) |
| `--[no-]paper-fill` | fill erased boxes with paper sampled from around them (color, noise, specks) instead of white (default on) |
| `--[no-]keep-lines` | detect horizontal rules crossing a box and redraw them synthetically after erasing (default on) |
| `--[no-]handwriting` | replace handwritten text with handwriting, one face per document (default on) |
| `--format` | `same` (default, keeps each source's extension), `png`, `jpeg`, or `both` (one render written to `png/` and `jpg/`) |
| `--jpeg-quality N` | quality for JPEG output (default 95) |
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

Categories that get replacement text (aliases after the slash): `age`,
`birth_date`/`date_of_birth`, `birth_place`/`place_of_birth`,
`citizenship_nationality`/`citizenship`, `city`, `home_address`/`address`,
`home_phone`/`phone`, `marital_status`, `name`, `other_id`,
`religion`/`religious_affiliation`, `ssn`/`ssn_like`, `state`, `street`, `zip`. Together
they cover 96% of the annotated RVL-CDIP boxes.

Any other category (`sex_gender`, `signature`, `health_status`, `other`) is
erased and logged — the region is filled with generated paper and any rule redrawn, but
nothing is written back.

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
- **Ink colour** (`--match-color`): sampled from the darkest half of the pixels that are ink at all, so a
  stroke's antialiased edges and the paper don't wash the reading out. Blue ballpoint is replaced in blue,
  and a coloured rule is redrawn in its own colour. Near-grey ink reads as black, so a grayscale scan comes
  out byte for byte as it did before this existed. Detection (typeface, weight, baselines, rules) stays on
  luminance — hue should not decide which face matched.
- **Position**: glyph tops align with the original words' left/top; empty boxes are centered vertically.
- **Line count**: addresses stay two lines only if the original spans two lines.
- **Style** (`--match-style`): e.g. `4/12/50` → `9/3/61`, `Marfa, Tex.` → `Olney, Ark.`,
  `(555) 014-2233` → `(804) 311-9072`, `Springfield, PA 19064` → city line only, `MARRIED` → upper case.
  Only the format is copied, never the value. It does reveal the original's format and approximate
  width, which matters if that is sensitive for your use.

`check` flags: `no_ocr_words`, `looks_like_date`, `looks_like_ssn`, `no_digits`, `only_digits`
(likely label problems) and `style_not_detected` (informational; a random format will be used).

## Known limitations

**Light text on a dark background is not handled.** Everything here assumes ink is darker than paper:
`sample_ink` looks for pixels darker than the page, `layout.draw_text` composites with
`ImageChops.darker`, `augment.overlay` copies every non-white pixel, and the measurements
(`detect_rules`, `weight`, `baseline`, the typeface masks) all threshold for dark ink. Paper sampling
is the exception and already copes, reading `(20, 20, 20)` off a black page.

Given a white-on-black or cream-on-navy box the erase is correct -- the box is refilled in the right
dark tone -- but the replacement is drawn in the wrong polarity and comes out as dark glyphs ringed
with bright halos. `sample_ink` also mistakes the dark background for the ink and returns it.

The cheap fix, when it is wanted: detect polarity per box by comparing the sampled ink's luminance
with the paper's, and for a light-on-dark box invert the crop, run the existing pipeline unchanged,
and invert the result back. That reuses typeface matching, weight, baselines and rule detection as
they are rather than teaching six modules about polarity. The genuinely new parts are a
polarity-aware `sample_ink` (the pixels furthest from paper, in either direction) and compositing by
coverage instead of `darker`.

This has not mattered so far: RVL-CDIP is grayscale and black-on-white throughout. It would matter
for slides, advertisements and forms with inverted panels.

Also still open: `sex_gender`, `signature`, `health_status` and `other` boxes are erased but not
replaced (3.8% of annotated boxes; see the category list above), and on a preprinted form whose date
field reads `April 10 ___ 19 87`, a box drawn around the whole field spans the printed `19`, so the
replacement can come out as something no date would be written as.

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
