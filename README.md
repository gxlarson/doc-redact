# doc-redact

Redact PII regions in scanned document images, either by whiting them out or by
replacing them with realistic fake text that is degraded to look printed.

Supports RVL-CDIP, FUNSD, Tobacco3482, Tobacco800 and any other set of page images
with AWS Textract OCR. The images and OCR are read from disk or downloaded from a web
server, depending on each annotation's `url`.

![Three documents before and after synthetic replacement: a typed memo, a
handwritten coupon expiry, and a degraded flyer](figures/redaction_pairs.png)

Each replacement is matched to the page it is written into: the typeface is
chosen per region from the words around it, the stroke weight and ink color are
measured from them, the text follows the scanned line's baseline and skew,
handwriting is replaced by handwriting, and any ruled line crossing the region is
redrawn. The replacement copies the format of the original but not its value.

## Install

```sh
pip install -e ".[dev]"
```

## Quick start

The tool can be tried without downloading any data: `synthetic_page.py` builds a
personnel file from the generators, with its own Textract-style OCR, and runs
the real pipeline over it:

```sh
python scripts/synthetic_page.py -o demo --variants serif --seed 0
```

That writes `demo/synthetic_serif_before.png` and `_after.png`. Every value on
the page is generated, so nothing on it belongs to anyone, and one page shows
most of what the tool does:

| in the before | in the after | because |
|---|---|---|
| `Sarah Chang` in the name field, and again in the signature | `Ryan Taylor` in both | one identity per person, across print and handwriting |
| `Jennifer Chang` as spouse | `David Taylor` | people sharing a surname keep a shared surname |
| `Sand Rock, MN 56617` | `Cotton Center, KY 42457` | the ZIP is drawn from its state's own range |
| `(345) 840-3350` | `(358) 613-0749` | the number obeys the North American numbering plan |
| `289-18-1554` | `713-58-4364` | the SSN keeps the shape the SSA issues |
| `12-27-1935`, handwritten | `3-10-1943`, handwritten | handwriting is replaced by handwriting, format kept |

Pass `--variants` to get the other three pages (`typewriter`, `elite`, `color`);
the color one exists because RVL-CDIP is grayscale and cannot show colored ink
being replaced in its own color.

The values above are what `--seed 0` produces, so the run should reproduce them.
The pixels will differ, because the seed determines the generated data but not the
paper texture and scanner wear drawn over it.

The script calls the pipeline directly from Python, whereas the `doc-redact` command
reads a page, its OCR and its annotations from files; the command is described under
**Usage**, with an example input in `examples/`.

## Usage

Every page to redact needs three things:

| | content | format |
|---|---|---|
| page image | the scan | any format that Pillow opens: JPEG, PNG, TIFF, ... |
| OCR | every word on the page and its position | AWS Textract JSON, one file per page |
| annotations | the boxes to replace and the category of each | one entry in an annotation file |

The annotation file is the only input named on the command line. It lists the
pages, and the `url` of each entry gives the location of the page image; the OCR
file is located from the same `url`. A `url` can be a web address, from which the
file is downloaded, or a path, from which it is read, so the same command works on
a folder of scans and on a corpus stored on a web server.

The folder `examples/` contains one complete input, built from generated values so
that none of it belongs to a real person:

```
examples/
  annotations.json          the boxes to replace; its url is "page.png"
  annotations_hosted.json   the same boxes, with the page's url on a web server
  page.png                  the page image
  page.json                 its OCR, in Textract's format
  redact_page.py            the same page redacted from Python (see Python interface)
```

### Annotation file

The annotation file is a JSON list with one entry per page:

```json
[
  {"url": "page.png",
   "annotations": [
     {"category": "name",       "bbox": {"left": 330, "top": 138, "width": 121, "height": 26}},
     {"category": "birth_date", "bbox": {"left": 330, "top": 242, "width": 154, "height": 26}}
   ]}
]
```

* `url` is the location of the page image: `https://...`, an absolute path, a
  `file://` URL, or a path relative to the folder of the annotation file. The output
  file has the same file name.
* `bbox` is in pixels, with the origin at the top-left corner of the image. Label
  Studio boxes in percent of the image size (`{"x", "y", "width", "height"}`) are also
  accepted and converted to pixels once the image size is known.
* `category` determines the kind of replacement text (see **Categories** below).
* Entries with the same `url` are merged into one page, so a page annotated in
  several passes is redacted once, with the boxes from every pass.

`doc-redact parse-labelstudio` converts a Label Studio export into this format.

### OCR file

The OCR file is the JSON response of the AWS Textract `DetectDocumentText` operation
for the page, saved unchanged. The tool reads only the `WORD` blocks and only the
following fields:

```json
{"Blocks": [
  {"BlockType": "WORD", "Text": "Sarah", "TextType": "PRINTED",
   "Geometry": {"BoundingBox": {"Left": 0.3667, "Top": 0.1683, "Width": 0.0611, "Height": 0.0317}}}
]}
```

`Left`, `Top`, `Width` and `Height` are fractions of the page width and height.
`TextType` is optional, and a word marked `HANDWRITING` is replaced in a handwriting
font. Other block types and fields are ignored, so OCR from another engine can be
used after conversion to this structure.

The following code produces an OCR file with Textract:

```python
import json, boto3
response = boto3.client("textract").detect_document_text(Document={"Bytes": open("page.png", "rb").read()})
json.dump(response, open("page.json", "w"))
```

The replacement for each box is matched to the OCR words: the words under the box
determine its format, typeface, size, weight, baseline, skew and handwriting, and the
words on the rest of the page are used to link repeated names and values. With
`--no-ocr` the tool still erases every box and fills it with paper texture, rule lines
and generated text, but it draws the text in the fallback font and size and does not
match it to the page; the `check` command requires OCR.

### Local files

In this setup the tool reads every file from disk, and when each OCR file is saved
next to its image, with the same name and the extension `.json`, the command needs
only the annotation file:

```sh
doc-redact redact examples/annotations.json -o output/example --seed 0
```

This command writes `output/example/page.png`. When the images and the OCR files are
in separate folders, `--images` and `--ocr` name the two folders, and the `url` of
each entry then needs to contain only the file name:

```sh
doc-redact redact annotations.json --images scans/ --ocr textract/ -o output/redacted
```

The same options read a downloaded copy of a corpus whose annotation file contains the
URLs of the server: the files are looked up by the names that the server uses. Either
option can be used alone, to read only the images or only the OCR from disk.

### Files on a web server

In this setup each `url` is a web address, and the image and OCR of each page are
downloaded when the page is processed. By default the OCR file is expected at the URL
of the image with `.json` in place of the extension (`.../page.png` and
`.../page.json`). For corpora that store their OCR elsewhere, `--dataset` selects a
named rule:

| `--dataset` | Image | OCR | Fallback font |
|---|---|---|---|
| `generic` (default) | `url` | `url` with `.json` | Liberation Serif 24 |
| `funsd` | `url` with `.png` | `FUNSD_ocr/<id>.json` beside `FUNSD/` | Inconsolata 14 |
| `tobacco3482` | `url` | `url` with `.json` | Liberation Serif 34 |
| `tobacco800` | `url` | `url` with `.json` | Courier Prime 21 |
| `rvlcdip` | `url`, adding `.jpg` if it does not have an extension | `rvlcdip_ocr/<id>.json` beside `rvlcdip/` | Inconsolata 13 |

The fallback font and size are used only when a box does not have measurable OCR
words nearby. `--cache DIR` saves every download under the same paths as on the server, so
later runs read the files from disk.

The following commands test this setup with a local web server that serves
`examples/`:

```sh
python -m http.server 8000 --directory examples &
doc-redact redact examples/annotations_hosted.json -o output/hosted --seed 0 --cache .cache
```

With the same `--seed`, the two setups write the same replacement text.

### Python interface

`pipeline.redact_document` redacts one page that is already in memory, with the boxes
given as Python objects, so it does not need an annotation file or a URL. The script
`examples/redact_page.py` uses it on the example page, step by step:

```sh
python examples/redact_page.py
```

The script writes `examples/output/page_white.png` (white boxes),
`examples/output/page_pseudo.png` (fake text) and `examples/output/comparison.png`, which
shows the original and the two redactions side by side with the replaced boxes outlined.
The core of the script is the following:

```python
import json
from PIL import Image
from doc_redact import fakedata, pipeline, render
from doc_redact.annotations import Annotation, BBox, Document

page = Image.open("examples/page.png").convert("RGB")
ocr = json.load(open("examples/page.json"))
doc = Document(url="page.png", annotations=[
    Annotation("name", BBox(left=330, top=138, width=121, height=26)),
    Annotation("ssn", BBox(left=330, top=190, width=121, height=26)),
])
options = pipeline.Options(pipeline.Mode.PSEUDO,
                           render.TextSettings(font="LiberationSerif-Regular.ttf", font_size=24))
fakedata.seed(0)
pipeline.redact_document(page, doc, ocr, options).save("redacted.png")
```

`TextSettings` has a field for each `--[no-]` option of the command line, and its
`font` and `font_size` are the fallback used when a box does not have measurable OCR
words nearby. Without OCR, pass `{"Blocks": []}` in place of the OCR.

### Released data

The annotations used for the paper are not released, because they record the location
of every sensitive entity on the page and the original pages are public, so the
coordinates would amount to a list of the values that they locate. The released
data is the corpus with those values already replaced:

| Dataset | Contents |
|---|---|
| [`stefan-hf/rvlcdip-redact`](https://huggingface.co/datasets/stefan-hf/rvlcdip-redact) | the de-identified RVL-CDIP corpus, with and without identification codes, as page images and OCR |
| [`stefan-hf/rvlcdip-label-fixes`](https://huggingface.co/datasets/stefan-hf/rvlcdip-label-fixes) | the label review of all 400,000 documents |
| [`stefan-hf/rvlcdip-duplicates`](https://huggingface.co/datasets/stefan-hf/rvlcdip-duplicates) | test and validation documents with a near-duplicate in training |
| [`stefan-hf/rvlcdip-id-codes`](https://huggingface.co/datasets/stefan-hf/rvlcdip-id-codes) | annotated and detected litigation identification codes |
| [`stefan-hf/rvlcdip-n`](https://huggingface.co/datasets/stefan-hf/rvlcdip-n) · [`-o`](https://huggingface.co/datasets/stefan-hf/rvlcdip-o) | the two out-of-distribution evaluation sets |

### Further examples

In the following examples, `annotations/redactions_*.json` stands for an annotation
file of your own, for a corpus stored as the named `--dataset` rules expect.

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

# Old behavior: fixed font size, random formats, no width fitting
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

## Full-corpus runs

The default settings are intended for a full run: pseudonymized text, matched typeface,
size, weight, skew and format, paper fill and preserved rules. A full run needs only
`-o`, `--seed` and `--cache`.

```sh
doc-redact redact rvlcdip_pii_annotated.json \
    --dataset rvlcdip -o output/rvlcdip_pseudo --seed 0 --cache .cache
```

Most of the running time is spent on typeface matching, which runs once for each box:
about 1.6 s per box with `--match-font` and 0.3 s without it (measured on RVL-CDIP scans
of roughly 1080p resolution; handwritten pages take roughly twice as long, because a
second matcher compares the handwriting fonts). For the 15,956 annotated RVL-CDIP pages
(39,435 boxes), the total is roughly 19 core-hours, or about 4.5 with `--no-match-font`.

Each page requires a download of about 780 KB (image and Textract JSON), so a full
RVL-CDIP run downloads about 12 GB. With `--cache` the data is downloaded only once,
and before a parallel run the following command downloads the corpus in advance, so
that the workers do not send simultaneous requests to the server:

```sh
doc-redact check rvlcdip_pii_annotated.json --dataset rvlcdip --cache .cache >/dev/null
```

### Parallel runs

Pages are processed independently, so the annotation file can be split into shards
and processed with one process per core. Output file names are derived from the URL of
each document, so all shards can write to the same `-o` directory.

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

Each shard needs a different `--seed`, because shards with the same seed generate the
same people, addresses and numbers for different pages. The running time decreases
almost linearly with the number of cores (the work is CPU-bound in numpy and scipy),
so the run above takes roughly 1.5 h on 12 cores and about 35 min on 32. Several
processes can read a shared `--cache` directory at the same time. The directory should
be filled in advance as shown above, so that the 12 workers do not each download the
same corpus.

A page that fails is logged and skipped, and a process exits with a nonzero status if
any of its pages failed. The exit codes therefore show whether a run was complete, and
a run without error messages may still be incomplete.

### Output files

The command writes one file per page directly into `--out`, which is created if it
does not exist. Each file is named after its source: RVL-CDIP and Tobacco pages keep
their `.jpg` names, and FUNSD pages their `.png` names. Grayscale scans are saved in
grayscale, although pages are edited in RGB.

JPEG sources are re-encoded, which changes pixels across the whole page and not only
in the replaced regions. The following table shows the effect on RVL-CDIP scans,
measured against the original pixels outside the annotated boxes:

| output | size/page | drift outside the replaced regions |
|---|---|---|
| `--format same` (JPEG, quality 95) | ~180 KB | 0.31 gray levels |
| `--format png` | ~190 KB | 0.003 (lossless) |
| `--format same --jpeg-quality 60` | ~86 KB | 1.7 gray levels |

`--format png` is preferable when the redacted pages will be compared with the
unmodified originals or used to train or evaluate a model, because re-encoding
artifacts across the page are a difference between the two sets that is unrelated to
redaction.

`--format both` writes each page twice from the same render, as a lossless PNG under
`png/` and as a JPEG under `jpg/`, with the same pixels and the same file name. The
page is fetched, matched and drawn once, so the second copy adds only its encoding
time (a few percent of the run), and the two copies contain the same replacements.
The option produces an archival copy and a smaller copy for distribution without a
second redaction, which would generate a different set of fake people.

The `redact` command has the following options:

| Option | Meaning |
|---|---|
| `--dataset` | `generic` (default), `funsd`, `tobacco3482`, `tobacco800`, `rvlcdip`; where each page's image and OCR are found, and the fallback font |
| `--images DIR` | read page images from this folder, by the file name in each `url` |
| `--ocr DIR` | read Textract JSON from this folder, by the file name in each `url` |
| `--no-ocr` | redact without OCR: boxes are erased and refilled in the fallback font, with no matching |
| `--mode` | `pseudo` (default) or `white` |
| `--[no-]match-font` | pick, per box, the bundled typeface that best matches the scanned words (default on) |
| `--system-fonts` | also consider installed fonts (Times New Roman, Courier New, Georgia, ...); results then depend on the machine |
| `--font` | use this one font everywhere (path, bundled file name, or system font name); turns off `--match-font` |
| `--font-size N` | fixed size (turns off auto sizing); also the fallback when auto sizing does not find words to measure |
| `--[no-]auto-font-size` | size text so the OCR words under the box, re-rendered in `--font`, span their original width (default on) |
| `--[no-]fit-width` | pick the candidate closest to the original text width, and shrink if it would overflow the box (default on) |
| `--[no-]match-weight` | thicken strokes to match the scanned text under the box (default on) |
| `--[no-]match-style` | mimic the original's format: date layout, state form, digit separators, address part, casing, trailing punctuation (default on) |
| `--[no-]match-skew` | rotate the replacement to the scanned line's baseline slope, falling back to the page skew (default on) |
| `--[no-]match-color` | draw in the original ink's color, sampled from the strokes; gray ink stays black (default on) |
| `--degrade` | `auto` (default), `always` or `never`: add scanner wear to replacements only on pages whose own text is worn |
| `--[no-]paper-fill` | fill erased boxes with paper sampled from around them (color, noise, specks); `--no-paper-fill` fills them with white (default on) |
| `--[no-]keep-lines` | detect horizontal rules crossing a box and redraw them synthetically after erasing (default on) |
| `--[no-]handwriting` | replace handwritten text with handwriting, one face per document (default on) |
| `--format` | `same` (default, keeps each source's extension), `png`, `jpeg`, or `both` (one render written to `png/` and `jpg/`) |
| `--jpeg-quality N` | quality for JPEG output (default 95) |
| `--only ID` | process only these documents (repeatable) |
| `--text CATEGORY=TEXT` | fixed replacement text for a category |
| `--seed N` | reproducible fake data. The seed determines the generated values but not the paper texture and scanner wear, so two runs with the same seed produce the same text on visibly different pixels |
| `--cache DIR` | keep downloaded images/OCR on disk (local files are read in place, never cached) |
| `--keep-unannotated` | also save pages without annotations |

Failed pages are logged and skipped, and the command exits with a nonzero status if
any page failed.

## Categories

The following categories receive replacement text (aliases after the slash): `age`,
`birth_date`/`date_of_birth`, `birth_place`/`place_of_birth`,
`citizenship_nationality`/`citizenship`, `city`, `home_address`/`address`,
`home_phone`/`phone`, `marital_status`, `name`, `other_id`,
`religion`/`religious_affiliation`, `ssn`/`ssn_like`, `state`, `street`, `zip`. Together
they cover 96% of the annotated RVL-CDIP boxes.

Boxes of any other category (`sex_gender`, `signature`, `health_status`, `other`) are
erased and logged: the region is filled with generated paper and any rule is redrawn,
but text is not written into it.

## Text placement

The tool groups the OCR words that lie at least 70% inside each box into lines, and
uses this text to determine the following properties:

- **Typeface** (`--match-font`): up to three words in the box (or, if none are usable, the nearest words on
  its line) are rendered in every candidate font at the scanned width. Scan and renderings are cleaned of
  specks, reduced to 1-px skeletons and compared by chamfer distance; words vote by rank.

- **Font size**: estimated from word widths (Textract pads boxes vertically, so heights overestimate the
  size). Without usable words in the box, the size is estimated from words in the same row, and after
  that taken from the dataset default.
- **Weight**: stroke thickness is measured on the scanned words (ink area / half the boundary length) and on the
  same words re-rendered in `--font`; text is drawn at 4x with the outline that best matches, then downsampled,
  so weight changes in 0.5 px steps. Fonts heavier than the scan are not thinned.
- **Ink color** (`--match-color`): sampled from the darker half of the ink pixels, so that the antialiased
  edges of a stroke and the paper do not lighten the measurement. Blue ballpoint is replaced in blue, and
  a colored rule is redrawn in its own color. Near-gray ink is treated as black, so the output for a
  grayscale scan is identical, byte for byte, to the output before color matching was added. Detection
  (typeface, weight, baselines, rules) uses luminance only, so that hue does not affect the choice of
  typeface.
- **Scanner wear** (`--degrade auto`): replacements receive low-ink dropouts only when the text of the
  page has them. The tool measures the midtones around the strokes of a page, per unit of stroke
  perimeter so that the measure does not change with font size: a born-digital glyph has an antialiased
  edge one pixel wide, and a scanned glyph has a blurred edge. On 40 RVL-CDIP scans and 15 clean
  renderings the two ranges did not overlap (scans 0.19-0.45, renderings 0.07-0.15), and the threshold is
  set between them at 0.17. Without this test, the replacement on a born-digital PDF would be the only
  worn text on the page.
- **Position**: glyph tops align with the original words' left/top; empty boxes are centered vertically.
- **Line count**: addresses stay two lines only if the original spans two lines.
- **Style** (`--match-style`): e.g. `4/12/50` → `9/3/61`, `Marfa, Tex.` → `Olney, Ark.`,
  `(555) 014-2233` → `(804) 311-9072`, `Springfield, PA 19064` → city line only, `MARRIED` → upper case.
  Only the format of the original is copied, and its value is not. The replacement does reveal the
  format and approximate width of the original, which matters if these are sensitive in your
  application.

The `check` command reports the flags `no_ocr_words`, `looks_like_date`, `looks_like_ssn`, `no_digits`
and `only_digits` (likely label problems) and `style_not_detected` (informational; a random format is
then used).

## Known limitations

The tool does not handle light text on a dark background, because the code assumes that ink is
darker than paper: `sample_ink` looks for pixels darker than the page, `layout.draw_text` composites
with `ImageChops.darker`, `augment.overlay` copies every non-white pixel, and the measurements
(`detect_rules`, `weight`, `baseline`, the typeface masks) all use thresholds for dark ink. Paper
sampling is the exception, and it reads `(20, 20, 20)` correctly from a black page.

For a white-on-black or cream-on-navy box, the erase is correct and the box is refilled in the right
dark tone, but the replacement is drawn in the wrong polarity, as dark glyphs surrounded by bright
halos. `sample_ink` also returns the dark background as the ink color.

A simple fix would detect the polarity of each box by comparing the luminance of the sampled ink with
that of the paper, and for a light-on-dark box invert the crop, run the existing pipeline unchanged
and invert the result. This would reuse typeface matching, weight, baselines and rule detection
without changes to the six modules that assume dark ink. The new parts would be a version of
`sample_ink` that handles both polarities (the pixels furthest from the paper color, in either
direction) and compositing by coverage in place of `darker`.

The limitation does not affect RVL-CDIP, whose pages are grayscale and black on white, but it would
affect slides, advertisements and forms with inverted panels.

Boxes of the categories `sex_gender`, `signature`, `health_status` and `other` are erased
but not replaced (3.8% of annotated boxes; see **Categories**). On a
preprinted form whose date field reads `April 10 ___ 19 87`, a box drawn around the whole field
includes the printed `19`, so the replacement can be a string that is not a valid date.

## Fonts

The package includes 19 open-license fonts (5.2 MB), so that typeface matching gives the same result
on every machine:

| Kind | Bundled | Substitute for |
|---|---|---|
| Serif | Liberation Serif (+ Bold), Gelasio, TeX Gyre Pagella / Schola / Bonum | Times, Georgia, Palatino, Century, Bookman |
| Sans | Liberation Sans (+ Bold), Carlito, DejaVu Sans | Arial, Calibri, Verdana |
| Typewriter, serifed | Courier Prime (+ Bold), Courier 10 Pitch (+ Bold), Cutive Mono, Special Elite | Courier and worn typewriter faces |
| Typewriter, sans | Liberation Mono, Inconsolata, DejaVu Sans Mono | Consolas, Lucida Console |

The licenses (SIL OFL, Apache 2.0, GUST, Bitstream, DejaVu) are listed in
`src/doc_redact/fonts/FONTS.txt` and included next to the fonts.

## Layout

```
src/doc_redact/
  cli.py          argument parsing, subcommands
  pipeline.py     per-document and batch redaction
  datasets.py     URL/output/font conventions per dataset
  remote.py       local files read in place; downloads with optional disk cache
  annotations.py  BBox/Document types, JSON I/O, Label Studio parser
  categories.py   annotation category names and their aliases
  ocr.py          Textract WORD blocks -> pixel boxes
  geometry.py     word coverage and line-height heuristics
  checks.py       annotation audit rules
  render.py       white-out, replacement planning and drawing
  background.py   erasing a box: generated paper texture and redrawn form lines
  baseline.py     baselines and line skew, measured from pixels
  layout.py       font loading, size estimation, width fitting, supersampled drawing
  weight.py       stroke thickness measurement and matching
  typeface.py     candidate fonts and per-box typeface matching
  handwriting.py  drawing in a handwriting font, with per-letter variation
  fonts/          bundled open-license fonts (see fonts/FONTS.txt)
  style.py        format detection (dates, places, digits, addresses, casing)
  augment.py      scan-like degradation and compositing
  fakedata.py     replacement text generators
  people.py       one fake identity per real person within a document
  addresses.py    one fake address per real address within a document
  values.py       one fake value per real value, for SSNs, phones, dates
  resources.py    bundled word lists and US state names
```

## Tests

```sh
pytest
```

## Citation

If you use this tool or its data, please cite:

```bibtex
@article{larson2026data,
  title   = {Data Quality in {RVL-CDIP}: A Systematic Audit},
  author  = {Larson, Stefan and Leach, Kevin},
  journal = {arXiv preprint arXiv:2610.32126},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.32126}
}
```
