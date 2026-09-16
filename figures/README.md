# Example before/after figures

A curated set for papers and talks. **No image here shows anyone's personal data**, by construction,
and every one is reproducible from the commands below.

Two kinds, because neither alone covers everything:

## Stand-ins on real scans

The "before" is genuine scanner output — real paper, real typeface, real skew, real rule lines — but
the region being redacted is *not* personal data. It is a printed document date that happens to be
shaped like a birth date, chosen automatically by `scripts/paper_figures.py`, which requires that

* every word in it appears on at least 8 distinct pages (preprinted boilerplate, not something a
  person wrote), and it lies clear of every annotated PII box; and
* the crop itself overlaps no annotated PII box, so the figure cannot publish real data from
  elsewhere on the page.

Wider crops need more than that, because the annotations only mark the *subject's* PII. Show 300px of
a real page and you will pull in a signing officer, a physician, an agency contact — real people whom
no box covers. So two further screens run over everything inside the crop:

* **unannotated names**, by rarity: a capitalised word appearing on almost no other page is somebody's
  name, while a form's printed furniture appears everywhere. Honorifics, credentials and
  initial-plus-name patterns count too.
* **sensitive or re-identifying content**: a diagnosis is sensitive with or without a name, and a
  career history ("Senior Analyst, Regional Water Authority of Northfield, 1981–83") names nobody while
  narrowing the subject to a handful of people.

For the same reason `find` skips the personal-record classes by default (`resume`, `questionnaire`,
`handwritten`): on those, the context *is* the personal data. Invoices, letters, budgets and forms
carry institutional context that is safe to show.

Even so, every crop here was reviewed by eye. Two candidates were dropped at that stage that the
screens had passed, and one crop was refused by the name screen because a "Mr. …" addressee line
clipped its bottom edge.

**In the paper, say that these regions are stand-ins.** The pipeline is doing exactly what it does to
a real PII box, but the "before" text was never personal.

| files | shows | command |
|---|---|---|
| `train_invoice_aa0c9b64-*` | style matching, and that only the annotated box changes: the invoice number beside it is untouched | `render --page train_invoice_aa0c9b64 --box 187,146,157,13 --category birth_date --seed 0 --scale 3` |
| `train_letter_99afcf16-*` | a date replaced in place under a full letterhead | `render --page train_letter_99afcf16 --box 381,196,100,14 --category birth_date --seed 0 --scale 2 --pad 340,190,340,45` |
| `train_scientific_report_48b1820b-*` | a grant form: the replacement sits on the printed rule, in the form's own hand-typed face | `render --page train_scientific_report_48b1820b --box 156,534,88,13 --category birth_date --seed 0 --scale 2` |
| `train_memo_b86bd5c9-*` | a month and a day with the year elsewhere ("week of August 18"), inside a business memo | `render --page train_memo_b86bd5c9 --box 242,448,81,14 --category birth_date --seed 0 --scale 2` |
| `train_presentation_a5066670-*` | a heavily degraded campaign flyer: bold typewriter matched on a dark, blotchy scan | `render --page train_presentation_a5066670 --box 647,242,109,20 --category birth_date --seed 0 --scale 2` |
| `train_form_27556f9f-*` | **handwritten**: a date written into a records form's DATE DELIVERED box, replaced in a matched hand, dashes and all | `render --page train_form_27556f9f --box 594,804,123,34 --category birth_date --seed 0 --scale 2` |
| `train_advertisement_909e8e03-*` | **handwritten**: a coupon expiry written in slashes, replaced on the printed rule | `render --page train_advertisement_909e8e03 --box 559,306,79,25 --category birth_date --seed 0 --scale 2` |

The two handwritten ones need `find --handwritten`, and a word about why they are safe. Document
frequency cannot vouch for handwriting — nobody's handwriting is boilerplate — so those stand-ins are
held to a narrower rule instead: **dates only**, on institutional documents, where a handwritten date
is when the document was delivered, stamped or expires rather than anything about a person. A
handwritten *number* is never accepted, because it could be an account or case number that no
annotation covers. The crop screens all still apply, and these candidates especially want an eye on
them: of the six found, four were rejected here because the crop caught a cursive signature.

Between them the matcher picks Special Elite, Cutive Mono, Courier Prime and Liberation Mono, so the
set is not five examples of the same face. `find` reports the typeface it settled on for each
candidate (and `render` for each figure), which is how to pick a varied set rather than a uniform one.

One candidate was rendered and dropped for being a *poor* match rather than an unsafe one: an FDA
letterhead whose stand-in box held just "June 2," — two short words, too few glyphs for the matcher,
which answered with a serif against the original's bold sans. It is a fair picture of what happens on
a very short box, but it misrepresents typical output, so it is not in the set.

The last two came from classes the corpus barely annotates — scientific reports, memos, specifications —
which is where the institutional documents live. `--fetch <classes>` downloads a class that is not in
the cache yet, whole rather than page by page: the token counts are what separate boilerplate from
someone's data, and counts taken over a handful of pages would call everything rare.

`--margin auto` (the default) takes the widest context a page allows before touching annotated PII;
`--pad LEFT,TOP,RIGHT,BOTTOM` is for pages that are only safe on some sides, like the letterhead
above, where the institution's name sits above the date and the addressee just below it.

A third candidate was rendered and then dropped, for a reason worth knowing: on a preprinted voucher
reading `April 10 ___ 19 87`, the `19` is part of the *form*, not the value. The stand-in box spanned
it, so the replacement came out as `October 18 45  87` — not a date. Real `birth_date` boxes on such
forms have the same problem, so it is a limitation to be aware of rather than only a bad figure.

```sh
python scripts/paper_figures.py render --annotations <annotations.json> --cache <cache> -o figures/ ...
python scripts/paper_figures.py find --annotations <annotations.json> --cache <cache> -o figures/  # more candidates
```

## A synthetic page

Stand-ins cannot show handwriting (handwriting is never boilerplate, so it never passes the
document-frequency test) or a filled-in name or address field. So `synthetic/` is a form built from
nothing: every value on it comes from the generators, and it is degraded to look scanned. It carries
its own Textract-style OCR, so the real pipeline runs on it unchanged.

Three variants, each a different office: what the form was printed with, what it was filled in with,
and whose hand signed it. Every one is a `*_before.png` / `*_after.png` pair, plus a crop of the
handwritten block and the annotations for rerunning.

| variant | printed in | filled in with | signed in |
|---|---|---|---|
| `synthetic_typewriter_*` | Liberation Sans | Courier Prime | Caveat |
| `synthetic_serif_*` | TeX Gyre Schola | Liberation Serif | Homemade Apple |
| `synthetic_elite_*` | Carlito | Special Elite | Architects Daughter |

```sh
python scripts/synthetic_page.py -o figures/synthetic --seed 0            # all three
python scripts/synthetic_page.py -o figures/synthetic --variants serif    # just one
```

Worth pointing out in a caption: the employee's surname is shared by the handwritten spouse field
(name linking), the ZIP matches its state, and the phone obeys the NANP.
