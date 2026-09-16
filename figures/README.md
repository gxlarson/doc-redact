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

Each crop here was also reviewed by eye, since those checks only know about boxes that *were*
annotated. One candidate was dropped at that stage: its crop carried CV details (research topic,
society, dates) that could identify someone indirectly even with no name in frame.

**In the paper, say that these regions are stand-ins.** The pipeline is doing exactly what it does to
a real PII box, but the "before" text was never personal.

| files | shows | command |
|---|---|---|
| `test_invoice_09998110-*` | style matching: a date replaced in its own format and typeface | `render --page test_invoice_09998110-857a --box 356,168,111,14 --category birth_date --seed 0` |
| `test_invoice_e64e7d01-*` | the rule under the text survives the erase | `render --page test_invoice_e64e7d01 --box 479,212,96,15 --category birth_date --seed 3` |
| `train_resume_72209ce5-*` | linking: the same value in two boxes gets one replacement | `render --page train_resume_72209ce5 --box 114,794,73,15 --box 18,765,72,14 --category birth_date --seed 0 --margin 24` |

```sh
python scripts/paper_figures.py render --annotations <annotations.json> --cache <cache> -o figures/ ...
python scripts/paper_figures.py find --annotations <annotations.json> --cache <cache> -o figures/  # more candidates
```

## A synthetic page

Stand-ins cannot show handwriting (handwriting is never boilerplate, so it never passes the
document-frequency test) or a filled-in name or address field. So `synthetic/` is a form built from
nothing: every value on it comes from the generators, and it is degraded to look scanned. It carries
its own Textract-style OCR, so the real pipeline runs on it unchanged.

| file | shows |
|---|---|
| `synthetic/synthetic_before.png` | the invented "original" page |
| `synthetic/synthetic_after.png` | the same page redacted |
| `synthetic/synthetic_handwriting_{before,after}.png` | the handwritten block: handwriting replaced by handwriting, one face per document |
| `synthetic/synthetic_annotations.json` | the boxes, for rerunning |

```sh
python scripts/synthetic_page.py -o figures/synthetic --seed 0
```

Worth pointing out in a caption: the employee's surname is shared by the handwritten spouse field
(name linking), the ZIP matches its state, and the phone obeys the NANP.
