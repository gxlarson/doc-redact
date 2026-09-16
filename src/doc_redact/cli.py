"""Command-line entrypoint: ``doc-redact`` or ``python -m doc_redact``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from doc_redact import checks, fakedata, pipeline, render
from doc_redact.annotations import Document, load_documents, parse_labelstudio, save_documents
from doc_redact.datasets import DATASETS
from doc_redact.remote import Fetcher


def _override(value: str) -> tuple[str, str]:
    category, sep, text = value.partition("=")
    if not sep or not category:
        raise argparse.ArgumentTypeError(f"expected CATEGORY=TEXT, got {value!r}")
    return category, text.replace("\\n", "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-redact", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    documents = argparse.ArgumentParser(add_help=False)
    documents.add_argument("annotations", type=Path, help="parsed annotations JSON")
    documents.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    documents.add_argument("--only", action="append", metavar="ID", help="process only this document id (repeatable)")
    documents.add_argument("--cache", type=Path, help="cache downloaded images/OCR in this directory")

    r = sub.add_parser("redact", parents=[documents], help="white out or pseudonymize annotated PII regions")
    r.add_argument("--mode", choices=[m.value for m in pipeline.Mode], default="pseudo")
    r.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    r.add_argument(
        "--format", dest="image_format", choices=["same", "png", "jpeg"], default="same",
        help="output image format; 'same' (default) keeps each source's extension. png avoids re-encoding "
             "artifacts on pages that were already JPEG, at a few times the file size",
    )
    r.add_argument(
        "--jpeg-quality", type=int, default=95, metavar="N",
        help="quality for JPEG output (default 95); lower values add compression artifacts across the whole "
             "page, not only the replaced regions",
    )
    r.add_argument(
        "--font",
        help="use this font for every box (path, bundled font, or system font name); disables --match-font",
    )
    r.add_argument(
        "--match-font", action=argparse.BooleanOptionalAction, default=True,
        help="pick, per box, the bundled typeface that best matches the scanned words",
    )
    r.add_argument(
        "--match-skew", action=argparse.BooleanOptionalAction, default=True,
        help="tilt replacement text to follow the slope of the scanned line",
    )
    r.add_argument(
        "--paper-fill", action=argparse.BooleanOptionalAction, default=True,
        help="erase boxes with generated paper tone and specks sampled around them, instead of pure white",
    )
    r.add_argument(
        "--keep-lines", action=argparse.BooleanOptionalAction, default=True,
        help="redraw horizontal form lines and underlines that run through erased boxes",
    )
    r.add_argument(
        "--handwriting", action=argparse.BooleanOptionalAction, default=True,
        help="replace text Textract marked as handwritten with jittered handwriting fonts",
    )
    r.add_argument(
        "--system-fonts", action="store_true",
        help="with --match-font, also consider installed fonts such as Times New Roman and Courier New",
    )
    r.add_argument("--font-size", type=int, help="fixed font size; disables --auto-font-size")
    r.add_argument(
        "--auto-font-size", action=argparse.BooleanOptionalAction, default=True,
        help="size text to match the OCR words under/beside each box",
    )
    r.add_argument(
        "--fit-width", action=argparse.BooleanOptionalAction, default=True,
        help="pick replacements close to the original text width and shrink to fit the box",
    )
    r.add_argument(
        "--match-style", action=argparse.BooleanOptionalAction, default=True,
        help="mimic the original format: date layout, state abbreviation, digit separators, casing",
    )
    r.add_argument(
        "--match-weight", action=argparse.BooleanOptionalAction, default=True,
        help="thicken strokes to match the scanned text under each box",
    )
    r.add_argument(
        "--text", action="append", type=_override, default=[], metavar="CATEGORY=TEXT",
        help='use fixed replacement text for a category, e.g. --text birth_date="Oct.8,14" ("\\n" for newline)',
    )
    r.add_argument("--seed", type=int, help="seed fake data and augmentation for reproducible output")
    r.add_argument("--keep-unannotated", action="store_true", help="also save pages with no PII annotations")

    c = sub.add_parser("check", parents=[documents], help="show the OCR text under each box and flag likely label errors")
    c.add_argument("--problems-only", action="store_true", help="only list boxes with problem flags")
    c.add_argument("--json", type=Path, dest="json_out", help="also write the full report to this file")

    p = sub.add_parser("parse-labelstudio", help="convert a Label Studio export to the annotations format")
    p.add_argument("export", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--base-url", required=True,
        help="URL prefix where the images are hosted, e.g. https://cumberland.isis.vanderbilt.edu/stefan/FUNSD/",
    )
    return parser


def _select_documents(args: argparse.Namespace) -> list[Document] | None:
    docs = load_documents(args.annotations)
    if not args.only:
        return docs
    wanted = set(args.only)
    docs = [d for d in docs if d.stem in wanted]
    missing = wanted - {d.stem for d in docs}
    if missing:
        logging.error("ids not found in %s: %s", args.annotations, ", ".join(sorted(missing)))
        return None
    return docs


def cmd_redact(args: argparse.Namespace) -> int:
    dataset = DATASETS[args.dataset]
    docs = _select_documents(args)
    if docs is None:
        return 2

    if args.seed is not None:
        import numpy as np

        fakedata.seed(args.seed)
        np.random.seed(args.seed)

    options = pipeline.Options(
        mode=pipeline.Mode(args.mode),
        text=render.TextSettings(
            font=args.font or dataset.font,
            font_size=args.font_size or dataset.font_size,
            auto_font_size=args.auto_font_size and args.font_size is None,
            fit_width=args.fit_width,
            match_style=args.match_style,
            match_weight=args.match_weight,
            match_font=args.match_font and args.font is None,
            match_skew=args.match_skew,
            handwriting=args.handwriting,
            paper_fill=args.paper_fill,
            keep_lines=args.keep_lines,
            system_fonts=args.system_fonts,
            overrides=dict(args.text),
        ),
    )
    failures = pipeline.run(
        docs, dataset, Fetcher(args.cache), args.out, options, keep_unannotated=args.keep_unannotated,
        image_format=args.image_format, jpeg_quality=args.jpeg_quality,
    )
    if failures:
        logging.error("%d document(s) failed: %s", len(failures), ", ".join(d.stem for d, _ in failures))
        return 1
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    dataset = DATASETS[args.dataset]
    docs = _select_documents(args)
    if docs is None:
        return 2

    report, failed, problem_boxes, total_boxes = [], 0, 0, 0
    for doc, result in pipeline.check(docs, dataset, Fetcher(args.cache)):
        if isinstance(result, Exception):
            failed += 1
            logging.error("failed on %s: %r", doc.stem, result)
            report.append({"id": doc.stem, "error": repr(result)})
            continue
        report.append({"id": doc.stem, "boxes": [b.to_dict() for b in result]})
        total_boxes += len(result)
        shown = [b for b in result if not args.problems_only or checks.PROBLEM_FLAGS & set(b.flags)]
        problem_boxes += sum(1 for b in result if checks.PROBLEM_FLAGS & set(b.flags))
        if not shown:
            continue
        print(doc.stem)
        for box in shown:
            flags = f"  [{', '.join(box.flags)}]" if box.flags else ""
            text = box.text.replace("\n", " | ") or "-"
            print(f"  #{box.index:<2} {box.category:<24} {text}{flags}")

    print(f"\n{total_boxes} boxes checked, {problem_boxes} with problem flags, {failed} document(s) failed", file=sys.stderr)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 1 if failed else 0


def cmd_parse_labelstudio(args: argparse.Namespace) -> int:
    with open(args.export, "r", encoding="utf-8") as f:
        export = json.load(f)
    docs = parse_labelstudio(export, args.base_url)
    save_documents(docs, args.output)
    logging.info("wrote %d documents to %s", len(docs), args.output)
    return 0


COMMANDS = {"redact": cmd_redact, "check": cmd_check, "parse-labelstudio": cmd_parse_labelstudio}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.verbose:
        logging.getLogger("doc_redact").setLevel(logging.DEBUG)
    return COMMANDS[args.command](args)
