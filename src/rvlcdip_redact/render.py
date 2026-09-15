"""Apply a redaction to one annotated region of a page image."""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from rvlcdip_redact import (
    addresses, augment, background, baseline, fakedata, geometry, handwriting, layout, ocr, people, style, typeface, weight,
)
from rvlcdip_redact.annotations import Annotation, BBox
from rvlcdip_redact.categories import canonical
from rvlcdip_redact.ocr import Word

log = logging.getLogger(__name__)

TEXT_INSET = 2  # px from the box's left edge when there are no OCR words to align with
MULTILINE_CATEGORIES = {"home_address"}


@dataclass
class TextSettings:
    font: str  # fallback, or the only font when match_font is off
    font_size: int  # used when auto sizing is off or finds no reference words
    auto_font_size: bool = True
    fit_width: bool = True
    match_style: bool = True
    match_weight: bool = True
    match_font: bool = True
    match_skew: bool = True  # tilt replacements to follow the scanned line
    paper_fill: bool = True  # erase with generated paper tone and specks instead of pure white
    keep_lines: bool = True  # redraw horizontal form lines that run through erased boxes
    handwriting: bool = True  # handwritten boxes get handwriting fonts with per-letter jitter
    handwriting_font: str = "PatrickHand-Regular.ttf"  # when no handwriting typeface can be matched
    system_fonts: bool = False  # also consider installed fonts (not bundled; results vary by machine)
    overrides: dict[str, str] = field(default_factory=dict)


def white_out(image: Image.Image, bbox: BBox) -> Image.Image:
    image = image.copy()
    # PIL rectangles include both corners, matching the original cv2.rectangle fill.
    ImageDraw.Draw(image).rectangle(
        [round(bbox.left), round(bbox.top), round(bbox.right), round(bbox.bottom)], fill="white"
    )
    return image


@dataclass
class Placement:
    text: str
    font: str
    font_size: int
    xy: tuple[float, float]
    spacing: float
    original: str
    stroke: int = 0
    angle_deg: float = 0.0
    baseline_y: float | None = None  # y of the first line's baseline at the text's left edge
    handwritten: bool = False


@dataclass
class Page:
    words: list[Word]
    gray: np.ndarray | None = None  # original page, for measuring weight, typeface and baselines
    typefaces: typeface.TypefaceMatcher | None = None
    skew: float = 0.0  # page-wide baseline slope, used when a line is too short to fit its own
    handwriting_typefaces: typeface.TypefaceMatcher | None = None
    rgb: np.ndarray | None = None  # original page in color, for sampling paper
    handwriting_face: typeface.Typeface | None = None  # the page's single handwriting typeface, if chosen
    identities: dict[Annotation, people.Person] = field(default_factory=dict)  # name boxes linked to one person
    addresses: dict[Annotation, addresses.FakeAddress] = field(default_factory=dict)  # shared address components
    shared_values: dict[Annotation, object] = field(default_factory=dict)  # repeated SSNs, phones, dates, ...


def document_handwriting_face(
    words: list[Word], matcher: typeface.TypefaceMatcher, max_words: int = 8
) -> typeface.Typeface | None:
    """One handwriting typeface for the whole page, voted on by its longest handwritten words.

    A document is assumed to be filled in by one person, so mixing handwriting
    fonts between its fields would look like several writers.
    """
    handwritten = sorted((w for w in words if w.handwritten), key=lambda w: -len(w.text))
    return matcher.choose(handwritten, max_words=max_words)


def erase(image: Image.Image, bbox: BBox, page: Page, settings: TextSettings) -> Image.Image:
    """Erase ``bbox`` with generated paper and redrawn lines, or plain white when the page wasn't kept."""
    if page.gray is None or page.rgb is None or not (settings.paper_fill or settings.keep_lines):
        return white_out(image, bbox)
    canvas = np.array(image.convert("RGB"))
    rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))  # follows --seed via numpy's global state
    rules = background.blank(canvas, page.rgb, page.gray, bbox, rng,
                             paper_fill=settings.paper_fill, keep_lines=settings.keep_lines, words=page.words)
    if rules:
        log.debug("redrew %d line(s) in box at (%d, %d)", len(rules), bbox.left, bbox.top)
    return Image.fromarray(canvas)


def locate_baseline(bbox: BBox, lines: list[list[Word]], page: Page) -> baseline.Baseline | None:
    """The scanned baseline the replacement's first line should sit on.

    The slope comes from the whole row of text across the page (or the page's
    skew); the height comes from the words in the box, or from that row when
    the box has no words of its own.
    """
    if page.gray is None:
        return None
    first = lines[0] if lines else []
    row_region = geometry.extent(first) or bbox
    row = geometry.words_in_row(row_region, page.words)
    line = baseline.fit_line(page.gray, row, fallback_slope=page.skew)
    slope = line.slope if line else page.skew
    return baseline.fit_line(page.gray, first, fallback_slope=slope) if first else line


def choose_font(bbox: BBox, inside: list[Word], page: Page, settings: TextSettings) -> tuple[str, bool]:
    """(font spec, handwritten) for this box.

    Whether the box is handwritten follows Textract's labels on the words in it
    (or the nearest words on its line when it has none). The typeface is then
    matched, within print or handwriting fonts, from those same words.
    """
    center = bbox.left + bbox.width / 2
    row = [w for w in geometry.words_in_row(bbox, page.words) if w not in inside]
    row.sort(key=lambda w: abs(w.bbox.left + w.bbox.width / 2 - center))
    handwritten = settings.handwriting and ocr.mostly_handwritten(inside or row[:3])

    if not settings.match_font:
        return settings.font, False
    if handwritten and page.handwriting_face is not None:
        # One writer per document: every handwritten box uses the typeface chosen for the page.
        log.debug("typeface for box at (%d, %d): %s (handwriting, per document)", bbox.left, bbox.top,
                  page.handwriting_face.name)
        return page.handwriting_face.spec, True
    matcher = page.handwriting_typefaces if handwritten else page.typefaces
    fallback = settings.handwriting_font if handwritten else settings.font
    if matcher is None:
        return fallback, handwritten
    by_length = sorted(inside, key=lambda w: -len(w.text))
    face = matcher.choose(by_length) or matcher.choose(row)
    log.debug("typeface for box at (%d, %d): %s%s", bbox.left, bbox.top,
              face.name if face else f"fallback {fallback}", " (handwriting)" if handwritten else "")
    return (face.spec if face else fallback), handwritten


RULE_GAP = 2.0  # px between a snapped baseline and the top of its rule


def snap_to_rules(
    baseline_y: float, x: float, font, spacing: float, n_lines: int, rules: list[background.Rule]
) -> tuple[float, float, float] | None:
    """Sit text on a redrawn form line when that line would otherwise run through the letters.

    Handwriting often overlaps its rule; copied exactly, a clean replacement then
    reads as struck through. Returns (baseline_y, spacing, angle_deg) with the
    first text line resting just above the rule and following its tilt, and
    later lines spaced to the rules below; None when no rule crosses the text.
    """
    x_height = layout.text_bbox(font, "x")
    x_height = x_height[3] - x_height[1]
    under = sorted((r for r in rules if r.x_start - 10 <= x <= r.x_end), key=lambda r: r.y_at(x))
    for i, rule in enumerate(under):
        rule_y = rule.y_at(x)
        if not (baseline_y - x_height - 1 <= rule_y <= baseline_y + rule.thickness + RULE_GAP):
            continue
        snapped = rule_y - rule.thickness / 2 - RULE_GAP
        line_body = layout.text_bbox(font, "A")[3]  # Pillow's line step is this plus the spacing
        if n_lines > 1 and i + 1 < len(under):
            pitch = under[i + 1].y_at(x) - rule_y
            if 0.6 * (line_body + spacing) <= pitch <= 1.8 * (line_body + spacing):
                spacing = max(0.0, pitch - line_body)
        return snapped, spacing, math.degrees(math.atan(rule.slope))
    return None


def plan_text(annotation: Annotation, page: Page, settings: TextSettings) -> Placement:
    """Decide the replacement text, its typeface, size, weight and where to draw it."""
    bbox, category = annotation.bbox, annotation.category
    page_words = page.words
    inside = geometry.words_inside(bbox, page_words)
    lines = geometry.group_lines(inside)
    original = geometry.reading_text(lines)
    ink = geometry.extent(inside)
    font_spec, handwritten = choose_font(bbox, inside, page, settings)

    size = settings.font_size
    if settings.auto_font_size:
        size = (
            layout.estimate_font_size(font_spec, inside)
            or layout.estimate_font_size(font_spec, geometry.words_in_row(bbox, page_words))
            or size
        )
        if handwritten and page.gray is not None and inside:
            # Cap by the writer's ink height too: wide cursive faces sized by width alone come out too tall.
            by_height = typeface.height_matched_size(font_spec, inside, page.gray)
            if by_height is not None:
                size = max(layout.MIN_FONT_SIZE, min(size, round(by_height)))
    font = layout.load_font(font_spec, size)
    spacing = layout.line_spacing_for_pitch(font, geometry.line_pitch(lines))

    if canonical(category) in MULTILINE_CATEGORIES:
        if lines:
            single_line = len(lines) == 1
        else:
            line_height = layout.text_bbox(font, "A")[3] + spacing
            single_line = bbox.height < 1.6 * line_height
    else:
        single_line = True

    detected = None
    if settings.match_style and original:
        style_source = original
        if ink and ink.right > bbox.right + 1:
            # Trailing punctuation past the box edge ("1958;") survives the white-out; don't draw it twice.
            style_source = original.rstrip(",;:.")
        detected = style.detect(category, style_source)
        if canonical(category) == "name" and isinstance(detected.detail, style.NameStyle):
            detected = dataclasses.replace(detected, detail=dataclasses.replace(
                detected.detail,
                title_outside=people.title_left_of(bbox, page_words),
                identity=page.identities.get(annotation),  # same person elsewhere on the page -> same fake name
            ))
    shared = page.addresses.get(annotation) or page.shared_values.get(annotation)
    if shared is not None and canonical(category) in fakedata.IDENTITY_CATEGORIES:
        # The same real value elsewhere on the page -> the same fake value, in this box's format
        # (applies even without style matching).
        detected = dataclasses.replace(detected or style.Style(), identity=shared)

    def generate() -> str:
        text = settings.overrides.get(category)
        if text is None:
            text = fakedata.fake_text(category, detected)
        return text.replace("\n", ", ") if single_line else text

    stroke = 0
    if settings.match_weight and page.gray is not None:
        # Short boxes ("m. 51;") have no measurable words; use the rest of the line.
        stroke = weight.estimate_stroke(font_spec, size, inside, page.gray) if inside else None
        if stroke is None:
            stroke = weight.estimate_stroke(font_spec, size, geometry.words_in_row(bbox, page_words), page.gray) or 0
    # The outline grows glyphs by half its thickness on every side.
    grow = layout.stroke_px(stroke) / 2

    x = (ink.left if ink else bbox.left + TEXT_INSET) + grow
    max_width = max(1.0, bbox.right - x - grow)
    if settings.fit_width and category not in settings.overrides:
        target = (ink.width if ink else 0.8 * max_width) - 2 * grow
        text = layout.choose_text(generate, font, max_width, target, spacing=spacing)
    else:
        text = generate()
    if settings.fit_width:
        size = layout.shrink_to_fit(font_spec, size, text, max_width, spacing)
        font = layout.load_font(font_spec, size)

    # Start the ink, not the drawing origin, where the original ink started: swashes and
    # side bearings otherwise push text left over labels like "SSN #".
    x -= layout.text_bbox(font, text, spacing)[0]
    _, ink_top, _, ink_bottom = layout.text_bbox(font, text, spacing)
    ascent = font.getmetrics()[0]  # Pillow's default anchor puts the first baseline this far below y
    scanned = locate_baseline(bbox, lines, page)
    angle = 0.0
    baseline_y = None
    if scanned is not None:
        baseline_y = scanned.y_at(x)
        if settings.match_skew:
            angle = scanned.angle_deg
        if settings.keep_lines and page.gray is not None:
            rules = background.detect_rules(page.gray, bbox, page.words)
            snapped = snap_to_rules(baseline_y, x, font, spacing, text.count("\n") + 1, rules)
            if snapped is not None:
                baseline_y, spacing, rule_angle = snapped
                if settings.match_skew:
                    angle = rule_angle
        y = baseline_y - ascent  # sit on the scanned baseline
    elif ink:
        y = ink.top + grow - ink_top  # no measurable ink: line glyph tops up with the OCR box
    else:
        y = bbox.top + (bbox.height - (ink_bottom - ink_top)) / 2 - ink_top
    return Placement(
        text=text, font=font_spec, font_size=size, xy=(x, y), spacing=spacing, original=original, stroke=stroke,
        angle_deg=angle, baseline_y=baseline_y, handwritten=handwritten,
    )


def pseudonymize(image: Image.Image, annotation: Annotation, page: Page, settings: TextSettings) -> Image.Image:
    """White out the region and draw replacement text into it with scan-like degradation."""
    plan = plan_text(annotation, page, settings)
    image = erase(image, annotation.bbox, page, settings)
    layer = Image.new("RGB", image.size, color="white")
    pivot = (plan.xy[0], plan.baseline_y) if plan.baseline_y is not None else None
    draw = handwriting.draw if plan.handwritten else layout.draw_text
    draw(layer, plan.xy, plan.text, plan.font, plan.font_size, plan.stroke, plan.spacing,
         angle_deg=plan.angle_deg, pivot=pivot)
    return augment.overlay(image, augment.degrade(layer))
