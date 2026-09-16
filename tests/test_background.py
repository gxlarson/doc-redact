import numpy as np
import pytest
from PIL import Image

from doc_redact import background, layout
from doc_redact.annotations import BBox
from doc_redact.ocr import Word

HAND = "PatrickHand-Regular.ttf"
BOX = BBox(left=200, top=80, width=260, height=40)


def _scan(paper=235, noise=6.0, specks=0.002, line_y=None, line_thickness=2, line_span=(60, 700), text="5-17-31"):
    """A grayscale-ish RGB scan: noisy paper, optional ruled line, handwriting sitting on the line."""
    rng = np.random.default_rng(0)
    gray = np.full((220, 800), paper, np.float32) + rng.normal(0, noise, (220, 800))
    speck_mask = rng.random((220, 800)) < specks
    gray[speck_mask] = 60
    if line_y is not None:
        gray[line_y:line_y + line_thickness, line_span[0]:line_span[1]] = 40
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")
    if text:
        layout.draw_text(img, (215, 72), text, HAND, 44)
    rgb = np.array(img)
    return rgb, np.asarray(img.convert("L"))


def _dark_inside(rgb, box, below=120):
    region = rgb[int(box.top):int(box.bottom) + 1, int(box.left):int(box.right) + 1].mean(axis=2)
    return region < below


def test_paper_is_sampled_from_the_ring_not_the_text():
    rgb, gray = _scan(line_y=None)
    paper = background.sample_paper(rgb, gray, BOX)
    assert paper.color[0] == pytest.approx(235, abs=3)
    assert paper.noise_std == pytest.approx(6, abs=2)
    assert paper.speck_density == pytest.approx(0.002, abs=0.002)


def test_blank_replaces_text_with_matching_paper():
    rgb, gray = _scan(line_y=None)
    out = rgb.copy()
    background.blank(out, rgb, gray, BOX, np.random.default_rng(1))
    patch = out[BOX.top:BOX.bottom + 1, BOX.left:BOX.right + 1].mean(axis=2)
    # Texture comparable to untouched paper of the same size elsewhere on the scan (grain + specks).
    reference = rgb[150:191, 500:761].mean(axis=2)
    assert patch.mean() == pytest.approx(reference.mean(), abs=3)
    assert patch.std() == pytest.approx(reference.std(), rel=0.3)
    assert _dark_inside(out, BOX).mean() < 0.01  # handwriting is gone; only a few specks


def test_writing_near_the_box_is_not_counted_as_paper_specks():
    rgb, gray = _scan(paper=250, noise=1.0, specks=0, line_y=None, text="")
    img = Image.fromarray(rgb)
    # Handwriting with i-dots and periods crossing the box edge and sitting in the sampling ring.
    layout.draw_text(img, (150, 60), "i.i.j.i Jim. ii", HAND, 60)
    layout.draw_text(img, (170, 118), "; : . , i", HAND, 30)
    rgb = np.array(img)
    paper = background.sample_paper(rgb, np.asarray(img.convert("L")), BOX)
    assert paper.speck_density < 0.0002


def test_clean_white_page_stays_white():
    rgb, gray = _scan(paper=255, noise=0, specks=0, line_y=None)
    out = rgb.copy()
    background.blank(out, rgb, gray, BOX, np.random.default_rng(1))
    assert (out[BOX.top:BOX.bottom + 1, BOX.left:BOX.right + 1] == 255).all()


def test_line_through_the_box_is_detected_and_redrawn_without_the_text():
    rgb, gray = _scan(line_y=110, line_thickness=2)
    [rule] = background.detect_rules(gray, BOX)
    assert rule.y_at(300) == pytest.approx(111, abs=0.6)
    assert rule.thickness == pytest.approx(2, abs=0.6)
    out = rgb.copy()
    background.blank(out, rgb, gray, BOX, np.random.default_rng(1))
    dark = _dark_inside(out, BOX)
    line_rows = slice(110 - BOX.top - 1, 112 - BOX.top + 1)
    assert dark[line_rows].any(axis=0).mean() > 0.95  # the line runs across the whole box
    off_line = np.delete(dark, np.r_[line_rows], axis=0)
    assert off_line.mean() < 0.01  # nothing of the handwriting survives off the line


def test_straight_underline_inside_the_box_is_kept_even_when_tilted():
    # Like a field underline the annotator's box fully contains: 2.7 deg tilt, 3 px short of both box edges.
    rgb, gray = _scan(line_y=None, text="")
    tilt = np.tan(np.radians(2.7))
    for x in range(203, 457):
        y = 108 + tilt * (x - 203)
        rgb[int(y):int(y) + 2, x] = 40
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    [rule] = background.detect_rules(gray, BOX)
    assert np.degrees(np.arctan(rule.slope)) == pytest.approx(2.7, abs=0.3)


def test_faint_notebook_rules_under_dark_handwriting_are_found():
    # White paper, light gray ruled lines (~25 levels darker, with breaks), dark writing crossing them.
    rgb, _ = _scan(paper=255, noise=0, specks=0, line_y=None, text="")
    for y in (88, 119):
        rgb[y, 20:780] = 230
        rgb[y, 20:780:17] = 255  # scanner breaks along the line
    img = Image.fromarray(rgb)
    layout.draw_text(img, (215, 72), "5-17-31 box", HAND, 44)
    gray = np.asarray(img.convert("L"))
    rules = background.detect_rules(gray, BOX)
    # A rule's position is the centre of its ink, so a line drawn on row r sits at r + 0.5.
    assert [round(y - 0.5) for y in sorted(r.y_at(300) for r in rules)] == [88, 119]
    assert all(220 <= r.gray <= 240 for r in rules)  # redrawn as faint as they were


def _typed_words(img, texts, font, size, x, y):
    """Draw typed words on one line; return OCR-like word boxes."""
    words = []
    f = layout.load_font(font, size)
    for t in texts:
        layout.draw_text(img, (x, y), t, font, size)
        l, top, r, b = f.getbbox(t)
        words.append(Word(t, BBox(x + l - 1, y + top - 3, r - l + 2, b - top + 5)))
        x += r + f.getlength(" ") * 1.2
    return words


def test_rows_of_typewriter_serifs_are_not_mistaken_for_lines():
    img = Image.new("RGB", (800, 220), "white")
    words = _typed_words(img, ["836", "HIMMEL", "MINIMUM", "LANE"], "CourierPrime-Bold.ttf", 34, 205, 80)
    gray = np.asarray(img.convert("L"))
    assert background.detect_rules(gray, BOX, words) == []


def test_faint_rule_crossing_handwriting_is_still_found_with_words():
    rgb, _ = _scan(paper=255, noise=0, specks=0, line_y=None, text="")
    rgb[112, 20:780] = 228
    img = Image.fromarray(rgb)
    words = _typed_words(img, ["5-17-31", "box"], HAND, 44, 215, 76)
    for w in words:
        object.__setattr__(w, "handwritten", True)
    gray = np.asarray(img.convert("L"))
    [rule] = background.detect_rules(gray, BOX, words)
    assert round(rule.y_at(300)) == 112


def test_noisy_or_halftone_paper_does_not_produce_lines():
    rgb, gray = _scan(paper=225, noise=9.0, specks=0.004, line_y=None, text="")
    assert background.detect_rules(gray, BOX) == []
    shaded = rgb.copy()
    shaded[60:140, 100:700] = 255
    shaded[60:140:3, 100:700:3] = 150  # halftone dot screen over the box area
    assert background.detect_rules(np.asarray(Image.fromarray(shaded).convert("L")), BOX) == []


def test_scan_texture_next_to_a_box_is_not_a_line():
    # Dense random shading (like a halftone-screened area) around a small box, with no words nearby.
    rng = np.random.default_rng(5)
    gray = np.full((220, 800), 255, np.uint8)
    shade = rng.random((60, 400)) < 0.5
    gray[70:130, 150:550][shade] = 90
    small = BBox(left=330, top=85, width=40, height=30)
    assert background.detect_rules(gray, small, []) == []
    # The same box on clean paper with a real rule through the shading's rows is still found.
    gray = np.full((220, 800), 255, np.uint8)
    gray[100:102, 100:700] = 60
    [rule] = background.detect_rules(gray, small, [])
    assert round(rule.y_at(350)) in (100, 101)


def test_short_or_wobbly_strokes_inside_the_box_are_not_lines():
    rgb, gray = _scan(line_y=100, line_thickness=2, line_span=(230, 330), text="")  # 100 px dash, < 60% of box
    assert background.detect_rules(gray, BOX) == []

    rgb, _ = _scan(line_y=None, text="")
    for x in range(205, 455):  # long handwritten-looking stroke: spans the box but wobbles
        y = int(round(105 + 3 * np.sin((x - 205) / 9)))
        rgb[y:y + 3, x] = 40
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    assert background.detect_rules(gray, BOX) == []


def test_text_snaps_above_a_rule_that_would_strike_through_it():
    from doc_redact import render

    font = layout.load_font(HAND, 30)
    tilt = np.tan(np.radians(2.0))
    rules = [
        background.Rule(x0=300, y0=110, slope=tilt, thickness=2, x_start=150, x_end=600, gray=30),
        background.Rule(x0=300, y0=150, slope=tilt, thickness=2, x_start=150, x_end=600, gray=30),
    ]
    # Scanned baseline 4 px below the first rule: the rule would cut through the letters.
    first_rule_y = rules[0].y_at(220)
    snapped = render.snap_to_rules(first_rule_y + 4, 220, font, 4.0, 2, rules)
    assert snapped is not None
    baseline_y, spacing, angle = snapped
    assert baseline_y == pytest.approx(first_rule_y - 1 - render.RULE_GAP)
    assert angle == pytest.approx(2.0, abs=0.01)
    # Second text line lands the same distance above the second rule.
    line_body = layout.text_bbox(font, "A")[3]
    assert baseline_y + line_body + spacing == pytest.approx(rules[1].y_at(220) - 1 - render.RULE_GAP)
    # Text already well above a rule is left alone.
    assert render.snap_to_rules(first_rule_y - 25, 220, font, 4.0, 1, rules) is None


def test_lines_can_be_turned_off():
    rgb, gray = _scan(line_y=110)
    out = rgb.copy()
    assert background.blank(out, rgb, gray, BOX, np.random.default_rng(1), keep_lines=False) == []
    assert _dark_inside(out, BOX).mean() < 0.01


def test_a_redrawn_rule_keeps_the_original_line_dark_and_in_place():
    """A scanned hairline is one dark row with light antialiasing either side. Counting every row
    under the threshold made it three rows thick, and the median of their values made it mid-grey,
    so the redrawn line came out as a grey band sitting half a pixel high."""
    rgb = np.full((200, 600, 3), 250, np.uint8)
    row = 120
    rgb[row] = 8               # the line's core
    rgb[row - 1] = 205         # antialiasing above
    rgb[row + 1] = 160         # and below
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    box = BBox(left=100, top=row - 12, width=300, height=24)

    [rule] = background.detect_rules(gray, box)
    assert rule.gray < 30, "the redrawn line should carry the core's darkness, not a blend with its edges"
    assert rule.thickness < 2.0, "one dark row, not the full extent of everything under the threshold"
    assert abs(rule.y_at(np.array([250.0]))[0] - (row + 0.5)) < 0.4  # ink centre, not row index

    canvas = rgb.copy()
    background.blank(canvas, rgb, gray, box, np.random.default_rng(0), paper_fill=False, keep_lines=True)
    drawn = canvas[:, int(box.left) + 10]
    assert drawn[row].min() < 40  # the line is back, and dark
    assert drawn[row - 2].min() > 200 and drawn[row + 2].min() > 200  # without a grey halo
