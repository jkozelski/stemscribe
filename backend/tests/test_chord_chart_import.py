"""Tests for backend.processing.chord_chart_import."""

import io
import os
import tempfile
from pathlib import Path

import pytest

from processing.chord_chart_import import (
    CHORD_RE,
    CHORD_TOKEN_RE,
    SECTION_RE,
    extract_text_from_pdf,
    is_chord_line,
    parse_chord_chart_text,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------- regex sanity ----------

@pytest.mark.parametrize(
    "tok",
    [
        "C", "Am", "F#m", "Bb", "G7", "Dm7", "Cmaj7", "F#m7b5",
        "Csus2", "Dsus4", "Cadd9", "G/B", "D/F#", "Bbmaj7", "C#dim",
    ],
)
def test_chord_regex_matches_common_chords(tok):
    assert CHORD_TOKEN_RE.match(tok), f"{tok} should match"


@pytest.mark.parametrize("tok", ["yesterday", "Hello", "12", "the", "ZZ"])
def test_chord_regex_rejects_non_chords(tok):
    # CHORD_TOKEN_RE requires a full-token match
    assert not CHORD_TOKEN_RE.match(tok), f"{tok} should not match as a whole chord token"


def test_is_chord_line_positive():
    assert is_chord_line("C       G       Am      F")
    assert is_chord_line("F#m   Bm   D   A")
    assert is_chord_line("Em7  A7  Dm  Dm/C  Bb  Dm/A  G7  C7")


def test_is_chord_line_negative():
    assert not is_chord_line("Yesterday all my troubles seemed so far away")
    assert not is_chord_line("")
    assert not is_chord_line("This is a normal sentence of lyrics with many words in it indeed")


def test_section_regex():
    assert SECTION_RE.match("[Verse]")
    assert SECTION_RE.match("[Chorus 2]")
    assert SECTION_RE.match("  [Bridge]  ")
    assert not SECTION_RE.match("Verse")
    assert not SECTION_RE.match("[Am]when the sun")  # inline-chord, not section


# ---------- parser ----------

def test_parse_empty_returns_no_sections():
    chart = parse_chord_chart_text("", source="paste")
    assert chart["sections"] == []
    assert chart["source"] == "imported:paste"


def test_parse_all_chords_no_lyrics():
    text = "[Intro]\nC  G  Am  F\n"
    chart = parse_chord_chart_text(text)
    assert len(chart["sections"]) == 1
    sec = chart["sections"][0]
    assert sec["name"] == "Intro"
    assert sec["lines"][0]["chords"].strip() == "C  G  Am  F"
    assert sec["lines"][0]["lyrics"] == ""


def test_parse_all_lyrics_no_chords():
    text = "[Verse]\nI am the walrus\nGoo goo g'joob\n"
    chart = parse_chord_chart_text(text)
    sec = chart["sections"][0]
    lyrics = [ln["lyrics"] for ln in sec["lines"]]
    assert "I am the walrus" in lyrics
    assert "Goo goo g'joob" in lyrics
    assert all(ln["chords"] == "" for ln in sec["lines"])


def test_parse_section_markers_only():
    text = "[Verse]\n[Chorus]\n[Bridge]\n"
    chart = parse_chord_chart_text(text)
    # All empty sections drop out — only ones with lines stay
    assert chart["sections"] == []


def test_parse_capo_line_pulls_metadata():
    text = "Capo: 3rd fret\n[Verse]\nC G Am F\nLine of lyrics\n"
    chart = parse_chord_chart_text(text)
    assert chart["capo"] == 3


def test_parse_key_line_pulls_metadata():
    text = "Key: Am\n[Verse]\nAm F C G\nThe lyric goes here\n"
    chart = parse_chord_chart_text(text)
    assert chart["key"] == "Am"


def test_parse_chord_lyric_pair():
    text = "[Verse 1]\nC       G       Am      F\nThis is a line of lyrics under the chords\n"
    chart = parse_chord_chart_text(text)
    sec = chart["sections"][0]
    assert sec["name"] == "Verse 1"
    line = sec["lines"][0]
    assert "C" in line["chords"]
    assert "G" in line["chords"]
    assert line["lyrics"].startswith("This is a line of lyrics")


def test_parse_ug_fixture():
    text = (FIXTURES / "sample_ug_chart.txt").read_text()
    chart = parse_chord_chart_text(text, source="txt")
    assert chart["capo"] == 5
    # Three sections: Verse 1, Verse 2, Bridge
    names = [s["name"] for s in chart["sections"]]
    assert names == ["Verse 1", "Verse 2", "Bridge"]
    # First Verse 1 line: chord row pairs with the lyric row below it
    v1_lines = chart["sections"][0]["lines"]
    chord_lines = [ln for ln in v1_lines if ln["chords"].strip()]
    assert len(chord_lines) >= 3
    assert "Yesterday all my troubles" in chord_lines[0]["lyrics"]
    # chords_used should contain at least F, Em7, A7, Dm, Bb, C7, G
    expected_subset = {"F", "Em7", "A7", "Dm", "Bb", "C7", "G"}
    assert expected_subset.issubset(set(chart["chords_used"]))


def test_parse_chordpro_directives_and_inline():
    text = (FIXTURES / "sample_chordpro.cho").read_text()
    chart = parse_chord_chart_text(text, source="chordpro")
    assert chart["title"] == "Amazing Grace"
    assert chart["artist"] == "Traditional"
    assert chart["key"] == "G"
    # Inline-chord parse produces a chord_line with chord names above lyrics
    all_lines = [ln for s in chart["sections"] for ln in s["lines"]]
    assert any("G" in ln["chords"] and "mazing" in (ln["lyrics"] or "") for ln in all_lines), (
        "Expected first inline-chord line to contain G chord and 'mazing' lyric"
    )


def test_parse_source_metadata_passed_through():
    chart = parse_chord_chart_text("[Verse]\nC G\nlyric\n", source="pdf")
    assert chart["source"] == "imported:pdf"


# ---------- PDF extraction ----------

def _make_test_pdf(lines):
    """Render a list of lines into a temp PDF using reportlab; return path."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Courier", 10)
    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 14
    c.save()
    return path


def test_extract_text_from_pdf_roundtrip():
    pdf_path = _make_test_pdf(
        [
            "[Verse 1]",
            "C       G       Am      F",
            "This is the lyric for verse one",
            "",
            "[Chorus]",
            "F       C       G",
            "And here is the chorus lyric",
        ]
    )
    try:
        text = extract_text_from_pdf(pdf_path)
        assert "Verse 1" in text
        assert "Am" in text
        assert "chorus lyric" in text
        chart = parse_chord_chart_text(text, source="pdf")
        assert [s["name"] for s in chart["sections"]] == ["Verse 1", "Chorus"]
        # First section has a chord-over-lyric pair
        first_lines = chart["sections"][0]["lines"]
        assert any("lyric for verse one" in (ln["lyrics"] or "") for ln in first_lines)
    finally:
        os.unlink(pdf_path)


def test_extract_text_from_empty_pdf_returns_empty_string():
    """A blank PDF (no text, no image content) has nothing to extract.
    pdfplumber returns empty, OCR fallback runs on a blank page and also
    returns empty. The route layer turns that into a 422."""
    # Make a PDF with no text at all (just a blank page)
    from reportlab.pdfgen import canvas
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    c = canvas.Canvas(path)
    c.showPage()
    c.save()
    try:
        text = extract_text_from_pdf(path)
        assert text.strip() == ""
    finally:
        os.unlink(path)


# ---------- OCR fallback ----------

def _make_image_only_pdf(text_to_render: str) -> str:
    """Render text into a PNG via PIL, then embed that PNG in a PDF.
    pdfplumber will get zero text out of this — it's image-only, like a
    UG 'Save as PDF' export."""
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import letter

    # Render text to a PNG
    img = Image.new("RGB", (1200, 1600), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", 36)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()
    y = 60
    for line in text_to_render.splitlines():
        draw.text((60, y), line, fill="black", font=font)
        y += 50
    png_fd, png_path = tempfile.mkstemp(suffix=".png")
    os.close(png_fd)
    img.save(png_path, "PNG")

    # Embed the PNG into a PDF (no text layer at all)
    pdf_fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(pdf_fd)
    c = rl_canvas.Canvas(pdf_path, pagesize=letter)
    c.drawImage(png_path, 50, 50, width=500, height=700)
    c.save()
    os.unlink(png_path)
    return pdf_path


def test_ocr_fallback_extracts_text_from_image_only_pdf():
    """Smoke test: an image-only PDF (like a UG 'Save as PDF' export) gets
    text out via the Tesseract OCR fallback. Skipped if tesseract isn't
    installed locally — prod has it."""
    pytest.importorskip("pytesseract")
    pytest.importorskip("pdf2image")
    pytest.importorskip("PIL")
    import shutil
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed locally")
    if not shutil.which("pdftoppm"):
        pytest.skip("poppler-utils (pdftoppm) not installed locally")

    pdf_path = _make_image_only_pdf("[Verse 1]\nC G Am F\nHello world lyric line")
    try:
        text = extract_text_from_pdf(pdf_path)
        # OCR isn't pixel-perfect, but at minimum we should see something
        # back from a clean black-on-white render.
        assert len(text.strip()) > 0, "OCR returned nothing for image-only PDF"
        # At least one recognizable chord token should survive
        assert any(tok in text for tok in ("C", "G", "Am", "F")), (
            f"Expected at least one chord token in OCR output, got: {text!r}"
        )
    finally:
        os.unlink(pdf_path)


def test_ocr_not_invoked_for_text_pdfs(monkeypatch):
    """When pdfplumber returns plenty of text, the OCR path must not run —
    OCR is slow and we don't want to pay for it on every text PDF."""
    called = {"n": 0}

    def fake_image_to_string(*args, **kwargs):
        called["n"] += 1
        return "should not be called"

    # Patch pytesseract if it's importable; otherwise the OCR branch can't
    # run anyway and this test still proves the negative.
    try:
        import pytesseract
        monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)
    except ImportError:
        pass

    pdf_path = _make_test_pdf(
        [
            "[Verse 1]",
            "C       G       Am      F",
            "This is the lyric for verse one with plenty of characters",
            "",
            "[Chorus]",
            "F       C       G",
            "And here is the chorus lyric also with plenty of text",
        ]
    )
    try:
        text = extract_text_from_pdf(pdf_path)
        assert "Verse 1" in text
        assert called["n"] == 0, "pytesseract.image_to_string should not be called for text PDFs"
    finally:
        os.unlink(pdf_path)


def test_ocr_failure_is_non_fatal(monkeypatch):
    """If pytesseract raises mid-OCR, extract_text_from_pdf must not
    propagate the exception — it should return whatever pdfplumber had
    (empty string for a scanned PDF) so the endpoint can hand back a
    clean 422 instead of a 500."""
    pytesseract = pytest.importorskip("pytesseract")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic OCR failure")

    monkeypatch.setattr(pytesseract, "image_to_string", boom)

    # Blank PDF triggers the OCR fallback path
    from reportlab.pdfgen import canvas
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    c = canvas.Canvas(path)
    c.showPage()
    c.save()
    try:
        # Must not raise
        text = extract_text_from_pdf(path)
        # And should return what pdfplumber had (empty)
        assert text.strip() == ""
    finally:
        os.unlink(path)
