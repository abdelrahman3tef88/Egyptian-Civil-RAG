"""
الملف ده هو أول خطوة في الـingestion pipeline.
وظيفته ببساطة:
يجيب الـPDF → يتأكد إنه موجود وسليم → يتأكد هل فيه Text حقيقي ولا Scan.
وهو مش مسؤول عن استخراج أرقام المواد أو تنظيف النص أو الـchunking.
"""

# PDF loading, validation, and the simple digital/scan type check.
#
# This module is the INPUT boundary of the ingestion pipeline. It knows:
#   - where the single source PDF lives (one configurable path constant)
#   - how to open it safely with clear errors
#   - how to answer one simple question:
#         is the PDF digitally extractable (1) or scanned (0)?
#
# It does NOT do column splitting, article detection, or cleaning.
# Those responsibilities live in extraction.py and cleaning.py.

from pathlib import Path

# We use the modern PyMuPDF module name. The legacy fitz module name
# is deprecated in the installed version and prints a warning, so it
# is avoided everywhere in this project.
# PyMuPDF ببساطة هي Python library للتعامل مع ملفات الـPDF.
# دي بتفتح الـPDF وتخليك تقدر تتعامل معاه صفحة صفحة.
# document = pymupdf.open(pdf_path)
#                |
# page = document.load_page(0)  # تجيب أول صفحة.
#                |
# text = page.get_text()        # تستخرج الـtext الموجود داخل الصفحة.
import pymupdf


# ============================================================
# Project PDF Path
# ----------------------------------------------------------
# The project has exactly ONE source document: the Egyptian law
# PDF inside data/raw/. Defining the path once here means no
# other file needs to hardcode it, and pathlib handles the
# Arabic filename safely without any shell tricks.
# ============================================================
# __file__ = .../src/rag_project/ingestion/loaders.py
# parents[0]=ingestion, [1]=rag_project, [2]=src, [3]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "القانون المصري.pdf"

# How many pages the digital/scan check looks at.
# A small sample keeps the check fast while still being reliable.
# for pdf check type
PDF_TYPE_SAMPLE_PAGES = 10

# A page with fewer extracted characters than this is considered
# "scanned": digitally created PDFs contain selectable text
# (hundreds of characters per page), scanned pages contain none.
# for pdf check type
MIN_CHARS_PER_PAGE = 200


# ============================================================
# Open and Validate the PDF
# ----------------------------------------------------------
# Every pipeline run starts here. The function fails fast with a
# clear message when the PDF is missing, unreadable, or empty,
# because continuing with broken input would produce corrupted
# article data later on.
# ============================================================
def open_pdf(pdf_path=None):
    """Open the Egyptian law PDF and run basic validation.

    Returns an open pymupdf Document. Raises FileNotFoundError or
    RuntimeError with a clear message when the input is unusable.
    """
    # Use the default project path when the caller gives no path.
    if pdf_path is None:
        pdf_path = PDF_PATH

    # Normalize to a Path object so .exists() always works.
    pdf_path = Path(pdf_path)

    # Error 1: the file simply does not exist.
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Error 2: the file exists but cannot be opened as a PDF
    # (truncated, password protected, not a PDF at all, ...).
    # لو الملف corrupted أو مش PDF صحيح أو فيه مشكلة تمنع فتحه:
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as error:
        raise RuntimeError(f"Cannot open PDF: {pdf_path} ({error})")

    # Error 3: the PDF opened but has zero pages, so there is
    # nothing we could ever extract from it.
    if document.page_count == 0:
        document.close()  # Release the file handle before failing.
        raise RuntimeError(f"PDF has no pages: {pdf_path}")

    return document


# ============================================================
# PDF Type Check: DIGITAL (1) vs SCAN (0)
# ----------------------------------------------------------
# This check is intentionally small and deterministic. It only
# answers: "can we extract real text from this PDF?".
# Digitally created PDFs store selectable text on their pages;
# scanned PDFs store only images, so get_text() returns close to
# nothing. We sample the first few pages and compare the average
# amount of extracted text against a simple threshold.
#
# NO OCR is performed here (and OCR is out of scope for this
# ingestion pipeline). A scanned PDF must stop the pipeline with
# a clear message instead of silently producing empty articles.
# ============================================================
def check_pdf_type(pdf_path=None):
    """Return 1 when the PDF is DIGITAL, 0 when it is a SCAN."""
    # Opening also validates existence, readability, and pages.
    document = open_pdf(pdf_path)

    # Never sample more pages than the document actually has.
    sample_pages = min(PDF_TYPE_SAMPLE_PAGES, document.page_count)

    # Collect how much text each sampled page contains.
    total_characters = 0
    for page_index in range(sample_pages):
        page = document.load_page(page_index)  # Load one sampled page.
        # PyMuPDF بيحاول يقرأ الـtext الموجود داخل الصفحة.
        text = page.get_text()  # Extract its raw text layer.
        total_characters += len(text.strip())  # Count the characters.

    # We are done with the file handle.
    document.close()

    # Average characters per sampled page for the threshold comparison.
    average_characters = total_characters / sample_pages

    # Enough text -> digitally created PDF.
    if average_characters >= MIN_CHARS_PER_PAGE:
        return 1  # DIGITAL

    # Almost no text -> scanned PDF (images only).
    return 0  # SCAN
