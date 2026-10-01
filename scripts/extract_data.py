"""extract_data.py — PDF → structured articles.json (ingestion entry point).

This is the ORCHESTRATION layer (an existing project script, not a new
ingestion module). It performs the full run:

  1. check the PDF type (digital / scan)          [loaders]
  2. open the PDF and extract all articles        [extraction + cleaning]
  3. cross-check the result vs. the English column
  4. print a calculated summary + validation report
  5. write data/processed/articles.json ONLY when validation passes

Run with:  python scripts/extract_data.py
"""

import json
import sys

# ============================================================
# Safe Console Output for Arabic Text
# ----------------------------------------------------------
# On Windows, a piped/captured stdout often uses a legacy encoding
# (e.g. cp1252) that CANNOT encode Arabic characters - printing a
# diagnostic sample would then crash the whole run before the report
# finishes. Forcing UTF-8 makes every print safe everywhere.
# ============================================================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # All Unicode is encodable.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # Error messages too.

from rag_project.ingestion import extraction, loaders


# ============================================================
# Output Path and Validation Thresholds
# ----------------------------------------------------------
# The JSON product of ingestion goes next to the other processed
# data. The thresholds below are named and evidence-based (set
# from inspecting this exact PDF) so validation never depends on
# hardcoded article counts.
# ============================================================
OUTPUT_PATH = loaders.PROJECT_ROOT / "data" / "processed" / "articles.json"

# Share of our Arabic article numbers that must be confirmable in
# the English column (with a 1-page window for boundary shifts).
MIN_ENGLISH_MATCH_RATIO = 0.80
# Share of the English header space we must have found articles for.
MIN_ENGLISH_COVERAGE_RATIO = 0.80
# The extracted text must be predominantly Arabic characters.
MIN_ARABIC_RATIO = 0.50


# ============================================================
# Small Diagnostic Helpers
# ============================================================
def _find_latin_words(text):
    """Return the list of long Latin-script tokens (possible contamination)."""
    words = []
    current = ""
    for character in text:
        if character.isascii() and character.isalpha():
            current += character  # Grow the current Latin run.
        else:
            if len(current) >= 6:  # Long English words only.
                words.append(current)
            current = ""  # Reset at every non-Latin character.
    if len(current) >= 6:
        words.append(current)  # Run that ends at the very end.
    return words


def _starts_with_heading_word(line):
    """True when a line starts with a chapter/section word (reported only)."""
    # Chapter/section headings are PRESERVED, never deleted; we only
    # count them so the report can mention their occurrence.
    heading_words = ("الباب ", "الفصل ", "القسم ", "العنوان ")
    for word in heading_words:
        if line.startswith(word):
            return True
    return False


# ============================================================
# Calculate Every Reported Value From the Actual Run
# ----------------------------------------------------------
# Nothing here is hardcoded: all counts and ratios are derived
# from the articles and pages that were really extracted. If the
# PDF changes, these numbers change with it.
# ============================================================
def compute_metrics(articles, preamble, rejected_headers, english_per_page, page_count):
    """Compute all summary/validation numbers from the real extraction result."""
    metrics = {}

    # --- basic article counts (0 when nothing was found) ---
    metrics["article_count"] = len(articles)
    metrics["first_article"] = articles[0]["article_number"] if articles else 0
    metrics["last_article"] = articles[-1]["article_number"] if articles else 0

    # --- one pass over the articles for all per-article diagnostics ---
    multi_page = 0  # Articles whose text spans more than one page.
    repealed = 0  # Articles inside a repealed range (kept, marked).
    empty_text = 0  # Articles with no ARABIC text.
    empty_english_text = 0  # Articles with no ENGLISH text.
    empty_both_languages_count = 0  # Articles with NO text in either language.
    monotonic_violations = 0  # Numbers must strictly increase.
    page_range_errors = 0  # page_start/page_end outside the document.
    previous_number = 0  # The last accepted article number.
    for article in articles:
        # An article whose text ends on a later page than it started.
        if article["page_end"] > article["page_start"]:
            multi_page += 1
        # Repealed articles are marked, never deleted.
        if article["is_repealed"]:
            repealed += 1
        # An article with no text in one language means that side of
        # the extraction failed for it. A record that is empty in BOTH
        # languages is a real extraction failure.
        if article["text"]["ar"].strip() == "":
            empty_text += 1
        if article["text"]["en"].strip() == "":
            empty_english_text += 1
        if article["text"]["ar"].strip() == "" and article["text"]["en"].strip() == "":
            empty_both_languages_count += 1
        # The article numbers must strictly increase (guard worked).
        if article["article_number"] <= previous_number:
            monotonic_violations += 1
        previous_number = article["article_number"]
        # page_start/page_end must stay inside the real document.
        if not (1 <= article["page_start"] <= article["page_end"] <= page_count):
            page_range_errors += 1
    metrics["multi_page_count"] = multi_page
    metrics["repealed_count"] = repealed
    metrics["empty_text_count"] = empty_text
    metrics["empty_english_text_count"] = empty_english_text
    metrics["empty_both_languages_count"] = empty_both_languages_count
    metrics["monotonic_violations"] = monotonic_violations
    metrics["page_range_errors"] = page_range_errors
    metrics["rejected_count"] = len(rejected_headers)

    # --- Arabic presence and English contamination ---
    # The Arabic ratio is measured on the ARABIC text only, because the
    # record now also contains the English translation of the same law.
    arabic_texts = [preamble] + [a["text"]["ar"] for a in articles]
    all_text = "\n".join(arabic_texts)
    arabic_chars = 0
    total_chars = 0
    for character in all_text:
        if not character.isspace():
            total_chars += 1  # Count every visible character.
            if "\u0600" <= character <= "\u06ff":
                arabic_chars += 1  # Count Arabic-script characters.
    metrics["arabic_ratio"] = arabic_chars / total_chars if total_chars else 0.0
    # Long Latin runs inside Arabic text = column-contamination signal.
    latin_words = _find_latin_words(all_text)
    unique_latin = []
    for token in latin_words:
        if token not in unique_latin:
            unique_latin.append(token)
    metrics["contamination_count"] = len(latin_words)
    metrics["contamination_tokens"] = unique_latin[:10]  # sample for report

    # --- chapter/section heading lines (preserved, only reported) ---
    # Count heading lines inside the ARABIC text only; English headings
    # use different words and are not part of this diagnostic.
    heading_count = 0
    for article in articles:
        for line in article["text"]["ar"].split("\n"):
            if _starts_with_heading_word(line):
                heading_count += 1  # We only count them, never remove them.
    metrics["heading_line_count"] = heading_count

    # --- English cross-check (validation only, page window +/-1) ---
    # Union of every "Article N" seen anywhere in the English column.
    english_union = set()
    for page_numbers in english_per_page:
        english_union = english_union | page_numbers
    metrics["english_union_count"] = len(english_union)

    # For each Arabic article, look for its number in the English
    # column on the same page OR the adjacent pages (columns can
    # break articles at slightly different pages), so stray English
    # cross-references on other pages do not fail validation.
    matched = 0
    for article in articles:
        index = article["page_start"] - 1  # 0-based page index.
        window = set()
        for neighbor in (index - 1, index, index + 1):  # +/-1 page tolerance.
            if 0 <= neighbor < page_count:
                window = window | english_per_page[neighbor]
        if article["article_number"] in window:
            matched += 1
    metrics["english_matched"] = matched
    metrics["english_match_ratio"] = matched / len(articles) if articles else 0.0
    metrics["english_coverage_ratio"] = (
        len(articles) / len(english_union) if english_union else 0.0
    )

    # --- numbering continuity (gaps: where the numbering jumps) ---
    # Each gap is a diagnostic: big jumps over 54-80 / 389-417 are
    # the (expected) source omission of the repealed ranges; other
    # gaps are suspicious extraction misses to review honestly.
    gap_count = 0
    gap_details = []
    for index in range(1, len(articles)):
        previous = articles[index - 1]["article_number"]
        current = articles[index]["article_number"]
        if current - previous > 1:
            gap_count += 1
            gap_details.append((previous, current))
    metrics["numbering_gaps"] = gap_count
    metrics["gap_details"] = gap_details

    return metrics


# ============================================================
# Validation Checks (each returns True/False with a clear rule)
# ----------------------------------------------------------
# Every rule is based on values calculated from the real run.
# A check only passes when the evidence supports it - failures
# are reported honestly instead of being papered over.
# ============================================================
def run_validation(articles, metrics):
    """Run the five validation checks and return {check_name: passed}."""
    checks = {}

    # 1) PDF Check: reaching this point means the file existed,
    #    opened, had pages, and was classified as DIGITAL.
    checks["PDF Check"] = True

    # 2) Arabic Extraction: text is mostly Arabic, no English
    #    column leaked into it, and articles were produced.
    checks["Arabic Extraction"] = (
        metrics["arabic_ratio"] >= MIN_ARABIC_RATIO
        and metrics["contamination_count"] == 0
        and metrics["article_count"] > 0
    )

    # 3) Article Parsing: articles exist, EVERY article has text in at
    #    least one language, numbering is strictly increasing, pages are
    #    sane, and the English column confirms both our numbers (match)
    #    and our count (coverage).
    #    NOTE: this source edition physically omits articles 55-80,
    #    389-417 and 1022 in BOTH columns, and the Arabic header of
    #    article 601 is stored with corrupted digits in the PDF itself.
    #    A few one-language records are therefore expected; they are
    #    reported explicitly instead of being hidden or fabricated.
    checks["Article Parsing"] = (
        metrics["article_count"] > 0
        and metrics["empty_both_languages_count"] == 0
        and metrics["monotonic_violations"] == 0
        and metrics["page_range_errors"] == 0
        and metrics["english_match_ratio"] >= MIN_ENGLISH_MATCH_RATIO
        and metrics["english_coverage_ratio"] >= MIN_ENGLISH_COVERAGE_RATIO
    )

    # 4) Multi-page Handling: every article has a valid page range
    #    AND at least one article really spans two pages (inspection
    #    confirmed multi-page articles exist in this PDF, so 0 is
    #    suspicious and must fail).
    checks["Multi-page Handling"] = (
        metrics["page_range_errors"] == 0 and metrics["multi_page_count"] >= 1
    )

    # 5) Repealed Handling: every flag must match an independent
    #    recomputation from the two repealed ranges - and we must
    #    never delete articles just because they are repealed.
    #    NOTE: this source edition does not contain the text of the
    #    repealed ranges at all (the numbering jumps 53->81 and
    #    388->418, and the PDF itself carries the repeal notes on
    #    p7 and p53), so repealed_count can legitimately be 0.
    #    The check therefore verifies consistent marking of every
    #    article that IS present, not the presence of the ranges.
    repealed_ok = True
    for article in articles:
        expected = False
        for start, end in [(54, 80), (389, 417)]:  # Independent copy of the ranges.
            if start <= article["article_number"] <= end:
                expected = True
        if article["is_repealed"] != expected:
            repealed_ok = False
    checks["Repealed Handling"] = repealed_ok

    return checks


# ============================================================
# Print the Diagnostics + Calculated Summary + Validation Report
# ----------------------------------------------------------
# The notes block carries the extra values required in the report
# (first/last article, processed pages, empty texts, rejected
# headers, contamination, headings). The two fixed blocks show the
# headline summary and PASS/FAIL per check.
# ============================================================
def print_report(metrics, checks, pdf_type, page_count, rejected_headers):
    """Print diagnostics, the ingestion summary, and the validation report."""
    type_label = "DIGITAL" if pdf_type == 1 else "SCAN"

    # --- diagnostics (all calculated from this run) ---
    print()
    print("NOTES (calculated from this run):")
    print(f"  Processed pages          : {page_count}")
    print(f"  First article            : {metrics['first_article']}")
    print(f"  Last article             : {metrics['last_article']}")
    print(f"  Empty article texts      : {metrics['empty_text_count']} (Arabic)")
    print(f"  Empty English texts      : {metrics['empty_english_text_count']}")
    print(f"  Arabic character ratio   : {metrics['arabic_ratio']:.2%}")
    print(f"  English match ratio      : {metrics['english_match_ratio']:.2%} "
          f"({metrics['english_matched']}/{metrics['article_count']}, window +/-1 page)")
    print(f"  English coverage ratio   : {metrics['english_coverage_ratio']:.2%} "
          f"({metrics['article_count']} articles / {metrics['english_union_count']} "
          f"English Article-N numbers)")
    print(f"  Numbering gaps (>1)      : {metrics['numbering_gaps']}")
    # Show every gap pair: 53->81 and 388->418 are the repealed-range
    # omissions of this source edition; any other pair is reviewed.
    for previous, current in metrics["gap_details"]:
        print(f"      {previous} -> {current} (missing {current - previous - 1})")
    print(f"  Rejected header candidates: {metrics['rejected_count']}")
    for page_number, line in rejected_headers[:5]:  # First 5 samples only.
        print(f"      p{page_number}: {line[:70]}")
    print(f"  Latin-token contamination : {metrics['contamination_count']} "
          f"{metrics['contamination_tokens']}")
    print(f"  Heading lines (preserved) : {metrics['heading_line_count']}")

    # --- headline summary block ---
    print()
    print("=" * 60)
    print("EGYPTIAN LAW INGESTION SUMMARY")
    print("=" * 60)
    print()
    print(f"PDF Type              : {type_label}")
    print(f"Pages                 : {page_count}")
    print(f"Arabic Text Extracted : {'YES' if metrics['arabic_ratio'] >= MIN_ARABIC_RATIO else 'NO'}")
    print(f"Articles Extracted    : {metrics['article_count']}")
    print(f"Repealed Articles     : {metrics['repealed_count']}")
    print(f"Multi-page Articles   : {metrics['multi_page_count']}")

    # --- validation block ---
    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print()
    for check_name, passed in checks.items():
        # Print each check on the fixed-width label + PASS/FAIL.
        print(f"{check_name:<24}: {'PASS' if passed else 'FAIL'}")
    print()
    print("=" * 60)


# ============================================================
# Main Orchestration: type check -> extract -> validate -> write
# ----------------------------------------------------------
# Simple linear flow. Every error path prints one clear message
# and returns a non-zero exit code; articles.json is written ONLY
# when all five validation checks pass.
# ============================================================
def main():
    """Run the full ingestion pipeline. Returns the process exit code."""
    # --- Stage 1: PDF type check (before any extraction) ---
    try:
        pdf_type = loaders.check_pdf_type(loaders.PDF_PATH)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1

    # A scanned PDF cannot be processed without OCR, which is out
    # of scope - stop here with a clear explanation.
    if pdf_type == 0:
        print("ERROR: PDF detected as SCAN (no extractable text).")
        print("OCR is outside the scope of this ingestion pipeline.")
        return 1

    # --- Stage 2: open the PDF and extract all articles ---
    try:
        document = loaders.open_pdf()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1

    page_count = document.page_count  # Total pages of the document.
    articles, preamble, rejected_headers = extraction.extract_articles(document)

    # English "Article N" numbers per page (validation only; this
    # text never enters the article records).
    english_per_page = []
    for page_index in range(page_count):
        page = document.load_page(page_index)  # Load the page again.
        english_per_page.append(extraction.english_article_numbers(page))
    document.close()  # Release the PDF file handle.

    # --- Stage 3: calculate every reported value ---
    metrics = compute_metrics(articles, preamble, rejected_headers, english_per_page, page_count)

    # --- Stage 4: run the five validation checks ---
    checks = run_validation(articles, metrics)

    # --- Stage 5: print notes + summary + validation report ---
    print_report(metrics, checks, pdf_type, page_count, rejected_headers)

    # --- Stage 6: write the JSON only when everything passed ---
    all_passed = True
    for passed in checks.values():
        if not passed:
            all_passed = False

    if not all_passed:
        print("Validation FAILED - articles.json was NOT written.")
        return 1

    # Build the output document: provenance + preamble + articles.
    payload = {
        "source_pdf": str(loaders.PDF_PATH),
        "page_count": page_count,
        "preamble": preamble,
        "articles": articles,
    }
    # Create data/processed/ if it does not exist yet.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write UTF-8 JSON with real Arabic characters preserved.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(articles)} articles to {OUTPUT_PATH}")
    return 0


# Entry point when the script is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())
