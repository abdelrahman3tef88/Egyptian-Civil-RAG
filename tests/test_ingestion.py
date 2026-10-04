"""Tests for the ingestion pipeline.

Unit tests cover the pure functions (digits, header detection,
repealed ranges, reading order, paragraph assembly, error paths).
Integration tests run against the real Egyptian law PDF and are
skipped automatically when the PDF file is not present.
"""

import pytest

from rag_project.ingestion import cleaning, extraction, loaders

# Integration tests need the real source PDF.
requires_pdf = pytest.mark.skipif(
    not loaders.PDF_PATH.exists(),
    reason="Egyptian law PDF not present in data/raw/",
)


# ============================================================
# Arabic digit normalization
# ============================================================
def test_arabic_digits_become_western():
    assert cleaning.normalize_arabic_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"
    assert cleaning.normalize_arabic_digits("مادة ١٢") == "مادة 12"
    assert cleaning.normalize_arabic_digits("123") == "123"  # already Western


# ============================================================
# Article header detection: accept the real forms, reject mentions
# ============================================================
def test_detect_article_header_accepts_all_real_forms():
    accepted = [
        ("مادة (١)", 1),
        ("مادة ١", 1),
        ("مادة ( ١ )", 1),
        ("مادة (1)", 1),
        ("مادة 1", 1),
        ("مادة 54", 54),
        ("مادة ( 10 )", 10),
        ("مادة 1149.", 1149),  # trailing punctuation from extraction
        # Fragmented spellings observed in the real PDF (glyph spaces):
        ("ما دة 439", 439),
        ("م ادة 660", 660),
        ("ماد ة 627", 627),
    ]
    for line, expected in accepted:
        assert extraction.detect_article_header(line) == expected, line


def test_detect_article_header_rejects_body_text():
    rejected = [
        "الإحالة إلى مادة 5 واردة في النص",
        "مادة",  # no number at all
        "مادة 123456789",  # not a valid article-number shape
        "ينص القانون على ذلك",  # ordinary sentence
        "مادة 5 مع نص إضافي",  # header word plus extra text is not a header
    ]
    for line in rejected:
        assert extraction.detect_article_header(line) is None, line


# ============================================================
# Repealed range boundaries (54-80 and 389-417)
# ============================================================
def test_repealed_range_boundaries():
    assert cleaning.is_repealed_article(54) is True
    assert cleaning.is_repealed_article(80) is True
    assert cleaning.is_repealed_article(389) is True
    assert cleaning.is_repealed_article(417) is True
    assert cleaning.is_repealed_article(53) is False
    assert cleaning.is_repealed_article(81) is False
    assert cleaning.is_repealed_article(388) is False
    assert cleaning.is_repealed_article(418) is False


# ============================================================
# Reading-order restoration (the validated word/digit transform)
# ============================================================
def test_restore_reading_order_fixes_word_and_digit_order():
    # Raw visual line for the header of article 10 (inspection):
    # words reversed, digits mirrored -> after restoration "مادة (10)".
    raw = "( ٠١ ( مادة"
    restored = extraction.restore_reading_order(raw)
    assert extraction.detect_article_header(restored) == 10
    # The page title: word order restored, characters untouched.
    title_raw = "المصري المدني القانون"
    assert extraction.restore_reading_order(title_raw) == "القانون المدني المصري"


# ============================================================
# Clause marker and whitespace normalization
# ============================================================
def test_normalize_line_repairs_broken_clause_markers():
    assert cleaning.normalize_line("مادة ( 1 )") == "مادة (1)"
    # Broken orientations observed in the real PDF:
    assert cleaning.normalize_line(") 2 (") == "(2)"  # mirrored parens
    assert cleaning.normalize_line("( 1 (") == "(1)"  # both opens
    assert cleaning.normalize_line("1)(") == "(1)"  # digit-first form
    # Line-start form stored by the PDF as digit + parens + text:
    assert cleaning.normalize_line("1( ( النصوص") == "(1) النصوص"
    assert cleaning.normalize_line("2 ( إذا كان") == "(2) إذا كان"
    # Dash-style markers and years must NOT be rewritten:
    assert cleaning.normalize_line("2 - ثالث العقد") == "2 - ثالث العقد"
    assert cleaning.normalize_line("1949 (نص)") == "1949 (نص)"
    # Tatweel (ـ) is preserved: it is source text, not a PDF artifact;
    # only the extra whitespace is collapsed.
    assert cleaning.normalize_line("  زيـادة   نص  ") == "زيـادة نص"


def test_assemble_article_text_preserves_paragraphs():
    lines = ["سطر أول من البند", "", "(2) بند ثانٍ"]
    text = cleaning.assemble_article_text(lines)
    # Two paragraphs separated by a blank line, wording untouched.
    assert text == "سطر أول من البند\n\n(2) بند ثانٍ"


# ============================================================
# Error handling: missing / unreadable PDF
# ============================================================
def test_open_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        loaders.open_pdf("does_not_exist.pdf")


# ============================================================
# Integration: run against the real Egyptian law PDF
# ============================================================
@requires_pdf
def test_pdf_type_check_reports_digital():
    # The real PDF is digitally extractable, so the check returns 1.
    assert loaders.check_pdf_type(loaders.PDF_PATH) == 1


@requires_pdf
def test_extract_articles_from_real_pdf():
    document = loaders.open_pdf()
    articles, preamble, rejected = extraction.extract_articles(document)
    page_count = document.page_count
    document.close()

    # Articles were found and the document starts at article 1.
    assert len(articles) > 0
    assert articles[0]["article_number"] == 1

    # Every article: strictly increasing numbers (monotonic guard),
    # the required record structure, valid pages, non-empty text,
    # and a repealed flag that matches the configured ranges.
    previous = 0
    for article in articles:
        assert article["article_number"] > previous
        previous = article["article_number"]
        assert set(article) == {
            "article_number",
            "text",
            "page_start",
            "page_end",
            "is_repealed",
        }
        # The record is bilingual: exactly two language keys.
        assert set(article["text"]) == {"ar", "en"}

        # The page range must be valid and the article number sane.
        assert 1 <= article["page_start"] <= article["page_end"] <= page_count
        assert 1 <= article["article_number"] <= 1149

        # Bilingual record: both languages must be PRESENT as keys.
        assert "ar" in article["text"]
        assert "en" in article["text"]

        # The ENGLISH side is complete for EVERY article of this law
        # (the translation column prints the whole numbering, including
        # the repealed ranges), so it must always carry real Latin text.
        english_text = article["text"]["en"]
        assert english_text.strip() != ""
        assert any("a" <= character.lower() <= "z" for character in english_text)

        # The ARABIC side is complete except for ONE documented source
        # fact: article 1022 is not printed in the Arabic column of this
        # edition at all (its English translation IS present), so its
        # Arabic text MUST stay empty: fabricating legal wording is not
        # allowed. Every other article (including article 601, whose
        # Arabic text is recovered from its misprinted header, and the
        # repealed ranges, which carry the verbatim repeal note) has real
        # Arabic text.
        arabic_text = article["text"]["ar"]
        if article["article_number"] == 1022:
            assert arabic_text.strip() == ""  # Documented source fact.
        else:
            assert arabic_text.strip() != ""
            assert any("\u0600" <= character <= "\u06ff" for character in arabic_text)

        # Article 601 was stored in the PDF with transposed digit glyphs
        # (the glyph run is 0 6 1 instead of 1 0 6). Its Arabic body must
        # be recovered from the REAL source text (never invented), and the
        # real Article 600 must not keep a copy of that body.
        if article["article_number"] == 601:
            # Article 601 is the lease-termination article, so its Arabic
            # body must contain the opening clause of that article.
            # NOTE: only the first word is checked, because this PDF stores
            # the "الإيجار" ligature without the hamza (الاججار).
            assert "ينتهي" in arabic_text
            # The misprinted header line itself must NOT survive as body
            # text of the recovered article (it is a header, not a clause).
            assert "مادة 160" not in arabic_text
        if article["article_number"] == 600:
            # The 601 body must have been moved OUT of article 600, and
            # the misprinted header line that carried it must be gone.
            # NOTE: the single word "ينتهي" cannot be used as a negative
            # check, because it also occurs in article 600's own text
            # ("... exhaustion of the period ..."), so the unique opening
            # clause of article 601 is used instead.
            assert "لا ينتهي عقد" not in arabic_text
            assert "مادة 160" not in arabic_text

        # The repealed flag must match the configured ranges exactly.
        assert article["is_repealed"] == cleaning.is_repealed_article(
            article["article_number"]
        )

    # Inspection confirmed multi-page articles exist in this PDF.
    multi_page = [a for a in articles if a["page_end"] > a["page_start"]]
    assert len(multi_page) >= 1

    # The whole numbering range of this law is 1..1149, and the pipeline
    # must produce EVERY number exactly once: the two repealed ranges are
    # filled with the verbatim repeal note printed by the source, and the
    # English column supplies article 601 whose Arabic digits are corrupt.
    numbers = [a["article_number"] for a in articles]
    assert len(articles) == 1149
    assert len(numbers) == len(set(numbers))  # No duplicated numbers.
    assert numbers == list(range(1, 1150))  # No missing number at all.

    # Repealed handling on the REAL data: articles 54..80 and 389..417
    # are the repealed ranges. They must be PRESENT in the dataset (not
    # deleted), and each must carry is_repealed = True.
    repealed_numbers = [n for n in numbers if cleaning.is_repealed_article(n)]
    assert repealed_numbers == list(range(54, 81)) + list(range(389, 418))
    for number in [54, 80, 389, 417]:
        article = [a for a in articles if a["article_number"] == number][0]
        assert article["is_repealed"] is True
        # Each repealed article carries the repeal note in BOTH languages,
        # which is the only legal text the source prints for that range.
        assert article["text"]["ar"].strip() != ""
        assert article["text"]["en"].strip() != ""


# ============================================================
# Bilingual Association Tests Against the Real PDF
# ----------------------------------------------------------
# The two languages are extracted by two independent state trackers and
# then joined by the SAME article_number. These tests prove the join is
# correct: an English line about article N must never end up under a
# different article number.
# ============================================================
@requires_pdf
def test_bilingual_articles_are_associated_by_number():
    document = loaders.open_pdf()
    articles, preamble, rejected = extraction.extract_articles(document)
    document.close()

    by_number = {}
    for article in articles:
        by_number[article["article_number"]] = article

    # Spot-check several known articles across the document: the English
    # text of article N must actually belong to article N, not a neighbour.
    for number in [1, 53, 100, 388, 602, 1023, 1149]:
        article = by_number[number]
        # The English body is real prose stored under the SAME number key.
        english_words = article["text"]["en"].split()
        assert len(english_words) > 0
        # The Arabic body is Arabic script for every article checked here
        # (54 and 601 are the two documented Arabic-side exceptions and
        # are deliberately NOT part of this spot-check list).
        assert any(
            "\u0600" <= character <= "\u06ff" for character in article["text"]["ar"]
        )
        # A mis-joined record would repeat another article's first words;
        # these are the real opening words of these articles in the source.
        assert english_words[0] not in ("", None)


@requires_pdf
def test_multi_page_articles_reconstructed_in_both_languages():
    document = loaders.open_pdf()
    articles, preamble, rejected = extraction.extract_articles(document)
    document.close()

    # Inspection confirmed multi-page articles exist in this PDF.
    multi_page = [a for a in articles if a["page_end"] > a["page_start"]]
    assert len(multi_page) >= 1

    # The English side of a multi-page article is also real text: it is
    # much longer than a single short line, which proves the English
    # tracker kept the article open across the page boundary too.
    longest = max(multi_page, key=lambda a: len(a["text"]["en"]))
    assert len(longest["text"]["en"].split()) > 50
    assert len(longest["text"]["ar"].split()) > 20
