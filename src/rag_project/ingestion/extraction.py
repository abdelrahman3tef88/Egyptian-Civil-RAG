"""
الملف ده هو Extraction + Reconstruction layer.

يعني بياخد الـPDF الثنائي اللغة:

┌───────────────────────┬───────────────────────┐
│ English column        │ Arabic column         │
│ Article 1             │ مادة 1                │
│ ...                   │ ...                   │
└───────────────────────┴───────────────────────┘

ويحوله إلى records بالشكل:

{
    "article_number": 1,
    "text": {
        "ar": "...",
        "en": "..."
    },
    "page_start": 1,
    "page_end": 1,
    "is_repealed": False
}

يعني الـpipeline هنا:

PDF
 ↓
فصل Arabic / English columns
 ↓
إصلاح Reading Order للعربي
 ↓
اكتشاف Article Headers
 ↓
تجميع المادة حتى لو على أكثر من صفحة
 ↓
مطابقة Arabic Article 10 مع English Article 10
 ↓
إضافة repealed status
 ↓
JSON records
"""

# Bilingual PDF extraction: two-column split, reading order, article reconstruction.
#
# Responsibilities of this module:
#   - extract BOTH columns of each page using dynamic coordinates
#     (the PDF is bilingual: Arabic right, English left)
#   - restore the reading order of raw Arabic lines (see below)
#   - detect real Arabic article headers (مادة + valid number)
#   - detect real English article headers (Article + valid number)
#   - rebuild complete articles across page boundaries with a simple
#     state tracker, one per language, then merge the two sides by the
#     SAME article_number into one bilingual record per article:
#         {
#             "article_number": 1,
#             "text": {"ar": "...", "en": "..."},
#             "page_start": 1,
#             "page_end": 1,
#             "is_repealed": False
#         }
#
# Neither language's text is ever paraphrased, translated, or rewritten:
# the English column is the translation printed in the source PDF itself.

import re

import pymupdf

# Cleaning owns digit/whitespace/paren normalization and the
# repealed ranges; extraction composes those safe helpers.
from rag_project.ingestion import cleaning


# ============================================================
# Article Header Detection Pattern
# ----------------------------------------------------------
# A real article header is a line that consists of the word
# "مادة" plus a valid article number, allowing the extraction
# variants observed in this PDF:
#     مادة 1        مادة (1)      مادة ( 1 )
#     مادة ١        مادة (١)      plus stray spaces/parens
#
# The pattern is ANCHORED to the whole line ($ at the end), so an
# ordinary sentence that merely mentions "مادة 5" inside body text
# can never be mistaken for a header. Digits are already Western
# by the time we match, because the line was normalized first.
#
# Breakdown of the regex:
#   ^\s*م\s*ا\s*د\s*ة  -> line STARTS with the word مادة (see note)
#   \s*[()\s]*         -> optional spaces / parentheses before number
#   (\d{1,4})          -> the article number itself (1..4 digits)
#   \s*[()\s]*         -> optional spaces / parentheses after number
#   [.:،]?\s*$         -> optional trailing punctuation, then END
#
# NOTE on the spaced letters (evidence from the real PDF): 13 headers
# are stored with spurious spaces INSIDE the word مادة, in the forms
# "ما دة 439", "م ادة 660", "ماد ة 627". Allowing \s* between the
# four letters matches these fragmented forms and the normal form
# with ONE pattern, while the whole-line anchor still rejects body
# text that merely mentions the word.
# دي أهم Regex لاكتشاف:
#
# مادة 1
# مادة (1)
# مادة ( 1 )
# مادة ١
# ما دة 439
# م ادة 660
# ماد ة 627
# ليه محتاجينها؟
#
# لأننا لازم نفرق بين:
#
# مادة 10
#
# اللي هي header حقيقي،
#
# وبين:
#
# ويحدد القانون مادة 10 من ...
#
# اللي دي مش header.
# ============================================================
ARTICLE_HEADER_PATTERN = re.compile(
    r"^\s*م\s*ا\s*د\s*ة\s*[()\s]*(\d{1,4})\s*[()\s]*[.:،]?\s*$"
)

# The English column writes headers as "Article 12". Used ONLY to
# cross-check our Arabic article numbers during validation.
# دي بتدور على:
#
# Article 10 مثلا
#
# في أي مكان داخل السطر.
#
# مهم: دي مش بالضرورة بتحدد إن ده header.
#
# هي بتقول فقط:
#
# لاقيلي أي occurrence لـ Article N.
#
# مستخدمة للـvalidation.
#
# مثلاً:
#
# This is provided under Article 554.
#
# هتلاقي:
#
# 554
#
# لكن ده مش معناه إن 554 header.
# تستخدم لل validation فقط
ENGLISH_ARTICLE_PATTERN = re.compile(r"\bArticle\s+(\d+)\b")

# A real English article header is a WHOLE line of the form
# "Article 12" (an optional trailing dot / parenthesis is tolerated).
# The anchor is what separates a real header from a cross-reference
# inside a sentence such as "provided for in Article 563".
# نفس الكلام للي في ARTICLE_HEADER_PATTERN بس بالانجلش بدل "مادة  هتكون "Article
ENGLISH_HEADER_PATTERN = re.compile(r"^\s*Article\s+(\d{1,4})\s*[\.\):]?\s*$")


# ============================================================
# Restore the Reading Order of a Raw Arabic Line
# ----------------------------------------------------------
# WHY the raw extraction is wrong (proven by inspection of this
# exact PDF, cross-checked against the English column):
#
#   1. PyMuPDF emits the WORDS of an RTL line in visual order,
#      i.e. word order is REVERSED:
#          raw:      "المصري المدني القانون"   (wrong)
#          correct:  "القانون المدني المصري"
#      but the CHARACTERS inside each word are already logical.
#
#   2. Digit runs are mirrored by the same bidi effect, so the
#      header of article 10 comes out as "01".
#
#   3. Clause parentheses come out with broken orientation
#      (handled later by cleaning.canonicalize_clause_markers).
#
# WHY NAIVE CHARACTER REVERSAL MUST NOT BE USED:
#   Reversing every character of the line (line[::-1]) was tested
#   and REJECTED: it flips the characters inside every word, which
#   destroys the Arabic words themselves (a lexicon test scored
#   -6741 mirrored-word hits versus +6741 correct-word hits for
#   the word-order fix implemented below).
#
# THE VALIDATED TRANSFORMATION (simple, deterministic):
#   a) reverse the ORDER of the whitespace-separated words
#   b) reverse each DIGIT RUN inside every word (fixes 01 -> 10)
#   c) leave all other characters exactly as extracted
# After this, header detection scores ~86% agreement with the
# English column ground truth (rest is page-boundary tolerance).
# def _reverse_text(match):
#
# دي function صغيرة جدًا.
#
# وظيفتها:
# تعكس sequence الأرقام فقط.
#
# مثلاً:
#
# 01
#
# تصبح:
# 10
#
# return match.group(0)[::-1]
# group(0) = النص اللي الـregex لاقاه وماسكه
#
# و:
#
# [::-1]
# بيعكسه.
#
# ليه؟
#
# لأن الـPDF عند استخراج العربي بيطلع أحيانًا الأرقام mirrored بسبب الـRTL/BiDi.
# ============================================================
def _reverse_text(match):
    """Reverse the matched digit run (used as an re.sub replacement)."""
    # re.sub passes a match object; group(0) is the matched digits.
    return match.group(0)[::-1]


# وظيفتها:
# تصلح ترتيب line عربي طالع من الـPDF بترتيب بصري غلط.
#
# مثلاً الـPDF ممكن يطلع:
# المصري المدني القانون
#
# لكن الصحيح:
# القانون المدني المصري
def restore_reading_order(line):
    """Turn one raw visual-order Arabic line into logical reading order."""
    # Split the line into words (whitespace split keeps it simple).
    # نقسم الجمله الي كلمات
    words = line.split()

    # The visual order lists the LAST logical word first, so we
    # walk the word list backwards to recover logical order.
    restored_words = []
    # نلف علي كلمه كلمه بس بالعكس "reversed" يعني القانون ثم المدني ثم المصري
    for word in reversed(words):
        # Reverse each digit run inside the word.
        # Python's \d also matches Arabic-Indic digits, and
        # cleaning.normalize_arabic_digits converts them later.
        # تتشيك كل كلمه هل فيها digits او ارقام متلغبط لو اة طبق الفانكشن _reverse_text علي الكلمه
        word = re.sub(r"\d+", _reverse_text, word)
        # ثم خزن الكلمه في restored_words
        restored_words.append(word)

    # Rejoin with single spaces (extra spaces are cleaned later).
    return " ".join(restored_words)


# ============================================================
# Extract the Arabic Right Column From Each PDF Page
# ----------------------------------------------------------
# The PDF page is bilingual and laid out as two columns:
#     RIGHT column -> Arabic (the legal text we need)
#     LEFT  column -> English (translation, validation only)
#
# Naive page-wide extraction mixes both columns and corrupts the
# reading order. We therefore clip extraction to the right half
# of the page using DYNAMIC coordinates:
#     mid_x = page_width / 2
# (inspection measured mid_x = 297.66 on this A4 PDF, but it is
# always recomputed from the page itself - never hardcoded).
#
# sort=True asks PyMuPDF to order the extracted lines by their
# Y coordinate, which preserves the top-to-bottom reading order
# of the column. Each line is then passed through the validated
# reading-order restoration.
# ============================================================
def extract_arabic_lines(page):
    """Return the cleaned Arabic (right column) lines of one page, top to bottom."""
    # Get the actual width of this page (dynamic, no fixed values).
    # بتجيب وتحدد   عرض الصفحة الحقيقي.
    page_width = page.rect.width
    # The horizontal midpoint separates Arabic (right) from English (left).
    # تحديد المنتصف
    # تاخد عرض الصفحه الحقيقي دة وتقسمه علي اتنين
    mid_x = page_width / 2
    # Clip rectangle covering ONLY the right half of the page.
    # يعني استخرج كل العمود اليمين فقط للي هو العربي
    # معايا mid_x عرض الصفحه ومعايا 0 دة الجزء اليمين ومعايا ال height بتاع الصفحه كلها
    # كدة ال right_column اصبح clip rectangle column اقدر اخد منه ال text
    right_column = pymupdf.Rect(mid_x, 0, page_width, page.rect.height)
    # Extract text inside the clip, sorted top-to-bottom by Y coordinate.
    text = page.get_text("text", clip=right_column, sort=True)

    lines = []
    # ناخد النص  ونحوله الي lines + نلف عليها واحد واحد
    for raw_line in text.splitlines():
        # Fix the visual word/digit order of this raw line.
        # اصلاح كل line وتطبق عليه restore_reading_order
        restored = restore_reading_order(raw_line)
        # Keep empty lines as paragraph markers (later cleaning
        # collapses repeated whitespace but never deletes them
        # before assembly), and clean everything else.
        # لو ال line is empty
        # هنخزنه لأن الـblank line ممكن تمثل paragraph boundary.
        if restored.strip() == "":
            lines.append("")
        else:
            # لو ال line is not empty طبق ليه normalize_line + خزنه
            lines.append(cleaning.normalize_line(restored))
    return lines


# ============================================================
# Extract the English Left Column (VALIDATION ONLY)
# ----------------------------------------------------------
# The English column is NEVER part of the article records. It is
# extracted only so the validation step can cross-check our
# Arabic article numbers against the "Article N" headers of the
# translation (English text is stored in logical order already,
# so no reading-order restoration is applied to it).
# ============================================================
def extract_english_lines(page):
    """Return the cleaned English (left column) lines of one page.

    The lines are rebuilt from the PDF WORDS grouped by their own
    (block, line) numbers, instead of the page text stream. Reason:
    the plain text stream sometimes merges the short header
    "Article 5" into the end of the previous line
    ("...responsibleArticle5 for prejudice..."), which lost 74
    English article headers. The word coordinates keep every header
    on its own line, in the correct left-to-right order.
    """
    # The horizontal midpoint separates the two columns (dynamic).
    mid_x = page.rect.width / 2
    # Get every word on the page with its coordinates and line IDs.
    #     ليه مش بنستخدم page.get_text("text") وخلاص؟
    #
    # لأن الـPDF أحيانًا يعمل:
    # responsibleArticle5 for prejudice
    #
    # بدل:
    #
    # responsible
    # Article 5
    #
    # فـplain text stream ممكن يدمج الـheader مع الـbody.
    #
    # عشان كده هنا بنستخدم:
    # words = page.get_text("words")
    # اللي بيرجع الكلمات مع coordinates.
    # علشان كدة هنستخرج كلمه كلمه في الانجلش وبعدها نجمعهم تاني
    words = page.get_text("words")

    # Group the left-column words by their PDF line number.
    lines_by_id = {}
    for word in words:
        # Keep only words that start inside the left (English) column.
        # لو الكلمه اكبر من mid_x يبقي الكلمه عربي ف تجاهلها
        if word[0] >= mid_x:
            continue
        # Skip Arabic glyphs that sit close to the column edge.
        # زيادة تاكيد لو اول حرف عربي   تجاهله
        first_letter = word[4][0]
        if "\u0600" <= first_letter <= "\u06ff":
            continue
        # The (block number, line number) pair identifies one PDF line.
        #         تحديد line
        # line_id = (word[5], word[6])
        #
        # الـPDF بيدي كل word معلومات عن:
        #
        # block
        # line
        #
        # فبنستخدمهم عشان نعرف:
        # الكلمات دي كانت في نفس السطر ولا لا ؟
        line_id = (word[5], word[6])
        # تجميع كلمات كل line
        # line 1 → words
        # line 2 → words
        # line 3 → words
        # كل كلمه بستخرجها بدبها id هي تبع ال line كذا
        lines_by_id.setdefault(line_id, []).append(word)

    lines = []
    # Sort the PDF lines by their Y coordinate: top to bottom.
    #     ترتيب ال lines حسب coordinate y
    #
    #     أعلى الصفحة
    #    ↓
    # السطر الأول
    # السطر الثاني
    # السطر الثالث
    #    ↓
    # أسفل الصفحة
    for line_id in sorted(lines_by_id, key=lambda k: lines_by_id[k][0][1]):
        # Sort the words of this line by X: left to right.
        # ترتيب الكلمات داخل ال line    بنرتب حسب X coordinate.
        # ← left       right →
        words_in_line = sorted(lines_by_id[line_id], key=lambda w: w[0])
        parts = []
        # تكوين ال line مجمه بالكلمات ومرتبه
        for word in words_in_line:
            parts.append(word[4])  # Collect the text of this word.
        # Join the words with single spaces.
        line = " ".join(parts)
        # Apply only safe cleaning (invisible chars + spaces).
        # تنظيف خاص بالانجلش lines فقط
        lines.append(cleaning.clean_english_line(line))
    return lines


# دي بتجيب أرقام الـArticles الموجودة في صفحة English.
#
# مثلاً:
#
# Article 10
# ...
# Article 11
# ...
# Article 12
#
# ترجع:
#
# {10, 11, 12}
#
# استخدامها الأساسي:
#
# validation / cross-check.
#
# يعني نقدر نقارن English مع Arabic.
def english_article_numbers(page):
    """Collect the set of 'Article N' numbers found on one English page."""
    numbers = set()
    for line in extract_english_lines(page):
        # Find every "Article 123" mention on this line.
        for match in ENGLISH_ARTICLE_PATTERN.finditer(line):
            numbers.add(int(match.group(1)))
    return numbers


# ============================================================
# Detect One Article Header Line
# ----------------------------------------------------------
# Returns the article number as an int when the line REALLY is a
# header, otherwise None. The line is normalized first so that
# Arabic-Indic digits, extra spaces, and broken parentheses are
# all handled by cleaning before the anchored pattern runs.


# ============================================================
# Detect an English Header That Shares Its Line with Body Text
# ----------------------------------------------------------
# On 11 pages of this PDF the English column prints the header and the
# first body line on the SAME PDF line, e.g.
#     "Article 277 If the option belongs to the debtor ..."
# The plain text stream keeps them together, so the whole-line pattern
# above cannot match. The line STARTS with "Article N" here, which is
# safe: an in-sentence cross-reference is never at the start of a line
# (it reads "in accordance with Article 554").
# Returns (article_number, remaining_text) or (None, original_line).
# ============================================================
# A REAL English article header is "Article 12". The group is written
# this way because the PDF sometimes clips the first character of the
# word at the column edge, and we then see "rticle 452". The lookarounds
# keep it a standalone word: a cross-reference inside a sentence
# ("in accordance with Article 554") can never match because the header
# must START the line, and the repeal note "* Articles 54-80 have been
# repealed" is excluded because the word must not be followed by "s"
# (that line is a repeal note, not an article header).
ENGLISH_INLINE_HEADER_PATTERN = re.compile(
    r"^\s*(?<![A-Za-z])[Aa]?rticle(?!s)\s*(\d{1,4})\b\s*(.*)$"
)


# أحيانًا English بيطلع:
#
# Article 277 If the option belongs to the debtor...
#
# يعني الـheader والـbody في نفس السطر.
# الـnormal header detector مش هيعرفه، لأن السطر مش:
#
# Article 277
#
# فقط.
#
# الـfunction ترجع:
# (number, rest_of_line)
#
# مثلاً:
#
# (
#     277,
#     "If the option belongs to the debtor..."
# )
#
# وبالتالي:
#
# 277 → header
# باقي الجملة → body
#
# فالفانكشن بتجهز للnormal header detector رقم ال article header بشكل منفصل بحيث يقدر يشتغل عليه علطول
# وبالتالي اال normal header detector يعرف ان احنا حاليا شغالين علي article 277
def split_english_inline_header(line):
    """Return (number, rest_of_line) for an inline English header line."""
    line = cleaning.clean_english_line(line)
    match = ENGLISH_INLINE_HEADER_PATTERN.match(line)
    if match:
        # Group 1 = the number, group 2 = the body text on the same line.
        return int(match.group(1)), match.group(2)
    return None, line


# ============================================================
# Detect One Arabic Article Header Line
# ============================================================
# Header Detector للعربي: تأخذ سطرًا واحدًا وتسأل: هل هذا السطر يمثل بداية مادة جديدة ولا لا ؟
# وإذا كان كذلك، ما رقم المادة؟
# ============================================================
def detect_article_header(line):
    """Return the article number of a header line, or None if it is not one."""
    # Normalize digits/spaces/parens (idempotent, safe for legal text).
    line = cleaning.normalize_line(line)
    # Anchored match: the WHOLE line must be "مادة + number".
    # لو ال line للي شغال عليه بيمثل مثلا مادة 5
    # اذن هو بيماتش match
    # يبقي هات واستخرج رقم المادة دة    5
    match = ARTICLE_HEADER_PATTERN.match(line)
    if match:
        # Group 1 is the number, e.g. "54" -> 54 as an integer.
        return int(match.group(1))
    # Not a header: ordinary legal text (or a mention of مادة).
    # لو مش بيماتش not match
    # اذن دة body line عادي
    return None


# ============================================================
# Detect One English Article Header Line
# ----------------------------------------------------------
# The English column writes a header as its own line: "Article 12".
# The pattern is ANCHORED to the whole line, so an in-sentence
# cross-reference such as "provided for in Article 563" is never
# mistaken for a header.
# ============================================================
def detect_english_article_header(line):
    """Return the article number of an English header line, else None."""
    # Clean the line first (safe representation fixes only).
    line = cleaning.clean_english_line(line)
    # Anchored match: the WHOLE line must be "Article + number".
    match = ENGLISH_HEADER_PATTERN.match(line)
    if match:
        # Group 1 is the number, e.g. "54" -> 54 as an integer.
        return int(match.group(1))
    # Not a header: it is body text or an in-sentence reference.
    return None


# ============================================================
# Build One Finished Article Record (BILINGUAL)
# ----------------------------------------------------------
# Converts the collected Arabic and English lines of ONE article
# into the target structure required by the later stages:
#     {
#         "article_number": 1,
#         "text": {"ar": "...", "en": "..."},
#         "page_start": 1,
#         "page_end": 2,
#         "is_repealed": False
#     }
#
# Arabic and English are matched by the SAME article_number, which
# is the stable key used by the Arabic header ("مادة N") and the
# English header ("Article N") of the same legal article.
# page_end is derived from the LAST line that actually contains
# text, so a multi-page article reports its true ending page.
# ============================================================
def _build_article(
    article_number, ar_entries, en_entries, page_start, english_text=None
):
    """Create one structured bilingual article record from its lines."""
    # Separate the two languages so no text can cross over.
    ar_lines = []
    for page_number, line in ar_entries:
        ar_lines.append(line)  # Keep the text, drop the page tag.

    en_lines = []
    for page_number, line in en_entries:
        en_lines.append(line)  # Keep the text, drop the page tag.

    # Find the last page that contributed real ARABIC text
    # (= page_end, the source page range of the legal article).
    page_end = page_start
    # هتلوب في كل صفحه علي ال lines بس بالعكس
    # اذن انا هاخد اخر line في الصفحه واتشيك
    for page_number, line in reversed(ar_entries):
        # لو ال line is not empty
        # يبقى دي آخر صفحة فيها محتوى حقيقي للمادة. وتكون حددت ال page end
        if line != "":
            page_end = page_number  # Last page with actual content.
            break

    # Join the Arabic lines into paragraph-structured legal text.
    arabic_text = cleaning.assemble_article_text(ar_lines)

    # Join the English lines the same safe way. The English clause
    # markers in this PDF are plain numbers, so we only rely on the
    # blank lines of the PDF (exactly like the Arabic side).
    if english_text is None:
        english_text = cleaning.assemble_article_text(en_lines)

    # The final record: bilingual text + the repealed marker
    # (repealed articles are never deleted, only marked).
    return {
        "article_number": article_number,
        "text": {
            "ar": arabic_text,
            "en": english_text,
        },
        "page_start": page_start,
        "page_end": page_end,
        "is_repealed": cleaning.is_repealed_article(article_number),
    }


# ============================================================
# Append One Text Line to the Currently Open Article
# ----------------------------------------------------------
# Before the first header, a line belongs to the preamble (the
# issuing decree / title). After the first header it belongs to the
# open article, together with its page number so page_end can grow.
# ============================================================
# دي function مساعدة.
#
# قبل أول Article header:
#
# Egyptian Civil Code
# Presidential Decree...
#
# ده اسمه:
# preamble
def _add_text_line(page_number, line, current_number, current_entries, preamble_lines):
    """Give one text line to the open article, or to the preamble."""
    # لو ال line للي شغالين عليه دلوقتي للي هو اول line في ال pdf كله للي
    # للي هو قبل اول header
    # ة يسمي preamble ليس ضمن ال article نفسه للي امسه open article
    if current_number is None:
        # No article opened yet: this is preamble text (title, decree).
        preamble_lines.append(line)
    else:
        # Inside an article: remember the page so page_end can update.
        current_entries.append((page_number, line))


# قبل أول مادة
#    ↓
# preamble
#
# بعد أول مادة
#    ↓
# current article


# ============================================================
# Reconstruct Complete Articles Across All Pages
# Articles
# ----------------------------------------------------------
# This is the simple state tracker (no framework, no classes):
#
#     - current_ar_number / current_ar_entries / current_ar_start
#     - current_en_number / current_en_entries / current_en_start
#
# Both columns are extracted in the same page loop, and each side is
# reconstructed independently in its own simple state tracker:
#   - a new valid header     -> finalize the previous one, open a new one
#   - anything else          -> append to the currently open article
#   - an end of page         -> does NOT close the article (multi-page
#                                articles stay open and keep collecting
#                                on the next page)
#   - the end of the document -> finalize the last open article
#
# MONOTONIC GUARD: a candidate header only opens a new article when
# its number moves FORWARD (candidate > current). A backwards or
# repeated number is a cross-reference inside body text; the line
# is kept as body text and recorded as a diagnostic - never lost,
# never allowed to split an article incorrectly.
# ============================================================
# الفكرة كلها أن الـPDF مش بيقول لنا صراحة:
#
# "السطور دي كلها Article 277"
#
# إحنا لازم نستنتج ده من الـheaders، ونحافظ على حالة الـArticle الحالية أثناء المرور على كل الصفحات.
#
# 1. الفكرة العامة
#
# عندنا سطور بالشكل:
#
# Page 10:
# مادة (54)
# يلتزم الطرف الأول...
# ويجب عليه...
#
# Page 11:
# ولا يجوز له...
# مادة (55)
# يجوز للطرف الثاني...
#
# الـfunction تمشي عليهم بالترتيب، وتستخدم state:
#
# current_number
# current_entries
# current_start_page
#
# يعني دائمًا عندها سؤال:
#
# أنا حاليًا بجمع أي Article؟
#
# مثلاً:
#
# current_number = 54
#
# معناه:
#
# أنا حاليًا داخل Article 54.
#
# 2. بداية الـfunction
# def _collect_articles(pages_lines, is_english):
#
# بتستقبل:
#
# pages_lines
#
# قائمة الصفحات، وكل صفحة فيها رقم الصفحة والسطور:
#
# [
#     (1, ["مادة (1)", "النص...", "..."]),
#     (2, ["تكملة النص...", "مادة (2)", "..."]),
# ]
# is_english
#
# يحدد إحنا بنجمع:
#
# Arabic
#
# ولا:
#
# English
#
# لأن الـheader detector مختلف.
#
# 3. collected
# collected = {}
#
# دي النتيجة النهائية.
#
# هتكون تقريبًا:
#
# {
#     54: (entries, 10),
#     55: (entries, 11),
#     56: (entries, 12)
# }
#
# يعني:
#
# Article 54
#     ↓
# entries الخاصة بيه
#     ↓
# بدأ في page 10
# 4. rejected
# rejected = []
#
# دي للـdiagnostics.
#
# لو قابلنا مثلًا:
#
# Article 901 has been delivered to him.
#
# لكن إحنا حاليًا في:
#
# Article 884
#
# فممكن الرقم 901 يكون cross-reference مش Header حقيقي.
#
# الكود يحتفظ بالسطر ويسجله في:
#
# rejected
#
# علشان نقدر نراجعه بعدين.
#
# 5. الـState
#
# عندنا:
#
# current_number = None
#
# في البداية:
#
# مفيش Article مفتوحة.
#
# ثم:
#
# current_entries = []
#
# دي هتحتوي body الخاص بالـArticle الحالية.
#
# و:
#
# current_start_page = 0
#
# صفحة بداية الـArticle.
#
# 6. المرور على الصفحات
# for page_number, lines in pages_lines:
#
# يعني:
#
# هات كل صفحة بالترتيب.
#
# مثلاً:
#
# page_number = 127
# lines = [...]
#
# وبعدها:
#
# for index, line in enumerate(lines):
#
# يمشي على كل line.
#
# index مهم جدًا بعدين في الـEnglish lookahead.
#
# 7. الـblank line
# if line == "":
#     if current_number is not None:
#         current_entries.append((page_number, line))
#     continue
#
# لو السطر فاضي:
#
# ""
#
# مش معناها إن الـArticle انتهت.
#
# دي ممكن تكون مجرد paragraph separator.
#
# فلو إحنا داخل:
#
# current_number = 277
#
# نضيف الـblank line للـentries:
#
# current_entries.append((page_number, ""))
#
# ثم:
#
# continue
#
# يعني:
#
# خلصنا التعامل مع السطر ده، روح للسطر اللي بعده.
#
# 8. Detect الـHeader
#
# هنا بيحدد هل السطر Header أم لا.
#
# rest_of_line = ""
#
# دي مهمة للـEnglish.
#
# لأن ممكن يكون:
#
# Article 277 If the option belongs to the debtor...
# لو English
# if is_english:
#     header_number = detect_english_article_header(line)
#
# أولًا يجرب الـnormal detector.
#
# لو لم يجد:
#
# if header_number is None:
#
# يجرب:
#
# header_number, rest_of_line = split_english_inline_header(line)
#
# فتتحول:
#
# Article 277 If the option belongs to the debtor...
#
# إلى:
#
# header_number = 277
# rest_of_line = "If the option belongs to the debtor..."
# 9. لو Arabic
# else:
#     header_number = detect_article_header(line)
#
# مثلاً:
#
# مادة (54)
#
# ترجع:
#
# header_number = 54
#
# ولو:
#
# ويجب على الطرف الأول...
#
# ترجع:
#
# header_number = None
# 10. أهم جزء: الـMonotonic Guard
#
# بعد ما نعرف رقم الـheader:
#
# if header_number is not None:
#
# نسأل:
#
# هل الرقم الجديد أكبر من الـArticle الحالية؟
#
# if current_number is None or header_number > current_number:
#
# مثلاً:
#
# current_number = 54
# header_number = 55
#
# ده منطقي:
#
# 54 → 55
#
# إذن:
#
# Header حقيقي غالبًا.
#
# لكن:
#
# current_number = 54
# header_number = 40
#
# ده رجوع للخلف:
#
# 54 → 40
#
# غالبًا مش Article جديدة، وإنما cross-reference.
#
# 11. ليه الـMonotonic Guard مهم؟
#
# تخيل:
#
# مادة (54)
# النص...
# Article 20 mentioned in this provision.
# النص...
# مادة (55)
#
# لو parser اعتبر:
#
# Article 20
#
# Header حقيقي، هيعمل:
#
# 54 → 20
#
# وهيبوظ الـstate.
#
# عشان كده:
#
# header_number > current_number
#
# شرط أساسي.
#
# 12. لكن الـEnglish عنده مشكلة إضافية
#
# هنا الجزء المعقد:
#
# if is_english and current_number is not None:
#
# ليه؟
#
# لأن الـEnglish ممكن يكون فيه:
#
# Article 901 has been delivered to him.
#
# داخل Article 884.
#
# وفي نفس الصفحة بعده:
#
# Article 885
#
# لو اعتبرنا 901 Header:
#
# 884 → 901
#
# هنقفز فوق:
#
# 885
# 886
# 887
# ...
# 900
#
# وده خطأ كبير.
#
# 13. الـLookahead
#
# الكود يبص لقدام:
#
# for later_line in lines[index + 1 :]:
#
# يعني:
#
# بص على السطور اللي بعد السطر الحالي في نفس الصفحة.
#
# مثلاً إحنا عند:
#
# Article 901 has been delivered...
#
# والـcurrent:
#
# 884
#
# بعده موجود:
#
# Article 885
#
# فيحسب:
#
# later_number = 885
#
# ثم:
#
# current_number < later_number < header_number
#
# يعني:
#
# 884 < 885 < 901
#
# صحيح.
#
# إذن:
#
# smaller_follows = True
# 14. النتيجة
#
# لو:
#
# smaller_follows:
#
# نعمل:
#
# header_number = None
#
# يعني:
#
# اعتبر Article 901 مش Header.
#
# بل body text.
#
# وبالتالي السطر:
#
# Article 901 has been delivered to him.
#
# يدخل في:
#
# current_entries
#
# بدل ما يفتح Article 901.
#
# ثم بعده:
#
# Article 885
#
# يُعتبر Header حقيقي.
#
# فتبقى السلسلة:
#
# 884
#  ↓
# 885
#
# وليس:
#
# 884
#  ↓
# 901 ❌
# 15. لو Header حقيقي، نعمل إيه؟
#
# لو الرقم valid:
#
# 54 → 55
#
# أول حاجة:
#
# if current_number is not None:
#     collected[current_number] = (
#         current_entries,
#         current_start_page
#     )
#
# يعني:
#
# اقفل الـArticle القديمة واحفظها.
#
# مثلاً:
#
# current_number = 54
#
# فتتحفظ:
#
# collected[54] = (
#     current_entries,
#     10
# )
# 16. افتح Article جديدة
#
# بعد كده:
#
# current_number = header_number
#
# مثلاً:
#
# current_number = 55
#
# ثم:
#
# current_entries = []
#
# يعني:
#
# ابدأ body جديدة من الصفر.
#
# ثم:
#
# current_start_page = page_number
#
# يعني:
#
# Article 55 بدأت في الصفحة الحالية.
#
# 17. لو الـHeader والـBody في نفس السطر
#
# دي نقطة مهمة جدًا للـEnglish.
#
# لو السطر:
#
# Article 277 If the option belongs to the debtor...
#
# فالـsplit أعطانا:
#
# header_number = 277
# rest_of_line = "If the option belongs to the debtor..."
#
# بعد فتح Article 277:
#
# if rest_of_line.strip() != "":
#     current_entries.append(
#         (page_number, rest_of_line)
#     )
#
# فتصبح:
#
# Article 277
#     ↓
# If the option belongs to the debtor...
#
# يعني الـheader نفسه لا يدخل الـbody، لكن الجزء الذي بعده يدخل.
#
# 18. ليه continue؟
# continue
#
# مهمة جدًا.
#
# لأننا بعد ما اكتشفنا Header وعالجناه، مش عايزين السطر كله يتضاف مرة ثانية كـbody.
#
# بدون continue ممكن يحصل:
#
# Article 277
# Article 277 If the option...
#
# داخل الـentries.
#
# لكن continue تقول:
#
# خلاص، عالجنا السطر كـHeader، روح للسطر التالي.
#
# 19. لو الرقم Backward أو Repeated
#
# الجزء:
#
# elif current_number is None or header_number <= current_number:
#
# مثلاً:
#
# current_number = 277
# header_number = 100
#
# أو:
#
# current_number = 277
# header_number = 277
#
# هنا مش هنفتح Article جديدة.
#
# بل:
#
# rejected.append((page_number, line))
#
# نسجلها كـdiagnostic.
#
# لكن لاحظ النقطة المهمة جدًا: السطر مش بيضيع.
#
# بعد الـif/elif، الكود يوصل إلى:
#
# if current_number is not None:
#     current_entries.append((page_number, line))
#
# فيضيف السطر نفسه إلى الـbody.
#
# إذن:
#
# Article 277
#     ↓
# "Article 100 is also mentioned..."
#
# يبقى داخل Article 277.
#
# وده بالضبط معنى التعليق:
#
# never lost, never allowed to split an article incorrectly
#
# 20. النص العادي
#
# لو السطر مش Header أصلاً:
#
# The debtor shall...
#
# فيوصل إلى:
#
# if current_number is not None:
#     current_entries.append((page_number, line))
#
# فيضاف للـArticle الحالية.
#
# مثلاً:
#
# Article 277
#     ↓
# If the option belongs...
#     ↓
# The debtor shall...
#     ↓
# The creditor may...
#
# كلهم في:
#
# current_entries
# 21. أهم نقطة: نهاية الصفحة لا تقفل الـArticle
#
# ودي من أهم أفكار الكود.
#
# افترض:
#
# Page 127
# Article 277
# If the option belongs...
# The debtor may...
#
# انتهت الصفحة.
#
# هل نعمل:
#
# current_number = None
#
# لا.
#
# نفضل:
#
# current_number = 277
#
# ثم الصفحة التالية:
#
# Page 128
# ...
# The creditor may...
# The obligation ends...
# Article 278
#
# فالـ... في Page 128 تضاف إلى:
#
# Article 277
#
# إلى أن يظهر:
#
# Article 278
#
# وقتها فقط:
#
# 277 → finalize
# 278 → open
#
# وده اللي يسمح بإعادة بناء Article ممتدة على أكثر من صفحة.
#
# 22. نهاية الـDocument
#
# في النهاية:
#
# if current_number is not None:
#     collected[current_number] = (
#         current_entries,
#         current_start_page
#     )
#
# ليه؟
#
# لأن آخر Article في الملف مفيش Header بعدها يخبرنا:
#
# اقفل الـArticle.
#
# مثلاً آخر الملف:
#
# Article 100
# The debtor shall...
# The creditor may...
#
# خلص الـPDF.
#
# لازم إحنا نقول:
#
# خلاص، Article 100 انتهت.
#
# فنعمل finalization يدوي.
#
# 23. وفي النهاية
# return collected, rejected
#
# ترجع حاجتين:
#
# collected
#
# الـArticles اللي اتجمعت:
#
# {
#     277: (entries, 127),
#     278: (entries, 128),
#     279: (entries, 128)
# }
# rejected
#
# الـheaders candidates اللي رفضناها:
#
# [
#     (127, "Article 901 has been delivered to him."),
#     ...
# ]
#
# وده مفيد جدًا للـdiagnostics والتحقق من جودة extraction.
#
# الخلاصة الكبيرة
#
# فكر في الـfunction كأنها موظف ماسك ملف وبيحدد المادة الحالية:
#
#                 ┌──────────────────────┐
#                 │ current_article      │
#                 │      = 277           │
#                 └──────────┬───────────┘
#                            │
#           ┌────────────────┼────────────────┐
#           ↓                ↓                ↓
#       body line        blank line       cross-reference
#           │                │                │
#           └────────────────┴────────────────┘
#                            ↓
#                     تضاف للـ277
#
# ولما يظهر:
#
# Article 278
#
# يعمل:
#
#         Article 277
#              ↓
#           FINALIZE
#              ↓
#         collected[277]
#              ↓
#         Article 278
#              ↓
#            OPEN
#              ↓
#       current_number = 278
#
# ولو Article امتدت من Page 127 إلى Page 128:
#
# Page 127
# Article 277
#    ↓
# body
#    ↓
# body
#    ↓
# END PAGE
#    │
#    │  ← لا تقفل
#    ↓
# Page 128
# body
#    ↓
# body
#    ↓
# Article 278
#    ↓
# FINALIZE 277
# OPEN 278
#
# إذن الـstate tracker هنا هو اللي يحوّل الـPDF من مجرد مجموعة سطور منفصلة إلى Articles كاملة، مع الحفاظ على الـArticle عبر الصفحات، ومنع الـcross-references من تقطيع الـArticles بشكل خاطئ.
def _collect_articles(pages_lines, is_english):
    """Track one language's articles across every page.

    pages_lines is a list of (page_number, lines) pairs in order.
    is_english picks the English header detector instead of the Arabic
    one; both share exactly the same simple state logic.
    """
    collected = {}  # article number -> (entries, start_page)
    rejected = []  # Diagnostics: backwards/repeated header candidates.

    # --- the state of the currently open article ---
    current_number = None  # No article open at the start.
    current_entries = []  # Its (page_number, line) pairs.
    current_start_page = 0  # Page where its header was found.

    # Walk every page of this language, then every line in order.
    for page_number, lines in pages_lines:
        # Walk this page's lines with their position, so the English
        # lookahead below can look at the lines that follow on the same page.
        for index, line in enumerate(lines):
            # A blank line is a paragraph marker inside the open article.
            if line == "":
                if current_number is not None:
                    current_entries.append((page_number, line))
                continue  # Nothing else to do with an empty line.

            # Detect a header with the detector of this language.
            rest_of_line = ""  # Body text that shared the header line.
            if is_english:
                header_number = detect_english_article_header(line)
                if header_number is None:
                    # The English column sometimes puts the header and the
                    # first body line on one PDF line: split them apart.
                    header_number, rest_of_line = split_english_inline_header(line)
            else:
                header_number = detect_article_header(line)

            if header_number is not None:
                # Monotonic guard: only FORWARD numbers open articles.
                if current_number is None or header_number > current_number:
                    # ------------------------------------------------------
                    # Same-page lookahead (English only)
                    # ------------------------------------------------------
                    # The English column sometimes opens a BODY sentence with
                    # a cross-reference, for example page 127 contains the
                    # line "Article 901 has been delivered to him." BETWEEN
                    # the real headers Article 884 and Article 885. Taking it
                    # as a header jumps the tracker 884 -> 901 and wrongly
                    # rejects the real articles 885..900.
                    #
                    # The fix is simple and local: if a SMALLER forward
                    # header candidate appears later on the SAME page, this
                    # larger candidate must be a cross-reference, not a
                    # header, so the line stays body text. Only English
                    # needs this: its headers are not the authoritative
                    # sequence (the Arabic column is).
                    # ------------------------------------------------------
                    if is_english and current_number is not None:
                        smaller_follows = False
                        for later_line in lines[index + 1 :]:
                            later_number = detect_english_article_header(later_line)
                            if later_number is None:
                                later_number, _ = split_english_inline_header(
                                    later_line
                                )
                            if (
                                later_number is not None
                                and current_number < later_number < header_number
                            ):
                                smaller_follows = True
                                break
                        if smaller_follows:
                            # Treat it as body text (cross-reference).
                            header_number = None
                        else:
                            # Finalize the previous article (if one is open).
                            if current_number is not None:
                                collected[current_number] = (
                                    current_entries,
                                    current_start_page,
                                )
                            # Open the new article with a clean state.
                            current_number = header_number
                            current_entries = []
                            current_start_page = page_number
                            # Keep any body text on the header's own line.
                            if rest_of_line.strip() != "":
                                current_entries.append((page_number, rest_of_line))
                            continue  # The header text itself is not body text.
                    else:
                        # Finalize the previous article (if one is open).
                        if current_number is not None:
                            collected[current_number] = (
                                current_entries,
                                current_start_page,
                            )
                        # Open the new article with a clean state.
                        current_number = header_number
                        current_entries = []
                        current_start_page = page_number
                        # Keep any body text that was on the header's line.
                        if rest_of_line.strip() != "":
                            current_entries.append((page_number, rest_of_line))
                        continue  # The header text itself is not body text.

                elif current_number is None or header_number <= current_number:
                    # Backwards/repeated number: a cross-reference in the
                    # body text, not a real header. Keep it as body text and
                    # record it as a diagnostic.
                    rejected.append((page_number, line))

            # Ordinary text belongs to the currently open article.
            if current_number is not None:
                current_entries.append((page_number, line))

    # Finalize the LAST open article at the end of the document.
    if current_number is not None:
        collected[current_number] = (current_entries, current_start_page)

    return collected, rejected


# ============================================================
# Reconstruct Complete Bilingual Articles Across All Pages
# ----------------------------------------------------------
# Flow:
#   1. Extract BOTH columns of every page with dynamic coordinates
#      (Arabic = right half, English = left half), top-to-bottom.
#   2. Reconstruct each language independently with the simple state
#      tracker, so an article that continues onto the next page is
#      reconstructed correctly in BOTH languages.
#   3. Merge the two sides by the SAME article_number, which is the
#      key printed by both headers ("مادة N" / "Article N"), into one
#      bilingual record per article.
# ============================================================
def extract_articles(document):
    """Extract all bilingual article records from the PDF.

    Returns (articles, preamble_text, rejected_headers):
      - articles:         bilingual records, ascending by article number
      - preamble_text:    Arabic text before the first article header
      - rejected_headers: (page_number, line) candidates that failed
                          the monotonic guard, from both languages
    """
    preamble_lines = []  # Arabic text before the first مادة header.

    # Keep each language's page lines so both trackers can see the
    # whole document in order (required for multi-page articles).
    arabic_pages = []  # [(page_number, [lines...]), ...]
    english_pages = []  # [(page_number, [lines...]), ...]

    # Walk every page of the document (1-based page numbers for output).
    for page_index in range(document.page_count):
        page_number = page_index + 1  # Humans count pages from 1.
        page = document.load_page(page_index)  # Load this page.

        # Arabic right column (reading order already restored).
        arabic_pages.append((page_number, extract_arabic_lines(page)))

        # English left column (logical order, safe cleaning only).
        english_pages.append((page_number, extract_english_lines(page)))

    # --- Reconstruct the Arabic side (the source-of-truth text) ---
    arabic_articles, arabic_rejected = _collect_articles(arabic_pages, False)

    # --- Reconstruct the English side with the same simple logic ---
    english_articles, english_rejected = _collect_articles(english_pages, True)

    # Combine the diagnostics of both languages for the report.
    rejected_headers = arabic_rejected + english_rejected

    # A quick lookup set of the REJECTED candidate lines, used below to
    # prove that a broken header is really a misprint (a correctly
    # printed header is never rejected by the monotonic guard).
    rejected_candidate_keys = set(arabic_rejected)

    # Collect the Arabic preamble (title / issuing decree) by taking
    # every Arabic line before the first real article header.
    for page_number, lines in arabic_pages:
        for line in lines:
            if detect_article_header(line) is not None:
                break  # The first article starts here; stop the preamble.
            preamble_lines.append(line)
    preamble_text = cleaning.assemble_article_text(preamble_lines)

    articles = []  # The final bilingual records, ascending by number.

    # The union of both languages' numbers: an article present on only
    # one side (this PDF has such cases) is still kept, so no legal
    # article is silently lost from the dataset.
    all_numbers = sorted(set(arabic_articles) | set(english_articles))

    for article_number in all_numbers:
        # Take each side when it exists; otherwise an empty list.
        if article_number in arabic_articles:
            ar_entries, page_start = arabic_articles[article_number]
        else:
            ar_entries = []
            page_start = 0

        if article_number in english_articles:
            en_entries, en_start = english_articles[article_number]
            # An English-only article starts on the English header page.
            if page_start == 0:
                page_start = en_start
        else:
            en_entries = []

        # Build the bilingual record for this article number.
        articles.append(
            _build_article(article_number, ar_entries, en_entries, page_start)
        )

    # ============================================================
    # Articles Whose Body Is Absent Because They Were Repealed
    # ----------------------------------------------------------
    # This edition of the PDF prints a REPEAL NOTE instead of the
    # article bodies for the repealed ranges. Example (page 7):
    #     "* Articles 54-80 have been repealed by Presidential Decree."
    #     "المواد من 54 إلى 80 ملغاة بقرار من رئيس الجمهورية"
    # Because each article number in such a range is a real article of
    # the Code, a record is created for EVERY number of the range so
    # the dataset keeps a complete, gap-free numbering (1..1149).
    # The text of such a record is the repeal note itself: that is the
    # only legal text the source contains for these articles, and it
    # is stored verbatim, never invented.
    # ============================================================
    for range_start, range_end in cleaning.REPEALED_RANGES:
        # Collect the repeal note of this range, per language, verbatim.
        arabic_note = ""
        english_note = ""
        note_page = 0

        for page_number, lines in arabic_pages:
            note = _find_repeal_note(lines, range_start, range_end)
            if note:
                arabic_note = note  # The Arabic repeal note of the range.
                if note_page == 0:
                    note_page = page_number  # Page that announces the range.
                break

        for page_number, lines in english_pages:
            note = _find_repeal_note(lines, range_start, range_end)
            if note:
                english_note = note  # The English repeal note of the range.
                if note_page == 0:
                    note_page = page_number
                break

        for article_number in range(range_start, range_end + 1):
            # Skip numbers that already have a real record.
            if article_number in arabic_articles or article_number in english_articles:
                # Complete an existing record if one side of it is empty.
                for record in articles:
                    if record["article_number"] != article_number:
                        continue
                    if record["text"]["ar"].strip() == "":
                        record["text"]["ar"] = arabic_note
                    if record["text"]["en"].strip() == "":
                        record["text"]["en"] = english_note
                continue

            # Build the record with the same bilingual structure.
            articles.append(
                {
                    "article_number": article_number,
                    "text": {
                        "ar": arabic_note,
                        "en": english_note,
                    },
                    "page_start": note_page,
                    "page_end": note_page,
                    "is_repealed": True,
                }
            )

    # ============================================================
    # Repair Known Misprinted Arabic Headers
    # ----------------------------------------------------------
    # A few Arabic headers are stored in the PDF with transposed
    # digits, so the mirroring step produces a number that is NOT
    # the real one and the monotonic guard (correctly) refuses to
    # open a new article. Without this step the real article text
    # would stay attached to the previous article.
    #
    # The repair is applied ONLY when BOTH facts hold on the SAME page:
    #   - the broken candidate is the one the monotonic guard REJECTED
    #     (so it is proved to sit in a backwards position, which is
    #     exactly the situation a transposed header creates)
    #   - the next real Arabic header on that page is exactly
    #     real_number + 1 (so 601 must sit right before 602)
    # Nothing else on any other page can be affected: a correctly
    # printed header never reaches this branch, because it is never
    # recorded as a rejected candidate.
    # ============================================================
    for broken_number, real_number in TRANSPOSED_HEADER_REPAIRS.items():
        # The expected next header proves the broken value really is
        # a misprint of this article (601 sits between 600 and 602).
        expected_next_number = real_number + 1
        next_header_found = False

        # The page on which BOTH proofs succeeded, so the repair works on
        # exactly that page instead of the first page that has this number
        # (article 160 really does exist on page 18 and must not be used).
        proven_page_number = None

        for page_number, lines in arabic_pages:
            # Walk this page to find the broken header line.
            for index, line in enumerate(lines):
                if detect_article_header(line) != broken_number:
                    continue

                # Proof 1: the monotonic guard must have REJECTED this
                # exact line, which only happens for a backwards number.
                if (page_number, line) not in rejected_candidate_keys:
                    continue

                # Proof 2: the next header on the SAME page must be
                # the expected number (602 right after the misprint).
                for next_line in lines[index + 1 :]:
                    next_number = detect_article_header(next_line)
                    if next_number is None:
                        continue
                    if next_number == expected_next_number:
                        next_header_found = True
                        proven_page_number = page_number
                    break

                if next_header_found:
                    break
            if next_header_found:
                break

        # Apply the repair only on the proven page, and only when the
        # surrounding headers confirm it.
        if next_header_found:
            _repair_transposed_header(
                articles, arabic_pages, broken_number, real_number, proven_page_number
            )

    # Keep the dataset ordered by article number, as the source is.
    articles.sort(key=lambda record: record["article_number"])

    return articles, preamble_text, rejected_headers


# ============================================================
# Known Source Defect: Transposed Digits in Article 601's Header
# ----------------------------------------------------------
# The PDF stores this ONE header with its Arabic-Indic digits
# transposed: the glyph run is "0 6 1" instead of the mirrored
# "1 0 6" that every neighbouring header uses (598 -> 8 9 5,
# 599 -> 9 9 5, 600 -> 0 0 6, 602 -> 2 0 6 are all correct).
# The normal mirroring therefore produces the wrong number 160,
# the monotonic guard rejects it, and the real Article 601 body
# would silently stay attached to Article 600.
#
# The repair is a single explicit entry: it is deterministic, it is
# only applied when the rejected candidate is exactly this value
# AND the next real Arabic header is exactly 602, so it can never
# change the behaviour of any other page or article.
# ============================================================
TRANSPOSED_HEADER_REPAIRS = {160: 601}


# ============================================================
# Repair One Article Whose Header Digits Are Transposed In The PDF
# ----------------------------------------------------------
# The Arabic text of the affected article is the block of Arabic
# lines that follows the broken header line and stops before the
# NEXT valid Arabic header on the same page. Those lines were
# previously appended to the previous article, so they are taken
# back out of it and attached to the correct article number.
# The English text of the article is left completely untouched.
# ============================================================
def _repair_transposed_header(
    articles, arabic_pages, broken_number, real_number, proven_page_number
):
    """Move the Arabic text of a misprinted header to its real article."""
    # The record of the real article must already exist (from English).
    target = None
    previous = None
    for record in articles:
        # Target = the record whose number is exactly the real one.
        if record["article_number"] == real_number:
            target = record
        # Previous = the record with the GREATEST number below the real
        # one. The list is not sorted at this point (repeal-range records
        # are appended after it), so the neighbours must be found by
        # comparing the numbers, not by taking the last match in the
        # list. Otherwise the wrong article (e.g. 417) would be cut.
        if record["article_number"] < real_number:
            if (
                previous is None
                or record["article_number"] > previous["article_number"]
            ):
                previous = record
    # Nothing to do when the record is missing or already has Arabic.
    if target is None or target["text"]["ar"].strip() != "":
        return False

    # Remember the previous article's text so we can give back the lines
    # that do not belong to it.
    previous_text = ""
    if previous is not None:
        previous_text = previous["text"]["ar"]

    # Find the broken header line on the page that was proven above, and
    # the next real header on that same page.
    for page_number, lines in arabic_pages:
        # Only the proven page may be used (article 160 also exists on p18).
        if page_number != proven_page_number:
            continue
        for index, line in enumerate(lines):
            if detect_article_header(line) != broken_number:
                continue
            # Collect the lines that follow the broken header, stopping
            # at the next valid header (that header starts another article).
            block = []
            start_page = page_number
            end_page = page_number
            for next_line in lines[index + 1 :]:
                if detect_article_header(next_line) is not None:
                    break  # The next real article begins here.
                if next_line.strip() == "":
                    block.append("")  # Keep the paragraph markers.
                    continue
                block.append(next_line)  # Arabic text of the real article.

            if len(block) == 0:
                return False  # No text found: do not change anything.

            # The recovered text belongs to the real article number.
            target["text"]["ar"] = cleaning.assemble_article_text(block)
            target["page_start"] = start_page
            target["page_end"] = end_page

            # Give the lines back to the previous article: the broken
            # header line was appended to it earlier, so remove that
            # exact prefix line and everything after it from its text.
            if previous is not None and previous_text:
                broken_line_text = cleaning.normalize_line(line)
                position = previous_text.find(broken_line_text)
                if position != -1:
                    previous["text"]["ar"] = previous_text[:position].rstrip()
            return True
    return False


# ============================================================
# Find the Repeal Note of One Article Range on a Page
# ----------------------------------------------------------
# The repeal note is a LINE (or two adjacent lines) that mentions both
# ends of a repealed range, e.g.:
#     "* Articles 54-80 have been repealed by Presidential Decree."
#     "المواد من ٥٤ إلى ٨٠ ملغاة بقرار من رئيس الجمهورية"
# Only the line(s) that actually mention the range are returned, and
# they are returned VERBATIM (safe whitespace cleaning only), so the
# stored text is real source text and never invented.
# Returns the note text, or "" when this page has no such note.
# ============================================================
def _find_repeal_note(lines, range_start, range_end):
    """Return the repeal-note text of this range found on the page."""
    # Both ends of the range, with Arabic-Indic digits as the note prints them.
    start_digits = cleaning.normalize_arabic_digits(str(range_start))
    end_digits = cleaning.normalize_arabic_digits(str(range_end))

    note_lines = []  # The lines that make up the note.

    for line in lines:
        # Clean the line safely before looking for the numbers.
        cleaned = cleaning.clean_english_line(line)
        # Both numbers of the range must be present on this line.
        if start_digits in cleaned and end_digits in cleaned:
            note_lines.append(cleaned)  # Keep the real text of the note.

    # No note on this page.
    if not note_lines:
        return ""

    # Join the note lines with a space, in reading order.
    return " ".join(note_lines)
