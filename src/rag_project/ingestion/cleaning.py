"""Conservative text cleaning and legal-status marking for ingestion.

This is LEGAL text, so cleaning is deliberately SAFE:
  - it fixes representation only (digits, spaces, invisible PDF
    artifacts, malformed clause parentheses)
  - it NEVER rewrites, paraphrases, summarizes, translates, or
    deletes legal wording
  - paragraph boundaries are preserved, not flattened away

The module also owns the repealed-article ranges, because marking
an article as repealed is a normalization/metadata concern: the
article text itself is always preserved.
"""

"""
دي مكتبة Python للـRegular Expressions.

بتعمل إيه؟

بتساعدك تبحث عن patterns معينة في النص.

مثلاً:

(1)
(2)
(أ)

أو:

)(2(

وتقدر تحدد pattern معين وتعدله.

هدفها هنا؟

استخدامها في:

اكتشاف الـclause markers
إصلاح الأقواس
اكتشاف بداية paragraph
تنظيف الـPDF artifacts
"""
import re


# ============================================================
# Repealed Article Ranges
# ----------------------------------------------------------
# Some articles of the code were repealed by later laws. We keep
# them (never delete legal source text) and only mark them so the
# later retrieval stage can filter or warn about them.
# ============================================================
REPEALED_RANGES = [
    (54, 80),
    (389, 417),
]

"""
المشكلة اللي بتحلها؟

بدل ما تعمل check يدوي لكل مادة، الـfunction بتعمل الـcheck أوتوماتيك.

باختصار:

is_repealed_article() = هل المادة دي ملغاة؟
"""
def is_repealed_article(article_number):
    """Return True when the article number falls in a repealed range."""
    # Check every known range with a simple loop.
    for start, end in REPEALED_RANGES:
        if start <= article_number <= end:
            return True  # Inside 54-80 or 389-417.

    return False  # Not repealed.


# ============================================================
# Arabic-Indic Digit Normalization
# ----------------------------------------------------------
# The PDF writes article numbers with Arabic-Indic digits (٠١..٩).
# Everything downstream (header detection, article numbers, JSON)
# works with Western digits (0-9), so we translate the digit
# symbols. This changes only the digit SYMBOL, never the value
# or the surrounding legal wording.
# ============================================================
# Arabic-Indic digits 0-9 followed by Extended Arabic-Indic 0-9.
# نوعين من الارقام العربيه
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
# The matching Western digits (same order, two digit systems).
# الارقام العاديه للي بيقدر يتعامل معاها 
WESTERN_DIGITS = "01234567890123456789"
# A translation table usable with str.translate().
# ديه مجرد transalation table  بتهيئ تحويل الارقام من ARABIC_DIGITS الي WESTERN_DIGITS
"""
يعني تقول لـPython:

٠ → 0
١ → 1
٢ → 2
...
الهدف؟

توحيد شكل الأرقام.
"""
DIGIT_TABLE = str.maketrans(ARABIC_DIGITS, WESTERN_DIGITS)

# توحد شكل الارقام وبتحوله من ARABIC_DIGITS الي WESTERN_DIGITS 
# دة ال transaltion الفعلي المطبق علي translation table
def normalize_arabic_digits(text):
    """Convert Arabic-Indic digits (٠١..٩) to Western digits (01..9)."""
    # translate() is a simple character-for-character replacement.
    return text.translate(DIGIT_TABLE)


# ============================================================
# Invisible PDF / Bidi Extraction Artifacts
# ----------------------------------------------------------
# PDF extraction can inject zero-width characters, bidi marks, and
# non-breaking spaces into the text. They are invisible technical
# noise: they carry no legal meaning and only break comparisons
# and regex matching, so they are replaced by a normal space.
# ============================================================
"""
دي بتحدد مجموعة من الـcharacters اللي ممكن تظهر أثناء استخراج الـPDF لكنها مش ظاهرة للمستخدم.

زي:

Zero-width characters
Bidirectional marks
Non-breaking spaces
المشكلة؟

ممكن يكون عندك:

المادة​ 15

وأنت شايفها كأنها:

المادة 15

لكن Python ممكن يعتبر بينهم character إضافي.

وده يعمل مشاكل في:

regex matching
comparison
search
article detection
"""

INVISIBLE_CHARACTERS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0]"
)


# ============================================================
# Clause Marker Parenthesis Normalization
# ----------------------------------------------------------
# Inspection of the real PDF showed clause markers coming out of
# extraction with broken parenthesis orientation, for example:
#       )( 2 (      ( 1 (      1)(
# instead of the readable logical form:
#       (2)         (1)        (1)
#
# We only rewrite a parenthesis pair that DIRECTLY surrounds a
# short number (a clause/article marker like 1, 12, 100) or a
# single Arabic letter (clause marker like (أ)). Parentheses
# elsewhere in the legal text are left untouched, because we
# cannot prove their orientation is wrong there.
# ============================================================
# Forms where a paren sits on BOTH sides of a number:
#   ")5(" , "(5(" , "(5)"  -> canonical "(5)"
CLAUSE_NUMBER_PARENS = re.compile(r"[()]\s*(\d{1,4})\s*[()]")
# Form where the digit comes FIRST: "5)(" -> canonical "(5)"
CLAUSE_NUMBER_TRAILING = re.compile(r"(\d{1,4})\s*\)\s*\(")
# Letter clause markers like ")أ(" -> canonical "(أ)"
CLAUSE_LETTER_PARENS = re.compile(r"[()]\s*([أ-ي])\s*[()]")
# Line-START markers stored by the PDF as "1( ( النصوص" or "1 ( النصوص"
# (digit first, one or two parens, then the clause text). Inspection of
# the generated articles found 303 such line starts, ALL of them clause
# markers, and ZERO lines starting with a 4-digit year followed by a
# paren - so this position-specific rule is safe for this document.
# Only 1-3 digit numbers qualify (a wrapped year like "1949 (" cannot
# match), and lines like "2 - نص" (dash style) are untouched because
# a parenthesis right after the digit is required.
"""
دي بتتعامل مع الـclause marker لما يكون في بداية السطر.

مثلاً:

1( ( النص القانوني

تتحول إلى:

(1) النص القانوني
"""
CLAUSE_LINE_START = re.compile(r"^(\d{1,3})\s*[()]\s*[()]?\s*")


""" 
المشكلة اللي بتحلها؟

الـPDF extraction ممكن يطلع الـclause marker بشكل غلط.

مثلاً:

)(2(

بدل:

(2)

أو:

2)(

بدل:

(2)

فالـfunction بتوحدهم إلى:

(2)
"""
def canonicalize_clause_markers(text):
    """Rewrite broken clause markers like ')(2(' into canonical '(2)'."""
    # Line-start form "1( ( نص" -> "(1) نص" (most common broken form).
    text = CLAUSE_LINE_START.sub(r"(\1) ", text, count=1)
    # Put a number (any parenthesis orientation) into "(number)".
    text = CLAUSE_NUMBER_PARENS.sub(r"(\1)", text)
    # Handle the digit-first form "2)(" found in real extraction output.
    text = CLAUSE_NUMBER_TRAILING.sub(r"(\1)", text)
    # Put single-letter clause markers into "(letter)" form.
    text = CLAUSE_LETTER_PARENS.sub(r"(\1)", text)
    # Everything else in the line is left exactly as extracted.
    return text


# ============================================================
# Line Cleaning (one composed, safe step)
# ----------------------------------------------------------
# This is the single cleaning entry point used by the extraction
# loop. It is idempotent: running it twice gives the same result.
# It performs ONLY representation fixes, never wording changes.
# ============================================================
# ال main function لل cleaning بتشتغل علي كل سطر معموله extraction تطبق عليه الفانكشن ديه 
def normalize_line(line):
    """Clean one extracted line: invisible chars, digits, clause parens, spaces."""
    # Remove invisible bidi/zero-width artifacts from extraction.
    line = INVISIBLE_CHARACTERS.sub(" ", line)
    # Make digits consistent (Arabic-Indic -> Western).
    line = normalize_arabic_digits(line)
    # Repair broken clause/article marker parentheses.
    line = canonicalize_clause_markers(line)
    # Collapse runs of spaces/tabs into single spaces and trim ends.
    # (Blank-line paragraph markers are handled by the caller before
    # this function ever sees them, so paragraphs survive.)
    # توحيد المسافات 
    line = " ".join(line.split())
    return line


# ============================================================
# English Line Cleaning
# ----------------------------------------------------------
# The English column is real (translated) legal text, so it gets the
# same conservative treatment as Arabic:
#   - invisible PDF/bidi artifacts are removed
#   - runs of spaces are collapsed to one space
#   - every other character (wording, punctuation, digits) is kept
# Nothing is paraphrased, translated, summarized, or deleted.
# NOTE: no reading-order restoration is needed here, because the
# English column is left-to-right and extracts in logical order.
"""
ليه function منفصلة؟

لأن الـArabic عنده مشاكل إضافية، خصوصًا:

Arabic digits
Bidi / RTL
Arabic clause markers

أما الـEnglish فمحتاج cleaning أبسط.
"""
# ============================================================
def clean_english_line(line):
    """Clean one extracted English line (safe representation fixes only)."""
    # Remove invisible zero-width / bidi / non-breaking-space artifacts.
    line = INVISIBLE_CHARACTERS.sub(" ", line)
    # Collapse repeated spaces/tabs and trim the ends.
    line = " ".join(line.split())
    return line


# ============================================================
# Detect the Start of a New Clause Paragraph
# ----------------------------------------------------------
# Legal articles are written as numbered/lettered clauses, and the
# PDF often wraps them across many physical lines. Treating each
# clause marker as the start of a paragraph preserves the real
# document structure for the later chunking stage, without
# changing any wording.
# ============================================================
"""
بتعمل إيه؟

تستخدم الـregex اللي تحت وتشوف:

هل السطر يبدأ بـ clause marker؟

مثلاً:

(2) يجوز...

ترجع:

True

لكن:

ويجوز للمحكمة...

ترجع:

False

المشكلة اللي بتحلها؟

مهمة جدًا في الحفاظ على legal paragraph structure.

لأن الـPDF ممكن يقسم clause واحدة على كذا physical line.
"""

CLAUSE_AT_START = re.compile(r"^\s*(?:\(\d{1,4}\)|\d{1,4}\)|\([أ-ي]\))")


def starts_new_clause(line):
    """True when the line begins with a clause marker like (2) or (أ)."""
    # A simple anchored match at the start of the line.
    return CLAUSE_AT_START.match(line) is not None


# ============================================================
# Assemble Article Text From Cleaned Lines
# ----------------------------------------------------------
# وظيفتها إنها تاخد lines الخاصة بمادة واحدة وتجمعهم في نص منظم.
# Input: the cleaned lines of one article, in extraction order.
# Output: one text string where
#   - a blank line from the PDF = paragraph break
#   - a new clause marker      = paragraph break
#   - all other lines are joined with single spaces
# This removes only HARD WRAPPING (a layout artifact) and keeps
# the meaningful paragraph structure of the legal text.
# ============================================================
# تحول الـPDF lines دي إلى paragraphs منطقية بدل ما تفضل كل جملة متقسمة حسب شكل الـPDF.
def assemble_article_text(lines):
    """Join an article's cleaned lines into paragraphs of legal text."""
    # دي List هنحط فيها الـparagraphs اللي خلصنا بنائها.
    paragraphs = []  # Finished paragraphs of the article.
    # الـparagraph اللي أنا لسه ببنيه دلوقتي. وشغال عليها دلوقتي
    current = []  # Lines of the paragraph currently being built.

    for line in lines:
        # لو السطر فاضي __ وده غالبًا معناه إن فيه paragraph خلصت. (BLANK LINE)
        if line == "":
            # A blank PDF line ends the current paragraph.
            if current:
                """
                هنا بيحصل حاجتين.

أولًا:

" ".join(current)

لو:

current = [
    "يلتزم المدين بالوفاء بالدين.",
    "وذلك في الميعاد المحدد."
]

تصبح:

"يلتزم المدين بالوفاء بالدين. وذلك في الميعاد المحدد."

يعني بدل ما كل line تبقى منفصلة، بنجمعهم بمسافة.

بعد كده:

paragraphs.append(...)

نضيف الـparagraph المكتملة إلى paragraphs.

فتصبح:

paragraphs = [
    "يلتزم المدين بالوفاء بالدين. وذلك في الميعاد المحدد."
]
"""
                paragraphs.append(" ".join(current))
                # تصفير ال current  يعني الـparagraph القديمة خلصت، ابدأ paragraph جديدة من الصفر.
                current = []
                """
                دي أهم حتة.

هنا بنسأل سؤالين:

السؤال الأول:
current

يعني:

هل أنا بالفعل جوه paragraph؟

السؤال الثاني:
starts_new_clause(line)

يعني:

هل الـline الجديدة بتبدأ بـ clause marker؟

زي:

(2)

أو:

(3)

أو:

(أ)
"""
        elif current and starts_new_clause(line):
            # A clause marker like (2) starts a new paragraph,
            # but only when we are already inside one (so the
            # very first line of the article never splits).
            # لو الحاله ديه اتحققت 
            """
            بنقول:

الـparagraph اللي كنت ببنيها خلصت.

فنجمع سطورها ونحطها في paragraphs.
"""
            paragraphs.append(" ".join(current))
            """
            بنبدأ الـparagraph الجديدة بالـline الحالية.

مثلاً:

line = "(2) إذا تأخر المدين في الوفاء..."

فتصبح:

current = [
    "(2) إذا تأخر المدين في الوفاء..."
]

وبعدين أي lines بعدها هتتضاف لنفس الـparagraph.
لو لقيت clause mark زي (..)
"""
            current = [line]
        else:
            # Ordinary continuation line of the current paragraph.
            """
            يعني لا:

blank line
ولا بداية clause جديدة

إذن دي مجرد continuation للـparagraph الحالية.

14. إضافة الـline
current.append(line)

مثلاً:

current = [
    "يلتزم المدين بالوفاء بالدين."
]

والـline الجديدة:

"وذلك في الميعاد المحدد."

فتصبح:

current = [
    "يلتزم المدين بالوفاء بالدين.",
    "وذلك في الميعاد المحدد."
]

وبعدين في النهاية:

" ".join(current)

تبقى:

يلتزم المدين بالوفاء بالدين. وذلك في الميعاد المحدد.
"""
            current.append(line)

    # Do not forget the final open paragraph.
    if current:
        paragraphs.append(" ".join(current))

    # Paragraphs are separated by a blank line in the output text.
    # هنا بنرجع كل الـparagraphs بعد فصلهم بـ: عن بعض
    return "\n\n".join(paragraphs)

"""
16. ليه assemble_article_text() مهمة للـRAG؟

لأن الـPDF عنده physical lines، لكن أنت محتاج logical paragraphs.

يعني:

PDF layout
   ↓
line 1
line 2
line 3
line 4

مش بالضرورة معناها:

paragraph 1
paragraph 2
paragraph 3
paragraph 4

فهي بتحاول ترجع structure منطقي للنص.

وده مهم جدًا بعدين في:

Chunking
   ↓
Embedding
   ↓
Retrieval

لأنك مش عايز chunk يتقطع بطريقة عشوائية وسط clause قانونية.
"""