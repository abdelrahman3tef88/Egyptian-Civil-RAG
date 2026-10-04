"""build_golden_dataset.py - build data/golden/golden_dataset.json.

The golden dataset is the evaluation input of the MLflow experiment
phase. It contains EXACTLY 25 questions and six question types:

    1. article_lookup               ->  5 questions
    2. article_identification       ->  4 questions
    3. legal_concept                ->  4 questions
    4. specific_provision           ->  3 questions
    5. multi_article                ->  3 questions
    6. paraphrased_question         ->  6 questions

The set is split by language and by group:

    Arabic  : 13 questions    English : 12 questions
    general/direct        : 20 questions
    challenging/diverse   :  5 questions

Every question is derived from the project's real article data
(data/processed/articles.json) - the same file the RAG index is built
from. NOTHING is invented:

  * ground_truth is the article's own stored text (Arabic for Arabic
    questions, English for the English ones),
  * ground_truth_contexts is that same source text, and
  * every curated question asserts the exact wording it relies on
    (`expect` in the tables below). If a claim about an article were
    wrong, generation FAILS loudly instead of emitting a bad question.

The ground truth is evaluator-side information only: run_experiments.py
sends the question - and nothing else - to the RAG pipeline.

Run with:  uv run python scripts/build_golden_dataset.py
"""

import json
import re
import sys
from collections import Counter

from rag_project.config import settings

# Ingestion output (the source of truth for every question).
ARTICLES_PATH = settings.resolve_path(settings.CONFIG["data"]["articles_json"])

# Where the golden dataset is written (project-relative, like every
# other path in this project).
OUTPUT_PATH = settings.PROJECT_ROOT / "data" / "golden" / "golden_dataset.json"

# Version label stored in the dataset file (logged to MLflow).
DATASET_VERSION = "golden-v1"

# Tolerant topic keywords -> display label.
# The keywords avoid the PDF's lam-alef reordering ("اإل") and its
# split "لا" ("ال"), so they match the stored text reliably.
TOPIC_KEYWORDS = [
    ("يجار", "الإيجار"),
    ("البيع", "البيع"),
    ("مقايضة", "المقايضة"),
    ("هبة", "الهبة"),
    ("وصية", "الوصية"),
    ("ميراث", "الميراث"),
    ("تقادم", "التقادم"),
    ("شفعة", "الشفعة"),
    ("كفالة", "الكفالة"),
    ("رهن", "الرهن"),
    ("امتياز", "الامتياز"),
    ("شركة", "الشركة"),
    ("وكالة", "الوكالة"),
    ("يداع", "الوديعة"),
    ("قرض", "القرض"),
    ("صلح", "الصلح"),
    ("ضرر", "المسؤولية عن الضرر"),
    ("تعويض", "التعويض"),
    ("ملكية", "الملكية"),
    ("حيازة", "الحيازة"),
    ("وقف", "الوقف"),
    ("حكر", "الاحتكار"),
    ("قرابة", "القرابة"),
    ("موطن", "الموطن"),
]


# ==========================================================
# Display-only text normalisation
# ----------------------------------------------------------
# Extracted PDF text carries representation artifacts (lam-alef
# reordering, a split "لا", stray spaces). The QUESTION text is cleaned
# for readability; the GROUND TRUTH is always the untouched source
# text, because that is what the index actually contains.
# ==========================================================
LAM_ALEF_FIXES = (
    ("اإل", "الإ"),
    ("األ", "الأ"),
    ("اآل", "الآ"),
)


def clean_for_display(text):
    """Make a source excerpt readable without changing its meaning."""
    cleaned = " ".join(text.split())
    for broken, fixed in LAM_ALEF_FIXES:
        cleaned = cleaned.replace(broken, fixed)
    # A standalone "ال" token is the PDF's split rendering of "لا".
    cleaned = re.sub(r"\bال\b", "لا", cleaned)
    # The PDF writes the end-of-line full stop BEFORE the last word
    # ("فى مصلحة .المدين"); move it back behind that word.
    cleaned = re.sub(r"\s\.(\S+)", r" \1.", cleaned)
    # Any leftover dangling space before a full stop.
    cleaned = cleaned.replace(" .", ".")
    return cleaned.strip()


def excerpt(text, limit=90):
    """Return the opening of a text, cut at a word boundary."""
    cleaned = clean_for_display(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "…"


def derive_topic(article_text):
    """Label the article's topic from its own wording (no invention)."""
    for keyword, label in TOPIC_KEYWORDS:
        if keyword in article_text:
            return label
    return "أحكام عامة"


def load_articles():
    """Read the ingestion output and index it by article number."""
    with open(ARTICLES_PATH, "r", encoding="utf-8") as articles_file:
        data = json.load(articles_file)
    return data["articles"]


# ==========================================================
# Curated questions
# ----------------------------------------------------------
# Every entry declares:
#   articles -> the real article numbers the question is built on
#   expect   -> wording that MUST be present in those articles
#               (proves the question matches the stored source text)
# A mismatch aborts generation, so a question can never assert a legal
# fact that the data does not actually contain.
# ==========================================================

# --- Type 3: legal_concept (20) ---
CONCEPT_QUESTIONS = [
    {
        # English concept: the context/answer come from the article's
        # English column, so the bilingual corpus stays exercised.
        "articles": [430],
        "expect": {430: "مؤجل الثمن"},
        "question": "In a sale where the price is deferred, may the vendor keep the ownership with the purchaser until the price is paid in full?",
        "language": "en",
        "difficulty": "medium",
        "topic": "Sale",
    },
    {
        "articles": [89],
        "expect": {89: "يتم العقد بمجرد"},
        "question": "ما هو الوقت الذي ينعقد فيه العقد وفقاً للقانون المدني المصري، وما الذي يشترط لانعقاده؟",
        "difficulty": "easy",
        "topic": "انعقاد العقد",
    },
    {
        "articles": [90],
        "expect": {90: "باللفظ"},
        "question": "ما هي الصور التي يجوز بها التعبير عن الإرادة في القانون المدني المصري؟",
        "difficulty": "easy",
        "topic": "التعبير عن الإرادة",
    },
    {
        "articles": [93],
        "expect": {93: "ميعاد للقبول"},
        "question": "ما مدى التزام الموجب بالبقاء على إيجابه إذا حُدد ميعاد للقبول؟",
        "difficulty": "medium",
        "topic": "الإيجاب والقبول",
    },
    {
        "articles": [94],
        "expect": {94: "مجلس العقد"},
        "question": "ما حكم الإيجاب الصادر في مجلس العقد دون تحديد ميعاد للقبول، وما حكم الإيجاب الصادر بالهاتف؟",
        "difficulty": "medium",
        "topic": "الإيجاب والقبول",
    },
    {
        "articles": [96],
        "expect": {96: "يزيد في"},
        "question": "ما حكم القبول الذي يزيد في الإيجاب أو يقيد منه أو يعدل فيه؟",
        "difficulty": "medium",
        "topic": "الإيجاب والقبول",
    },
    {
        "articles": [98],
        "expect": {98: "السكوت"},
        "question": "متى يعتبر السكوت عن الرد قبولاً في القانون المدني المصري؟",
        "difficulty": "hard",
        "topic": "الإيجاب والقبول",
    },
    {
        "articles": [147],
        "expect": {147: "شريعة المتعاقدين"},
        "question": "ما معنى أن العقد شريعة المتعاقدين، وما أثر الحوادث الاستثنائية العامة على تنفيذ العقد؟",
        "difficulty": "medium",
        "topic": "آثار العقد",
    },
    {
        "articles": [148],
        "expect": {148: "حسن"},
        "question": "كيف يجب تنفيذ العقد وفقاً لأحكام القانون المدني المصري، وما الذي يتضمنه العقد بخلاف ما ورد فيه صراحة؟",
        "difficulty": "medium",
        "topic": "آثار العقد",
    },
    {
        "articles": [149],
        "expect": {149: "تعسف"},
        "question": "ما سلطة القاضي في الشروط التعسفية الواردة في عقد الإذعان، وما حكم الاتفاق المخالف لذلك؟",
        "difficulty": "hard",
        "topic": "عقود الإذعان",
    },
    {
        "articles": [100],
        "expect": {100: "ذعان"},
        "question": "علام يقتصر القبول في عقود الإذعان؟",
        "difficulty": "medium",
        "topic": "عقود الإذعان",
    },
    {
        "articles": [163],
        "expect": {163: "خطأ"},
        "question": "ما هو أساس المسؤولية المدنية عن الفعل الشخصي في القانون المدني المصري؟",
        "difficulty": "easy",
        "topic": "المسؤولية التقصيرية",
    },
    {
        "articles": [164],
        "expect": {164: "مميز"},
        "question": "ما حكم مسؤولية الشخص عن أعماله غير المشروعة، وما حكم الضرر الواقع من شخص غير مميز؟",
        "difficulty": "medium",
        "topic": "المسؤولية التقصيرية",
    },
    {
        "articles": [165],
        "expect": {165: "قوة قاهرة"},
        "question": "ما حكم الشخص إذا أثبت أن الضرر نشأ عن سبب أجنبي لا يد له فيه؟",
        "difficulty": "hard",
        "topic": "أسباب الإعفاء من المسؤولية",
    },
    {
        "articles": [166],
        "expect": {166: "دفاع شرعى"},
        "question": "ما حكم من أحدث ضرراً وهو في حالة دفاع شرعي عن نفسه أو ماله أو عن نفس الغير أو ماله؟",
        "difficulty": "hard",
        "topic": "أسباب الإعفاء من المسؤولية",
    },
    {
        "articles": [169],
        "expect": {169: "متضامنين"},
        "question": "إذا تعدد المسؤولون عن عمل ضار واحد، فكيف تكون مسؤوليتهم وكيف توزع فيما بينهم؟",
        "difficulty": "medium",
        "topic": "المسؤولية التقصيرية",
    },
    {
        "articles": [170],
        "expect": {170: "التعويض"},
        "question": "كيف يقدر القاضي مدى التعويض عن الضرر، وما حكمه إذا لم يتيسر له تعيينه تعييناً نهائياً؟",
        "difficulty": "hard",
        "topic": "التعويض",
    },
    {
        "articles": [375],
        "expect": {375: "حق دوري"},
        "question": "ما مدة التقادم في الحقوق الدورية المتجددة، وما الأمثلة التي أوردها النص عليها؟",
        "difficulty": "hard",
        "topic": "التقادم",
    },
    {
        "articles": [1030],
        "expect": {1030: "يكسب الدائن"},
        "question": "ما هو الرهن الرسمي وما الحق الذي يكسبه الدائن بمقتضاه؟",
        "difficulty": "medium",
        "topic": "الرهن",
    },
    {
        "articles": [915],
        "expect": {915: "الشريعة"},
        "question": "ما القانون الذي تسري به أحكام الوصية وفقاً للقانون المدني المصري؟",
        "difficulty": "medium",
        "topic": "الوصية",
    },
    {
        "articles": [1130],
        "expect": {1130: "ولوية"},
        "question": "ما هو الامتياز، وهل يكون للحق امتياز بدون نص في القانون؟",
        "difficulty": "easy",
        "topic": "الامتياز",
    },
]


# --- Type 6: paraphrased_question (15) ---
# The article number is deliberately NOT mentioned: the question
# restates the legal issue in other words.
PARAPHRASED_QUESTIONS = [
    {
        # English paraphrase: the context/answer come from the article's
        # English column. Each English question uses a different article
        # from the selected Arabic ones, so nothing is a translated
        # duplicate of another question in the dataset.
        "articles": [935],
        "expect": {935: "رخصة"},
        "question": "What is preemption, and when may a person take the purchaser's place in the sale of immovable property?",
        "language": "en",
        "difficulty": "easy",
        "topic": "Preemption",
    },
    {
        "articles": [772],
        "expect": {772: "الكفالة عقد"},
        "question": "What is a contract of suretyship, and what does the surety undertake towards the creditor?",
        "language": "en",
        "difficulty": "medium",
        "topic": "Suretyship",
    },
    {
        "articles": [487],
        "expect": {487: "قبلها"},
        "question": "When is a gift complete, and who is allowed to accept it on behalf of the donee?",
        "language": "en",
        "difficulty": "medium",
        "topic": "Gift",
    },
    {
        "articles": [374],
        "expect": {374: "خمس عشرة"},
        "question": "After how many years does an ordinary obligation become prescribed under the Egyptian Civil Code?",
        "language": "en",
        "difficulty": "easy",
        "topic": "Prescription",
    },
    {
        "articles": [418],
        "expect": {418: "البيع عقد"},
        "question": "كيف يُعرَّف عقد البيع وما هو محل التزام البائع فيه؟",
        "difficulty": "easy",
        "topic": "البيع",
    },
    {
        "articles": [558],
        "expect": {558: "يجار عقد"},
        "question": "ما هو عقد الإيجار وما التزام المؤجر بمقتضاه؟",
        "difficulty": "easy",
        "topic": "الإيجار",
    },
    {
        "articles": [374],
        "expect": {374: "خمس عشرة"},
        "question": "بعد انقضاء كم سنة يتقادم الالتزام وفقاً للقاعدة العامة في القانون المدني المصري؟",
        "difficulty": "easy",
        "topic": "التقادم",
    },
    {
        "articles": [601],
        "expect": {601: "بموت"},
        "question": "هل ينقضي عقد الإيجار بموت المؤجر أو بموت المستأجر؟",
        "difficulty": "easy",
        "topic": "الإيجار",
    },
    {
        "articles": [772],
        "expect": {772: "الكفالة عقد"},
        "question": "ما هو عقد الكفالة وما الذي يتعهد به الكفيل تجاه الدائن؟",
        "difficulty": "medium",
        "topic": "الكفالة",
    },
    {
        "articles": [773],
        "expect": {773: "الكتابة"},
        "question": "هل يصح إثبات الكفالة بالبينة أم يجب أن تثبت بالكتابة؟",
        "difficulty": "medium",
        "topic": "الكفالة",
    },
    {
        "articles": [487],
        "expect": {487: "قبلها"},
        "question": "ما الذي تتطلبه صحة الهبة من جانب الموهوب له؟",
        "difficulty": "medium",
        "topic": "الهبة",
    },
    {
        "articles": [935],
        "expect": {935: "رخصة"},
        "question": "ما هي الرخصة التي تجيز لصاحبها الحلول محل المشتري في بيع العقار؟",
        "difficulty": "medium",
        "topic": "الشفعة",
    },
    {
        "articles": [916],
        "expect": {916: "مرض الموت"},
        "question": "ما حكم التصرف الذي يصدر من شخص في مرض الموت ويكون مقصوداً به التبرع؟",
        "difficulty": "medium",
        "topic": "الوصية",
    },
    {
        "articles": [1001],
        "expect": {1001: "الميراث"},
        "question": "كيف ينتقل حق المحتكر إلى غيره من الأشخاص؟",
        "difficulty": "medium",
        "topic": "الاحتكار",
    },
    {
        "articles": [776],
        "expect": {776: "صحيحة"},
        "question": "ما الشرط الذي لا تكون الكفالة صحيحة بدونه من حيث الالتزام المكفول؟",
        "difficulty": "hard",
        "topic": "الكفالة",
    },
    {
        "articles": [500],
        "expect": {500: "يرجع"},
        "question": "هل يجوز للواهب أن يرجع في هبته، وما الحكم إذا لم يقبل الموهوب له الرجوع؟",
        "difficulty": "hard",
        "topic": "الهبة",
    },
    {
        "articles": [376],
        "expect": {376: "بخمس سنوات"},
        "question": "ما مدة تقادم حقوق أصحاب المهن الحرة مثل الأطباء والمحامين والمهندسين؟",
        "difficulty": "hard",
        "topic": "التقادم",
    },
    {
        "articles": [377],
        "expect": {377: "ثالث سنوات"},
        "question": "ما مدة تقادم الضرائب والرسوم المستحقة للدولة؟",
        "difficulty": "hard",
        "topic": "التقادم",
    },
    {
        "articles": [602],
        "expect": {602: "حرفة"},
        "question": "إذا أُبرم الإيجار بسبب يتعلق بحرفة المستأجر ثم مات، فما حكم العقد؟",
        "difficulty": "hard",
        "topic": "الإيجار",
    },
]


# --- Type 5: multi_article (10) ---
# Each question needs information from TWO articles, so it can only be
# answered correctly when retrieval returns both of them.
MULTI_ARTICLE_QUESTIONS = [
    {
        "articles": [89, 147],
        "expect": {89: "يتم العقد بمجرد", 147: "شريعة المتعاقدين"},
        "question": "كيف ينعقد العقد في القانون المدني المصري، وما القوة الملزمة التي تترتب عليه بعد انعقاده؟",
        "difficulty": "hard",
        "topic": "انعقاد العقد وآثاره",
    },
    {
        "articles": [147, 148],
        "expect": {147: "شريعة المتعاقدين", 148: "حسن"},
        "question": "ما القوة الملزمة للعقد، وكيف يجب تنفيذ ما اشتمل عليه وفقاً لما يوجبه القانون المدني؟",
        "difficulty": "hard",
        "topic": "تنفيذ العقد",
    },
    {
        "articles": [163, 169],
        "expect": {163: "خطأ", 169: "متضامنين"},
        "question": "ما أساس المسؤولية عن الفعل الضار، وكيف تكون المسؤولية إذا تعدد المسؤولون عن الضرر؟",
        "difficulty": "hard",
        "topic": "المسؤولية التقصيرية",
    },
    {
        "articles": [164, 165],
        "expect": {164: "مميز", 165: "سبب أجنبى"},
        "question": "ما حكم مسؤولية المميز عن أعماله غير المشروعة، وما أثر ثبوت أن الضرر نشأ عن سبب أجنبي؟",
        "difficulty": "hard",
        "topic": "المسؤولية التقصيرية",
    },
    {
        "articles": [166, 167],
        "expect": {166: "دفاع شرعى", 167: "صدر إليه من رئيس"},
        "question": "ما حكم من أحدث ضرراً أثناء دفاعه الشرعي، وما حكم الموظف العام الذي أضر بالغير تنفيذاً لأمر رئيسه؟",
        "difficulty": "hard",
        "topic": "أسباب الإعفاء من المسؤولية",
    },
    {
        "articles": [558, 601],
        "expect": {558: "يجار عقد", 601: "بموت"},
        "question": "ما تعريف عقد الإيجار في القانون المدني المصري، وهل ينقضي هذا العقد بموت المؤجر أو المستأجر؟",
        "difficulty": "medium",
        "topic": "الإيجار",
    },
    {
        "articles": [558, 605],
        "expect": {558: "يجار عقد", 605: "ملكية العين المؤجرة"},
        "question": "ما التزام المؤجر في عقد الإيجار، وما حكم الإيجار إذا انتقلت ملكية العين المؤجرة إلى شخص آخر؟",
        "difficulty": "hard",
        "topic": "الإيجار",
    },
    {
        "articles": [487, 500],
        "expect": {487: "قبلها", 500: "يرجع"},
        "question": "ما الذي تتطلبه صحة الهبة من جانب الموهوب له، وهل يجوز للواهب الرجوع في هبته؟",
        "difficulty": "hard",
        "topic": "الهبة",
    },
    {
        "articles": [772, 776],
        "expect": {772: "الكفالة عقد", 776: "صحيحة"},
        "question": "ما تعريف عقد الكفالة، وما شرط صحة الالتزام المكفول لصحة هذا العقد؟",
        "difficulty": "hard",
        "topic": "الكفالة",
    },
    {
        "articles": [375, 376],
        "expect": {375: "حق دوري", 376: "بخمس سنوات"},
        "question": "ما مدة تقادم الحقوق الدورية المتجددة، وما مدة تقادم حقوق أصحاب المهن مثل الأطباء والمحامين؟",
        "difficulty": "medium",
        "topic": "التقادم",
    },
]


# ==========================================================
# Eligibility + deterministic selection
# ==========================================================
# Articles skipped by the data-driven question types:
#   * repealed articles  -> they carry a repeal notice, not a provision
#   * very short texts   -> nothing meaningful to ask about
#   * the pre-books       (1-2) -> publication/implementation wording
MIN_ARABIC_LENGTH = 80

# Question-type -> number of questions (25 in total).
QUESTION_COUNTS = {
    "article_lookup": 5,
    "article_identification": 4,
    "legal_concept": 4,
    "specific_provision": 3,
    "multi_article": 3,
    "paraphrased_question": 6,
}

# CHALLENGING/DIVERSE questions per type. These are the `hard` ones; the
# types missing from this mapping contribute only general questions, so
# the total is 1 (concept) + 3 (multi-article) + 1 (paraphrased) = 5.
CHALLENGING_COUNTS = {
    "legal_concept": 1,
    "multi_article": 3,
    "paraphrased_question": 1,
}

# Maximum Arabic length for the template-driven types. It keeps every
# generated question in the "general/direct" group: article_identification
# is medium below 200 characters, and specific_provision below 400, so the
# stricter 200 bound covers both.
IDENTIFICATION_MAX_ARABIC_LENGTH = 200
PROVISION_MAX_ARABIC_LENGTH = 400

# Language split per type, and the totals they must produce. The dataset
# is deliberately almost half English, so the bilingual corpus is really
# exercised in both directions (13 Arabic + 12 English = 25).
LANGUAGE_BY_TYPE = {
    "article_lookup": {"ar": 2, "en": 3},
    "article_identification": {"ar": 2, "en": 2},
    "legal_concept": {"ar": 3, "en": 1},
    "specific_provision": {"ar": 1, "en": 2},
    "multi_article": {"ar": 3, "en": 0},
    "paraphrased_question": {"ar": 2, "en": 4},
}
EXPECTED_LANGUAGE_COUNTS = {"ar": 13, "en": 12}

# Language split of the template-driven types (the curated types use
# LANGUAGE_BY_TYPE above, resolved by the curated selection).
LOOKUP_ENGLISH = LANGUAGE_BY_TYPE["article_lookup"]["en"]
IDENTIFICATION_ENGLISH = LANGUAGE_BY_TYPE["article_identification"]["en"]
PROVISION_ENGLISH = LANGUAGE_BY_TYPE["specific_provision"]["en"]


def select_curated(table, count, challenging_count, english_count):
    """Pick `count` curated entries with an exact language split.

    Returns `challenging_count` hard entries plus general ones, of which
    exactly `english_count` are English. Selection follows table order, so
    the dataset is deterministic, and it fails loudly if the table cannot
    supply the requested mix.
    """
    hard = [entry for entry in table if entry["difficulty"] == "hard"]
    general = [entry for entry in table if entry["difficulty"] != "hard"]

    challenging = hard[:challenging_count]
    needed_general = count - challenging_count
    needed_arabic = needed_general - english_count

    english = [entry for entry in general if entry.get("language", "ar") == "en"][
        :english_count
    ]
    arabic = [entry for entry in general if entry.get("language", "ar") != "en"][
        :needed_arabic
    ]

    if len(english) < english_count or len(arabic) < needed_arabic:
        raise SystemExit(
            f"Curated table cannot supply {count} questions "
            f"({challenging_count} hard, {english_count} English, "
            f"{needed_arabic} Arabic): found {len(hard)} hard, {len(english)} "
            f"English, {len(arabic)} Arabic."
        )

    return challenging + english + arabic


def curated_article_numbers():
    """Every article number already used by a curated question."""
    used = set()
    for table in (CONCEPT_QUESTIONS, PARAPHRASED_QUESTIONS, MULTI_ARTICLE_QUESTIONS):
        for entry in table:
            used.update(entry["articles"])
    return used


def build_eligible_pool(articles, excluded):
    """Return article numbers usable by the template-driven question types."""
    pool = []
    for article in articles:
        number = article["article_number"]
        if number in excluded:
            continue
        if number < 3:
            continue
        if article["is_repealed"]:
            continue
        arabic = article["text"]["ar"]
        english = article["text"]["en"]
        if len(arabic.strip()) < MIN_ARABIC_LENGTH or not english.strip():
            continue
        # Require a confident topic label so metadata is never guessed.
        if derive_topic(arabic) == "أحكام عامة":
            continue
        pool.append(number)
    return sorted(pool)


def choose_stride(size):
    """Pick a stride that visits every pool entry exactly once."""
    candidate = 7
    while True:
        if size % candidate != 0:
            return candidate
        candidate += 2


def spread_selection(pool, offset, count):
    """Pick `count` pool entries spread evenly across the pool."""
    stride = choose_stride(len(pool))
    selected = []
    for step in range(count):
        index = (offset + step * stride) % len(pool)
        selected.append(pool[index])
    return selected


def verify_curated(by_number):
    """Fail loudly if a curated question asserts wording that is not there."""
    problems = []
    for table in (CONCEPT_QUESTIONS, PARAPHRASED_QUESTIONS, MULTI_ARTICLE_QUESTIONS):
        for entry in table:
            for number, needle in entry["expect"].items():
                article = by_number.get(number)
                if article is None:
                    problems.append(f"article {number} does not exist")
                    continue
                if article["is_repealed"]:
                    problems.append(f"article {number} is repealed")
                if not article["text"]["ar"].strip():
                    problems.append(f"article {number} has no Arabic text")
                if needle not in article["text"]["ar"]:
                    problems.append(
                        f"article {number} does not contain {needle!r} "
                        f"(question: {entry['question'][:45]}...)"
                    )
    if problems:
        raise SystemExit(
            "Golden dataset generation aborted - curated questions do not "
            "match the source data:\n  - " + "\n  - ".join(problems)
        )


# ==========================================================
# Record builder
# ==========================================================
SOURCE_LABEL = "data/processed/articles.json"


def make_record(
    question_id,
    language,
    question_type,
    difficulty,
    question,
    articles,
    by_number,
    topic=None,
    ground_truth=None,
    label_articles=False,
):
    """Build one golden-dataset record straight from the source articles."""
    contexts = [by_number[number]["text"][language] for number in articles]

    if ground_truth is None:
        if label_articles and len(articles) > 1:
            # Multi-article answers label each quoted article so the
            # ground truth is unambiguous about which text came from where.
            if language == "ar":
                ground_truth = "\n\n".join(
                    f"المادة {number}:\n{text}"
                    for number, text in zip(articles, contexts)
                )
            else:
                ground_truth = "\n\n".join(
                    f"Article {number}:\n{text}"
                    for number, text in zip(articles, contexts)
                )
        else:
            ground_truth = "\n\n".join(contexts)

    return {
        "id": question_id,
        "language": language,
        "question_type": question_type,
        "difficulty": difficulty,
        "question": question,
        "ground_truth": ground_truth,
        "relevant_articles": [str(number) for number in articles],
        "ground_truth_contexts": contexts,
        "metadata": {
            "source": SOURCE_LABEL,
            "topic": topic or derive_topic(by_number[articles[0]]["text"]["ar"]),
            "requires_multiple_articles": len(articles) > 1,
        },
    }


# ==========================================================
# Type 1: article_lookup
# ==========================================================
LOOKUP_TEMPLATES_AR = (
    "ما هو نص المادة {n} من القانون المدني المصري؟",
    "اذكر نص المادة {n} من القانون المدني المصري.",
    "ما الذي تنص عليه المادة {n} من القانون المدني المصري؟",
    "أورد نص المادة {n} من القانون المدني المصري كما ورد فيه.",
)
LOOKUP_TEMPLATE_EN = "What is the text of Article {n} of the Egyptian Civil Code?"


def build_lookup(number, index, total, by_number):
    """Build one article_lookup question (easy: single article, direct)."""
    is_english = index >= total - LOOKUP_ENGLISH
    if is_english:
        return {
            "question": LOOKUP_TEMPLATE_EN.format(n=number),
            "language": "en",
            "difficulty": "easy",
        }
    return {
        "question": LOOKUP_TEMPLATES_AR[index % len(LOOKUP_TEMPLATES_AR)].format(
            n=number
        ),
        "language": "ar",
        "difficulty": "easy",
    }


# ==========================================================
# Type 2: article_identification
# ==========================================================
IDENTIFICATION_TEMPLATES_AR = (
    "أي مادة من القانون المدني المصري تنص على ما يلي: «{frag}»؟",
    "حدد رقم المادة التي ورد فيها النص الآتي من القانون المدني المصري: «{frag}»",
)
IDENTIFICATION_TEMPLATE_EN = (
    "Which article of the Egyptian Civil Code contains the following "
    "provision: “{frag}”?"
)


def build_identification(number, index, total, by_number):
    """Build one article_identification question from the article's own text."""
    is_english = index >= total - IDENTIFICATION_ENGLISH
    language = "en" if is_english else "ar"
    source_text = by_number[number]["text"][language]
    fragment = excerpt(source_text, 100)

    # Longer provisions are harder to pin down to a single article.
    difficulty = "hard" if len(by_number[number]["text"]["ar"]) > 200 else "medium"

    if is_english:
        question = IDENTIFICATION_TEMPLATE_EN.format(frag=fragment)
    else:
        question = IDENTIFICATION_TEMPLATES_AR[index % 2].format(frag=fragment)

    if language == "ar":
        answer = f"المادة {number} من القانون المدني المصري: {source_text}"
    else:
        answer = f"Article {number} of the Egyptian Civil Code: {source_text}"

    return {
        "question": question,
        "language": language,
        "difficulty": difficulty,
        "ground_truth": answer,
    }


# ==========================================================
# Type 4: specific_provision
# ==========================================================
PROVISION_TEMPLATES_AR = (
    "ما الحكم الذي قررته المادة {n} من القانون المدني المصري بشأن «{frag}»؟",
    "وضّح ما تقرره المادة {n} من القانون المدني المصري فيما يتعلق بـ «{frag}».",
)
PROVISION_TEMPLATE_EN = (
    "What does Article {n} of the Egyptian Civil Code provide regarding “{frag}”?"
)


def build_provision(number, index, total, by_number):
    """Build one specific_provision question anchored on the article text."""
    is_english = index >= total - PROVISION_ENGLISH
    language = "en" if is_english else "ar"
    source_text = by_number[number]["text"][language]
    fragment = excerpt(source_text, 80)

    # Long, multi-paragraph provisions need more context -> harder.
    difficulty = "hard" if len(by_number[number]["text"]["ar"]) > 400 else "medium"

    if is_english:
        question = PROVISION_TEMPLATE_EN.format(n=number, frag=fragment)
    else:
        question = PROVISION_TEMPLATES_AR[index % 2].format(n=number, frag=fragment)

    return {
        "question": question,
        "language": language,
        "difficulty": difficulty,
    }


# ==========================================================
# Assembly
# ==========================================================
def build_all_questions(articles, by_number):
    """Build the 25 records (deterministic order, one id per question)."""
    verify_curated(by_number)

    # Article numbers already used by curated questions are kept out of
    # the template-driven types so every type shows different articles.
    pool = build_eligible_pool(articles, curated_article_numbers())
    lookup_count = QUESTION_COUNTS["article_lookup"]
    identification_count = QUESTION_COUNTS["article_identification"]
    provision_count = QUESTION_COUNTS["specific_provision"]
    needed = lookup_count + identification_count + provision_count
    if len(pool) < needed:
        raise SystemExit(
            f"Article pool too small: {len(pool)} eligible articles, "
            f"but {needed} are needed."
        )

    # The template-driven types must stay in the general/direct group, so
    # they are drawn from articles short enough to keep every generated
    # question "easy"/"medium". One pool (filtered once) is then split
    # into three disjoint blocks, so no article is used by two types.
    template_pool = [
        number
        for number in pool
        if len(by_number[number]["text"]["ar"]) <= IDENTIFICATION_MAX_ARABIC_LENGTH
    ]
    if len(template_pool) < needed:
        raise SystemExit(
            f"Template pool too small: {len(template_pool)} short articles, "
            f"but {needed} are needed."
        )

    lookup_numbers = spread_selection(template_pool, offset=0, count=lookup_count)
    identification_numbers = spread_selection(
        template_pool, offset=lookup_count, count=identification_count
    )
    provision_numbers = spread_selection(
        template_pool,
        offset=lookup_count + identification_count,
        count=provision_count,
    )

    # The three slices must not overlap (guarantees article diversity).
    slices = [lookup_numbers, identification_numbers, provision_numbers]
    flattened = [n for group in slices for n in group]
    if len(set(flattened)) != len(flattened):
        raise SystemExit(
            "Template selection produced duplicate articles across question types."
        )

    records = []

    # 1. article_lookup
    for index, number in enumerate(lookup_numbers):
        spec = build_lookup(number, index, lookup_count, by_number)
        records.append(
            make_record(
                None,
                spec["language"],
                "article_lookup",
                spec["difficulty"],
                spec["question"],
                [number],
                by_number,
            )
        )

    # 2. article_identification
    for index, number in enumerate(identification_numbers):
        spec = build_identification(number, index, identification_count, by_number)
        records.append(
            make_record(
                None,
                spec["language"],
                "article_identification",
                spec["difficulty"],
                spec["question"],
                [number],
                by_number,
                ground_truth=spec["ground_truth"],
            )
        )

    # 3. legal_concept (curated: a challenging one + general ones)
    for entry in select_curated(
        CONCEPT_QUESTIONS,
        QUESTION_COUNTS["legal_concept"],
        CHALLENGING_COUNTS["legal_concept"],
        LANGUAGE_BY_TYPE["legal_concept"]["en"],
    ):
        records.append(
            make_record(
                None,
                entry.get("language", "ar"),
                "legal_concept",
                entry["difficulty"],
                entry["question"],
                entry["articles"],
                by_number,
                topic=entry["topic"],
                label_articles=len(entry["articles"]) > 1,
            )
        )

    # 4. specific_provision
    for index, number in enumerate(provision_numbers):
        spec = build_provision(number, index, provision_count, by_number)
        records.append(
            make_record(
                None,
                spec["language"],
                "specific_provision",
                spec["difficulty"],
                spec["question"],
                [number],
                by_number,
            )
        )

    # 5. multi_article (curated - every one of these is challenging, because
    #    no single article can answer the question)
    for entry in select_curated(
        MULTI_ARTICLE_QUESTIONS,
        QUESTION_COUNTS["multi_article"],
        CHALLENGING_COUNTS["multi_article"],
        LANGUAGE_BY_TYPE["multi_article"]["en"],
    ):
        records.append(
            make_record(
                None,
                "ar",
                "multi_article",
                entry["difficulty"],
                entry["question"],
                entry["articles"],
                by_number,
                topic=entry["topic"],
                label_articles=True,
            )
        )

    # 6. paraphrased_question (curated)
    for entry in select_curated(
        PARAPHRASED_QUESTIONS,
        QUESTION_COUNTS["paraphrased_question"],
        CHALLENGING_COUNTS["paraphrased_question"],
        LANGUAGE_BY_TYPE["paraphrased_question"]["en"],
    ):
        records.append(
            make_record(
                None,
                entry.get("language", "ar"),
                "paraphrased_question",
                entry["difficulty"],
                entry["question"],
                entry["articles"],
                by_number,
                topic=entry["topic"],
            )
        )

    # Sequential ids in build order.
    for position, record in enumerate(records, start=1):
        record["id"] = f"q_{position:03d}"

    return records


# ==========================================================
# Validation
# ==========================================================
REQUIRED_FIELDS = (
    "id",
    "language",
    "question_type",
    "difficulty",
    "question",
    "ground_truth",
    "relevant_articles",
    "ground_truth_contexts",
    "metadata",
)
LANGUAGES = {"ar", "en"}
DIFFICULTIES = {"easy", "medium", "hard"}

# Expected size of the dataset and of its two groups.
EXPECTED_QUESTION_COUNT = 25
# "challenging/diverse" == the hard questions; everything else is
# "general/direct". The split is approximate by design.
EXPECTED_CHALLENGING = sum(CHALLENGING_COUNTS.values())


def validate(records, by_number):
    """Check the dataset contract and its traceability to the source data."""
    problems = []

    if len(records) != EXPECTED_QUESTION_COUNT:
        problems.append(
            f"expected exactly {EXPECTED_QUESTION_COUNT} questions, "
            f"found {len(records)}"
        )

    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        problems.append("duplicate question ids")

    for record in records:
        tag = record.get("id", "?")

        for field in REQUIRED_FIELDS:
            if field not in record:
                problems.append(f"{tag}: missing field {field}")

        if record["language"] not in LANGUAGES:
            problems.append(f"{tag}: bad language {record['language']}")
        if record["difficulty"] not in DIFFICULTIES:
            problems.append(f"{tag}: bad difficulty {record['difficulty']}")
        if not str(record["question"]).strip():
            problems.append(f"{tag}: empty question")
        if not str(record["ground_truth"]).strip():
            problems.append(f"{tag}: empty ground_truth")
        if not record["ground_truth_contexts"]:
            problems.append(f"{tag}: empty ground_truth_contexts")

        for key in ("source", "topic", "requires_multiple_articles"):
            if key not in record["metadata"]:
                problems.append(f"{tag}: metadata missing {key}")

        for number_text in record["relevant_articles"]:
            number = int(number_text)
            article = by_number.get(number)
            if article is None:
                problems.append(f"{tag}: article {number} does not exist")
                continue
            if article["is_repealed"]:
                problems.append(f"{tag}: article {number} is repealed")
            source_text = article["text"][record["language"]]
            if not source_text.strip():
                problems.append(
                    f"{tag}: article {number} has no {record['language']} text"
                )
                continue
            # The ground truth must quote the stored source text...
            if source_text not in record["ground_truth"]:
                problems.append(
                    f"{tag}: ground_truth is not derived from article {number}"
                )
            # ...and the context must BE the stored source text.
            if source_text not in record["ground_truth_contexts"]:
                problems.append(f"{tag}: ground_truth_contexts miss article {number}")

        # Multi-article questions must really need more than one article.
        needs_multiple = record["metadata"]["requires_multiple_articles"]
        if needs_multiple != (len(record["relevant_articles"]) > 1):
            problems.append(f"{tag}: inconsistent requires_multiple_articles")

    # Every question type must be present in its planned quantity.
    type_counts = Counter(record["question_type"] for record in records)
    for question_type, expected in QUESTION_COUNTS.items():
        if type_counts.get(question_type, 0) != expected:
            problems.append(
                f"type {question_type}: expected {expected}, "
                f"found {type_counts.get(question_type, 0)}"
            )

    # All six question types must actually be represented.
    missing_types = set(QUESTION_COUNTS) - set(type_counts)
    if missing_types:
        problems.append(f"question types not represented: {sorted(missing_types)}")

    # The challenging/diverse group (the hard questions) must be present
    # in roughly the planned quantity, and the language split must be exact.
    challenging = sum(1 for record in records if record["difficulty"] == "hard")
    if challenging != EXPECTED_CHALLENGING:
        problems.append(
            f"expected {EXPECTED_CHALLENGING} challenging (hard) questions, "
            f"found {challenging}"
        )

    language_counts = Counter(record["language"] for record in records)
    for language, expected in EXPECTED_LANGUAGE_COUNTS.items():
        if language_counts.get(language, 0) != expected:
            problems.append(
                f"expected {expected} {language} questions, "
                f"found {language_counts.get(language, 0)}"
            )

    # Per-type language split, so one type cannot be single-language.
    for question_type, expected_split in LANGUAGE_BY_TYPE.items():
        for language, expected in expected_split.items():
            actual = sum(
                1
                for record in records
                if record["question_type"] == question_type
                and record["language"] == language
            )
            if actual != expected:
                problems.append(
                    f"type {question_type}: expected {expected} {language}, "
                    f"found {actual}"
                )

    # No two questions may ask the same thing.
    seen_questions = [record["question"].strip().lower() for record in records]
    if len(set(seen_questions)) != len(seen_questions):
        problems.append("duplicate questions found in the dataset")

    if problems:
        raise SystemExit(
            "Golden dataset validation failed:\n  - " + "\n  - ".join(problems)
        )

    return type_counts


# ==========================================================
# Main
# ==========================================================
def main():
    """Build, validate and write data/golden/golden_dataset.json."""
    # Keep Arabic readable in the Windows console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    articles = load_articles()
    by_number = {article["article_number"]: article for article in articles}

    records = build_all_questions(articles, by_number)
    type_counts = validate(records, by_number)

    payload = {
        "version": DATASET_VERSION,
        "source": SOURCE_LABEL,
        "question_count": len(records),
        "question_types": list(QUESTION_COUNTS),
        "questions": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        # ensure_ascii=False keeps the Arabic questions readable.
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    difficulty_counts = Counter(record["difficulty"] for record in records)
    language_counts = Counter(record["language"] for record in records)
    challenging = difficulty_counts.get("hard", 0)

    print("=" * 64)
    print("GOLDEN DATASET BUILT")
    print("=" * 64)
    print(f"Source articles : {ARTICLES_PATH}")
    print(f"Output          : {OUTPUT_PATH}")
    print(f"Questions       : {len(records)}")
    print(f"Languages       : {dict(language_counts)}")
    print(f"Difficulty      : {dict(difficulty_counts)}")
    print(
        f"Groups          : general/direct = {len(records) - challenging}, "
        f"challenging/diverse = {challenging}"
    )

    print("-" * 64)
    print("Question types:")
    for question_type in QUESTION_COUNTS:
        print(f"  {question_type:<26} {type_counts[question_type]}")
    print("-" * 64)
    print("Sample questions:")
    for record in records[:2] + records[20:22] + records[-2:]:
        print(
            f"  [{record['id']}] ({record['question_type']}, "
            f"{record['difficulty']}, {record['language']}) "
            f"articles={record['relevant_articles']}"
        )
        print(f"        {record['question'][:110]}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
