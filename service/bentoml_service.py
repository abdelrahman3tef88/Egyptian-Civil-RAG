# ==========================================================
# bentoml_service.py - طبقة الـserving المحلية
# ==========================================================
#
# الهدف ده بسيط: نلفّ الـRAG chain الموجود بـBentoML ونطلّع /ask.
#
# Flow بتاع الطلب:
#     User question
#       -> BentoML /ask
#       -> generation/chain.py  (الـchain الموجود فعلًا)
#       -> الـretrieval الموجود
#       -> الـcontext الموجود
#       -> الـprompt الموجود
#       -> LLM client
#       -> Generated answer
#
# مهم جدًا:
#   * مفيش تكرار لـretrieval جوّا الملف ده.
#   * مفيش prompt تاني متكتوب هنا.
#   * مفيش تكوين رابع للـRAG chain.
#   * /ask هنا streaming: بيرجّع chunks تدريجيًا مع قياس response_time
#     و TTFT و TBT في اللوج (باستخدام time.perf_counter()).

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import bentoml

from rag_project.generation.chain import build_rag_components, format_docs
from service.llm_client import stream_from_prompt

# ==========================================================
# Logging بسيط ومؤقت (development logging بس)
# ----------------------------------------------------------
# مفيش نظام logging كامل في المشروع، فبنجهز إعداد صغير يساعدنا نشوف:
#   * الـservice بدأت
#   * الـRAG chain بيتبني
#   * الـRAG chain بقى جاهز
#   * الطلب وصل
#   * الـRAG شغل
#   * الـRAG خلص
#   * حصل error
#
# مبنجّوش أي:
#   * API key
#   * secrets
#   * credentials
#   * config كامل
# ==========================================================

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# مؤقت: بنضيف مجلد logs وقت التشغيل وبس.
_LOG_DIR.mkdir(exist_ok=True)


def setup_logging() -> logging.Logger:
    """إعداد logging مؤقت ومباشر للـservice."""

    logger = logging.getLogger("rag_service")

    if logger.handlers:
        # لو الإعداد اتعمل قبل كده منتجهاش handlers زيادة.
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # على الـconsole عشان نشوفه فورًا أثناء التطوير.
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # وفي ملف مؤقت عشان نرجّعه بعد كده لو حصل مشكلة.
    # (ده development logging بس - مش جزء من أي monitoring حقيقي.)
    file_handler = logging.FileHandler(
        _LOG_DIR / "rag_service.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# ==========================================================
# BentoML service
# ----------------------------------------------------------
# الـRAG chain بيتبني مرة واحدة أثناء startup بدل ما يتبني
# عند أول request.
#
# ده مهم لأن الـembedding model والـChroma تقال، وأول request
# كان ممكن يستنى وقت طويل ويتضرب بسبب BentoML request timeout.
#
# بعد startup:
#
#   BentoML starts
#        ↓
#   build_rag_chain()
#        ↓
#   BGE-M3 + Chroma + Retriever + Prompt + vLLM adapter
#        ↓
#   RAG chain ready
#        ↓
#   /ask يستقبل requests
#
# ==========================================================


@bentoml.service
class RAGService:
    """خدمة BentoML بتلفّ الـRAG chain الموجود."""

    def __init__(self):
        # نفس الـretrieval wiring اللي بيستخدمه build_rag_chain()
        # (نسخة واحدة بس من الكود) - من غير LLM عشان محتاجين نstream.
        #
        # بتتبني مرة واحدة وقت startup بدل ما نستنى أول request،
        # عشان تحميل الـembedding model والـChroma ياخدوا وقت.
        logger.info("Service startup - building the existing RAG components")

        self.retriever, self.prompt = build_rag_components()

        # الرسالة دي معناها إن الـservice جاهزة تستقبل /ask.
        logger.info("RAG components ready - service startup completed")

    # ==========================================================
    # /ask endpoint - STREAMING
    # ----------------------------------------------------------
    # POST /ask بيرجّع الإجابة chunk-by-chunk (مش JSON واحدة).
    #
    # Flow بتاع الطلب:
    #   question
    #     -> الـretriever الموجود          (نفسه اللي في الـchain)
    #     -> format_docs() الموجود         (من غير أي تعديل)
    #     -> الـexisting prompt            (generation/prompts.py)
    #     -> vLLM عبر stream_from_prompt() (service/llm_client)
    #     -> chunks تتبعت للعميل واحدة واحدة
    #
    # الـasync عشان الخدمة تستقبل أكتر من request.
    # ==========================================================

    @bentoml.api
    async def ask(self, question: str) -> AsyncIterator[str]:
        # داخل الدالة بنمرّر السؤال لوحده. مفيش ground truth
        # ولا metadata ولا difficulty بتتبعت للـgeneration.

        if not question or not question.strip():
            logger.warning("Request received - empty question rejected")

            # نرجّع رسالة واضحة من غير ما نكشف أي حاجة بالداخل.
            yield "Error: the question is empty."
            return

        # ----------------------------------------------------------
        # قياس التوقيت: بنستخدم time.perf_counter() عشان يبقى دقيق
        # ومضبوط (مش مرتبطة بساعة النظام).
        #
        #   request_start    -> لحظة استلام الطلب
        #   retrieval_end    -> لحظة رجوع الـretriever (BGE-M3 + Chroma)
        #   first_token_time -> أول token نزل من vLLM = TTFT
        #   previous_token_time -> عشان نحسب الفرق بين tokens = TBT
        #
        # التقسيم في اللوج بيبقى:
        #   retrieval  = retrieval_end - request_start
        #   generation = response_time - retrieval  (كل زمن vLLM)
        # ----------------------------------------------------------
        request_start = time.perf_counter()
        first_token_time = None
        previous_token_time = None
        tbt_values = []
        chunks_sent = 0

        try:
            logger.info("RAG execution started")

            # 1) retrieval - الـretriever الموجود، مش بنعيد بناءه.
            #    sync (LangChain .invoke) فبنلفّه في to_thread عشان
            #    مانحبسش الـevent loop.
            #
            #    التوقيت هنا بيقيس الاستدعاء الموجود نفسه (BGE-M3 embedding
            #    للسؤال + بحث Chroma) - مفيش أي استدعاء إضافي للـembedding
            #    عشان القياس، بنوقّت اللي بيحصل فعلًا بس.
            retrieval_end = None
            documents = await asyncio.to_thread(
                self.retriever.invoke,
                question,
            )
            retrieval_end = time.perf_counter()

            # 2) نفس format_docs() الموجود بالظبط (من غير أي تعديل).
            prompt_value = await self.prompt.ainvoke(
                {
                    "context": format_docs(documents),
                    "question": question,
                }
            )

            # 3) vLLM streaming - كل chunk بيتبعت للعميل فورًا.
            async for chunk in stream_from_prompt(prompt_value):
                now = time.perf_counter()

                if first_token_time is None:
                    # أول chunk نزل: ده TTFT.
                    first_token_time = now
                else:
                    # الفرق عن الـchunk اللي قبله = TBT للوقفة دي.
                    tbt_values.append(now - previous_token_time)

                previous_token_time = now

                chunks_sent += 1
                yield chunk

            # ----------------------------------------------------------
            # بعد ما الإجابة كلها نزلت:
            #   response_time = المدة الكلية
            #   ttft          = لحد أول chunk
            #   tbt           = متوسط الفواصل بين الـchunks
            # ----------------------------------------------------------
            response_time = time.perf_counter() - request_start

            if first_token_time is None:
                # مفيش chunks أصلاً (الرد فاضي) - بنكتبه عشان نعرف
                # إن المشكلة مش في التوقيت.
                logger.error("RAG execution completed - no chunks received")
            else:
                ttft = first_token_time - request_start
                average_tbt = sum(tbt_values) / len(tbt_values) if tbt_values else 0.0

                # تفصيل المراحل (كلها time.perf_counter()، بنفس وحدات الثواني):
                #   retrieval_s    = نهاية الـretrieval - بداية الطلب.
                #                  (BGE-M3 embedding للسؤال + بحث Chroma)
                #   generation_s   = نهاية الرد - نهاية الـretrieval.
                #                  (كل زمن vLLM: الانتظار + التنزيل token-by-token)
                # مجموع الاتنين = response_time (إلا فرق تجهيز الـprompt
                # البسيط اللي جوه الفاصلتين، عشان كده بنحسبها صراحة).
                retrieval_s = retrieval_end - request_start
                generation_s = response_time - retrieval_s

                logger.info(
                    "RAG execution completed | response_time=%.3fs "
                    "| retrieval=%.3fs | generation=%.3fs "
                    "| ttft=%.3fs | tbt=%.4fs | chunks=%d",
                    response_time,
                    retrieval_s,
                    generation_s,
                    ttft,
                    average_tbt,
                    chunks_sent,
                )

        except Exception as error:  # noqa: BLE001
            # نسجّل الـerror كامل جوّا اللوج (بالداخل بس)،
            # ونرجّع للعميل رسالة عامة من غير stack trace ولا secrets.
            logger.error(
                "RAG execution failed: %s",
                type(error).__name__,
            )

            logger.debug(
                "full error",
                exc_info=True,
            )

            # لو حصل error بعد ما بدأت نبعت chunks بنوقف ونسجّله،
            # وبنبعت رسالة خطأ واحدة بس لو لسه مبعتناش حاجة.
            if first_token_time is None:
                yield ("Error: could not generate an answer. Please try again.")


# ملاحظة: /ask هنا streaming (يرجع chunks) عشان المقاييس token-level
# (TTFT / TBT) تطلع من بيانات حقيقية، مش من تقديرات:
#
#   Locust events      <-  من قراية الـstream الفعلي
#   reports/locust_report.html  <-  من Locust نفسه (--html)
#
# مفيش GPU utilization هنا عمدًا - البنود المطلوبة هي Response Time
# و P50 و P95 و P99 و Requests/sec و TTFT و TBT بس.
#
# ==========================================================


# ==========================================================
# Local development
# ----------------------------------------------------------
# لتشغيل BentoML محليًا (والـvLLM لازم يكون شغال الأول):
#
#   uv run bentoml serve service.bentoml_service:RAGService --port 8001
#
# الـvLLM شغال على port 8000،
# لذلك BentoML شغال على port 8001.
#
# للتأكد إن الـstreaming شغال (chunks بتنزل تدريجيًا):
#
#   curl -N -X POST http://127.0.0.1:8001/ask \
#     -H "Content-Type: application/json" \
#     -d "{\"question\": \"...\"}"
#
# مفيش تشغيل يدوي للـservice هنا؛
# BentoML بتشغّل RAGService لما ننادي `bentoml serve`.
# ==========================================================
