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
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import bentoml

from rag_project.config.settings import get_drift_config, get_evaluation_config
from rag_project.generation.chain import (
    build_rag_components,
    format_docs,
    format_sources,
)
from service.llm_client import (
    generate_from_prompt,
    get_vllm_config,
    is_configured,
    stream_from_prompt,
)

# Observability / guardrail / drift modules used by the JSON endpoint.
# These are all lightweight and imported once at startup.
from guardrails import pii_detector
from monitoring import metrics as m
from monitoring import evaluate_embedding_drift as drift
from observability import langfuse_tracing as lf

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


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag (1/true/yes/on), else *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Default OUTPUT guardrail policy: redact in place and continue, so the
# legal content of a generated answer survives while every detected PII
# value is removed before the answer leaves the service. Set
# GUARDRAIL_OUTPUT_BLOCK=1 to apply the strict default policy instead
# (national id -> withhold the whole answer).
_OUTPUT_MASK_ALL = {
    pii_detector.EGYPTIAN_NATIONAL_ID: pii_detector.MASK,
    pii_detector.EGYPTIAN_MOBILE: pii_detector.MASK,
    pii_detector.EMAIL: pii_detector.MASK,
}


def start_metrics_server() -> None:
    """Expose THIS process's Prometheus metrics on a small HTTP server.

    The BentoML process records token usage, drift, guardrail and RAGAS
    metrics in its own registry, which is separate from the FastAPI
    process. Without this server those numbers would never be scrapeable.
    Prometheus scrapes the exported endpoint; it never reads metrics.py.

    Port comes from BENTOML_METRICS_PORT (default 9001; set empty/0 to
    disable). Failures (port already in use) are logged and never take
    the service down.
    """
    raw = os.getenv("BENTOML_METRICS_PORT", "9001").strip()
    if raw in ("", "0"):
        return
    try:
        from prometheus_client import start_http_server

        port = int(raw)
        start_http_server(port)
        logger.info("Prometheus metrics exposed on :%d/metrics", port)
    except Exception as error:  # noqa: BLE001 - metrics must never break serving
        logger.warning(
            "Could not start Prometheus metrics server: %s",
            type(error).__name__,
        )


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


class _RequestRejected(Exception):
    """Request rejected by validation or a guardrail.

    The message is deliberately client-safe (no stack, no secrets, no
    internal details): BentoML may surface it in the HTTP error response,
    and the FastAPI gateway only ever returns generic messages anyway.
    """


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

        # Expose this process's Prometheus metrics (tokens, drift, guardrail,
        # RAGAS) so Prometheus can scrape them separately from FastAPI.
        start_metrics_server()

        # Lazy-cached drift baseline (loaded once, never recomputed per request).
        self._drift_baseline = "__unset__"

        # Reuse the SAME embedding model that retrieval already loaded, for
        # the optional embedding-drift check. Extracted defensively from the
        # retriever's underlying vector store so we never load BGE-M3 twice.
        # If the attribute layout differs, drift is simply skipped (it is an
        # optional signal and must never break a request).
        self._query_embedder = self._resolve_query_embedder(self.retriever)

        # الرسالة دي معناها إن الـservice جاهزة تستقبل /ask.
        logger.info("RAG components ready - service startup completed")

    @staticmethod
    def _resolve_query_embedder(retriever):
        """Best-effort extraction of a single-query embedding callable.

        Tries the common LangChain / Chroma attribute paths. Returns None if
        none is found, in which case drift evaluation is skipped.
        """
        # VectorStoreRetriever exposes the vector store as `.vectorstore`.
        vectorstore = getattr(retriever, "vectorstore", None)
        for obj in (vectorstore, retriever):
            if obj is None:
                continue
            # Chroma (langchain_chroma) stores it as `_embedding_function`.
            embed_fn = getattr(obj, "_embedding_function", None) or getattr(
                obj, "embedding_function", None
            )
            if embed_fn is not None and hasattr(embed_fn, "embed_query"):
                return embed_fn.embed_query
            if embed_fn is not None and callable(embed_fn):
                return embed_fn
        return None

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
    async def ask(
        self, question: str, langfuse_trace_id: str = ""
    ) -> AsyncIterator[str]:
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

    # ==========================================================
    # Helpers for the non-streaming JSON endpoint (ask_json)
    # ==========================================================
    @staticmethod
    def _online_eval_enabled() -> bool:
        """Are per-request RAGAS faithfulness calls enabled?

        An LLM-as-judge call per request adds latency and cost, so it is
        configurable: the ONLINE_EVAL_ENABLED environment variable wins
        over configs/config.yaml (evaluation.enabled). Set
        ONLINE_EVAL_ENABLED=0 to disable the online judge.
        """
        raw = os.getenv("ONLINE_EVAL_ENABLED")
        if raw is not None:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(get_evaluation_config().get("enabled", False))

    async def _evaluate_drift(self, question: str, trace_id: str) -> None:
        """Score the query against the cached reference baseline (optional).

        Reuses the SAME embedding function the retriever already loaded
        (one extra embed_query call per request - documented in the
        README). The baseline JSON is loaded ONCE and cached on the
        instance; the 50 reference embeddings are never recomputed here.

        FAILURE POLICY (distinguishes detector failure from real drift):
        drift is a signal, never a reason to fail a request, so a detector
        failure is swallowed and the pipeline CONTINUES. But it is never
        silent: the failure is counted in
        ``rag_embedding_drift_errors_total`` and written to the drift
        report with ``status="error"``, so "the detector is broken" is
        observable separately from "this query drifted". Only the
        exception TYPE is recorded, never the raw embedding or question.
        """
        if not _env_flag("DRIFT_ENABLED", True):
            return
        if self._query_embedder is None:
            return
        try:
            if self._drift_baseline == "__unset__":
                try:
                    self._drift_baseline = drift.load_baseline()
                except FileNotFoundError:
                    logger.info("Drift baseline missing - drift check disabled")
                    self._drift_baseline = None
            if not self._drift_baseline:
                return

            with lf.span("embedding_drift", as_type="embedding", trace_id=trace_id):
                query_embedding = await asyncio.to_thread(
                    self._query_embedder, question
                )
                report = drift.evaluate_query(
                    query_embedding, baseline=self._drift_baseline
                )

            m.record_drift(
                "bentoml", report.get("max_similarity"), bool(report.get("drift"))
            )
            # Privacy-safe report: masked/truncated question + correlation id.
            drift.append_drift_report(
                {
                    "timestamp": report.get("timestamp"),
                    "trace_id": trace_id,
                    "question": pii_detector.mask(question)[:160],
                    "model": report.get("model"),
                    "max_similarity": report.get("max_similarity"),
                    "threshold": report.get("threshold"),
                    "closest_reference_id": report.get("closest_reference_id"),
                    "drift": report.get("drift"),
                    "status": "ok",
                }
            )
        except Exception as drift_error:  # noqa: BLE001 - see the docstring
            # Detector failure: continue the request, but make it visible.
            m.record_drift_error("bentoml", type(drift_error).__name__)
            try:
                drift.append_drift_report(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trace_id": trace_id,
                        "question": pii_detector.mask(question)[:160],
                        "model": get_drift_config().get("model"),
                        "error_type": type(drift_error).__name__,
                        "status": "error",
                    }
                )
            except Exception:  # noqa: BLE001 - reporting must never raise
                logger.debug("drift error report skipped", exc_info=True)
            logger.warning(
                "Drift detector failed (%s); continuing without a drift score.",
                type(drift_error).__name__,
            )

    @staticmethod
    def _record_token_usage(model: str, answer: str, prompt_value, usage: dict) -> None:
        """Exact usage from vLLM metadata, else a clearly-labelled estimate.

        Streamed chunks are NEVER counted as tokens. When the server sends
        no usage object, the fallback goes to rag_token_estimates_total
        (separate metric) so an estimate can never be mistaken for exact.
        """
        exact_in = usage.get("input_tokens")
        exact_out = usage.get("output_tokens")
        if exact_in or exact_out:
            m.record_tokens("bentoml", model, exact_in, exact_out)
            m.record_cost_from_usage("bentoml", model, exact_in, exact_out)
            return
        try:
            prompt_chars = len(prompt_value.to_string())
        except Exception:  # noqa: BLE001 - estimate must never raise
            prompt_chars = 0
        if prompt_chars:
            m.record_token_estimate(
                "bentoml", model, "input", max(1, prompt_chars // 4)
            )
        if answer:
            m.record_token_estimate(
                "bentoml", model, "output", max(1, len(answer) // 4)
            )

    async def _evaluate_faithfulness(
        self, question: str, answer: str, documents, trace_id: str
    ):
        """Optional online RAGAS faithfulness -> score attached to THIS trace.

        The judge runs in a worker thread (it manages its own event loop).
        A failure is recorded as a metric and swallowed: the answer has
        already been generated and guarded, so an unavailable evaluation
        backend must not fail the user's request.
        """
        with lf.span("ragas_faithfulness", as_type="evaluator", trace_id=trace_id):
            try:
                from rag_project.evaluation.faithfulness import (
                    evaluate_faithfulness,
                    format_contexts,
                )

                result = await asyncio.to_thread(
                    evaluate_faithfulness,
                    question,
                    answer,
                    format_contexts(documents),
                )
                score = float(result.score)
                m.record_ragas("bentoml", "success", score)
                lf.score_trace(
                    trace_id, "faithfulness", score, comment=f"backend={result.backend}"
                )
                return score
            except Exception as eval_error:  # noqa: BLE001
                logger.warning(
                    "faithfulness evaluation failed: %s", type(eval_error).__name__
                )
                logger.debug("eval error", exc_info=True)
                m.record_ragas("bentoml", "failure")
                return None

    # ==========================================================
    # /ask_json endpoint - NON-STREAMING JSON contract
    # ----------------------------------------------------------
    # The FastAPI gateway's POST /ask forwards HERE and expects the
    # structured contract back:
    #     {"answer": str, "sources": [str, ...], "langfuse_trace_id": str}
    #
    # Every value below is a LOCAL - no request state is stored on self -
    # so concurrent requests can never see each other's question,
    # context, sources or answer. Errors are raised as _RequestRejected
    # with client-safe messages: never a stack trace, key or secret.
    # ==========================================================

    @bentoml.api
    async def ask_json(self, question: str, langfuse_trace_id: str = "") -> dict:
        request_start = time.perf_counter()
        trace_id = langfuse_trace_id or lf.new_trace_id()
        endpoint = "/ask_json"

        # --- validation -------------------------------------------------
        if not question or not question.strip():
            m.record_request("bentoml", endpoint, "invalid", 0.0)
            raise _RequestRejected("The question is empty.")

        # Missing vLLM config must fail clearly - never fall back to
        # loading a large model locally.
        if not is_configured():
            m.record_error("bentoml", endpoint, "vllm_not_configured")
            m.record_request("bentoml", endpoint, "error", 0.0)
            raise _RequestRejected(
                "The LLM is not configured (VLLM_MODEL is not set in the "
                "environment or configs/config.yaml)."
            )

        # --- 1) embedding drift (optional signal) -----------------------
        await self._evaluate_drift(question, trace_id)

        try:
            # --- 2) retrieval (existing retriever + Chroma) --------------
            # PII-masked copy for Langfuse: raw input PII must never be
            # sent to the trace backend even if the gateway was bypassed.
            safe_question = pii_detector.mask(question)
            with lf.span(
                "retrieval",
                as_type="retriever",
                trace_id=trace_id,
                input={"question": safe_question},
            ) as retrieval_span:
                documents = await asyncio.to_thread(self.retriever.invoke, question)
                if retrieval_span is not None:
                    try:
                        retrieval_span.update(metadata={"documents": len(documents)})
                    except Exception:  # noqa: BLE001 - tracing is optional
                        pass

            # --- 3) prompt construction (existing prompt, unchanged) -----
            with lf.span(
                "prompt_construction",
                trace_id=trace_id,
                metadata={"documents": len(documents)},
            ):
                prompt_value = await self.prompt.ainvoke(
                    {
                        "context": format_docs(documents),
                        "question": question,
                    }
                )

            # --- 4) generation through the LLM client (remote vLLM) ------
            usage: dict = {}
            model = get_vllm_config().get("model") or "unknown"
            with lf.span(
                "llm_generation",
                as_type="generation",
                trace_id=trace_id,
                metadata={"model": model},
            ) as generation:
                # generate_from_prompt is sync; running it in a worker
                # thread keeps the event loop free AND lets it use its own
                # asyncio.run path instead of blocking this loop.
                answer = await asyncio.to_thread(
                    generate_from_prompt, prompt_value, usage
                )
                if generation is not None:
                    try:
                        generation.update(
                            usage_details={
                                "input": int(usage.get("input_tokens") or 0),
                                "output": int(usage.get("output_tokens") or 0),
                            }
                        )
                    except Exception:  # noqa: BLE001 - tracing is optional
                        pass

            # --- token metrics: exact usage, else labelled estimate ------
            self._record_token_usage(model, answer, prompt_value, usage)

            # --- 5) output guardrail ------------------------------------
            with lf.span(
                "output_guardrail",
                as_type="guardrail",
                trace_id=trace_id,
                metadata={"phase": "output"},
            ):
                scan = pii_detector.detect(answer)
                policy = (
                    None
                    if _env_flag("GUARDRAIL_OUTPUT_BLOCK", False)
                    else _OUTPUT_MASK_ALL
                )
                action = pii_detector.decide(scan, policy)
                m.record_guardrail(
                    "bentoml", "output", action, ",".join(scan.kinds) or "none"
                )
                if action == pii_detector.BLOCK:
                    m.record_request(
                        "bentoml",
                        endpoint,
                        "blocked",
                        time.perf_counter() - request_start,
                    )
                    raise _RequestRejected(
                        "The response was withheld by the output guardrail."
                    )
                if action == pii_detector.MASK:
                    answer = pii_detector.mask(answer)

            # --- 6) sources: real article citations, never chunk ids -----
            sources = format_sources(documents)

            # --- 7) optional RAGAS faithfulness -> THIS trace ------------
            faithfulness = None
            if self._online_eval_enabled():
                faithfulness = await self._evaluate_faithfulness(
                    question, answer, documents, trace_id
                )

            duration = time.perf_counter() - request_start
            m.record_request("bentoml", endpoint, "success", duration)

            response = {
                "answer": answer,
                "sources": [str(s) for s in sources],
                "langfuse_trace_id": trace_id,
            }
            if faithfulness is not None:
                response["faithfulness"] = faithfulness

            lf.flush()
            return response

        except _RequestRejected:
            # Already recorded above; propagate the client-safe message.
            raise
        except Exception as error:  # noqa: BLE001
            duration = time.perf_counter() - request_start
            logger.error("ask_json failed: %s", type(error).__name__)
            logger.debug("full error", exc_info=True)
            m.record_error("bentoml", endpoint, type(error).__name__)
            m.record_request("bentoml", endpoint, "error", duration)
            # Client-safe message: no stack trace, no secrets, no internals.
            raise _RequestRejected(
                "The RAG service failed to generate an answer."
            ) from None

    # ==========================================================
    # /health endpoint - Liveness check
    # ----------------------------------------------------------
    # Does NOT initialize the RAG chain or contact the LLM.
    # Reports service status, vLLM config presence, and component readiness.
    # ==========================================================

    @bentoml.api
    def health(self) -> dict:
        """Liveness check for the BentoML RAG service."""
        vllm_cfg = get_vllm_config()
        return {
            "status": "ok",
            "service": "bentoml-rag-service",
            "vllm_configured": bool(vllm_cfg.get("model")),
            "vllm_base_url": vllm_cfg.get("base_url"),
            "retriever_ready": self.retriever is not None,
            "prompt_ready": self.prompt is not None,
            "drift_baseline_loaded": self._drift_baseline is not None
            and self._drift_baseline != "__unset__",
            "tracing_enabled": lf.is_enabled(),
            "metrics_port": int(os.getenv("BENTOML_METRICS_PORT", "9001")),
        }


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
#   uv run bentoml serve service.bentoml_service:RAGService --port 3000
#
# الـvLLM شغال على port 8000 (remote GPU)،
# لذلك BentoML شغال على port 3000.
#
# للتأكد إن الـstreaming شغال (chunks بتنزل تدريجيًا):
#
#   curl -N -X POST http://127.0.0.1:3000/ask \
#     -H "Content-Type: application/json" \
#     -d "{\"question\": \"...\"}"
#
# للتأكد إن الـhealth endpoint شغال:
#
#   curl http://127.0.0.1:3000/health
#
# مفيش تشغيل يدوي للـservice هنا؛
# BentoML بتشغّل RAGService لما ننادي `bentoml serve`.
# ==========================================================
