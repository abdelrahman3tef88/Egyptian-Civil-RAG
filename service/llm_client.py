# ==========================================================
# llm_client.py - بوابة الاتصال بـLLM server البعيد (vLLM)
# ==========================================================
#
# الملف ده مسؤول بس عن الـcommunication مع الـvLLM server اللي هيشتغل
# بعدين على GPU بعيدة. مفيش أي RAG logic هنا: لا retrieval، لا prompt،
# لا embeddings - الاتصال بس.
#
# الرد المطلوب هيكون كده:
#     الـRAG prompt  ->  الـclient ده  ->  remote vLLM HTTP API  ->  answer
#
# المرحلة دي بنجهز الـintegration point بس. الـvLLM server لسه مش شغال
# والاتصال فعلًا مش هيتنفذ دلوقتي.
#
# >>> اتحدّث: دلوقتي الـadapter ده هو نقطة الدمج الفعلية بتاعة الـRAG
#     chain (chain.py). الـexisting prompt هو اللي بيتعمله convert لرسالة
#     واحدة بس، مش بنعيد بناء أي system prompt.
#
# >>> اتحدّث: الـstreaming بقى متاح فعلًا عبر generate_stream()
#     (token-by-token من vLLM) و stream_from_prompt() اللي بياخد ناتج
#     الـexisting prompt مباشرة. الـnon-streaming (generate()) زي ما هو.

import asyncio
from concurrent.futures import ThreadPoolExecutor

import os

from dotenv import load_dotenv

# بنحمّل ملف .env مرة واحدة هنا عشان الـsecrets تفضل جوّاه
# ومتحطّش جوه الكود أبدًا.
load_dotenv()

# الـdefaults دي placeholders آمنة للتطوير المحلي بس، مش URLs حقيقية.
# vLLM default port is 8000 (OpenAI-compatible /v1 endpoint).
# In production, VLLM_BASE_URL must point to the REMOTE GPU server.
#
# Resolution order for every value (see get_vllm_config):
#   environment variable  ->  configs/config.yaml (vllm section)  ->  default below
# The environment always wins, so the real remote URL, model and API
# key are never read from a file that is committed to git. The YAML
# section is the project's declared default for this serving path and
# was previously ignored entirely, leaving timeout_seconds and
# max_tokens dead configuration.
_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = ""
_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_MAX_TOKENS = 4096


def _config_defaults() -> dict:
    """Read the ``vllm`` section of configs/config.yaml.

    Returns an empty dict when the config file is unavailable, so the
    client degrades to its safe local defaults instead of raising.
    """
    try:
        from rag_project.config.settings import get_vllm_config as _yaml_config

        return dict(_yaml_config())
    except Exception:  # noqa: BLE001 - config absence must not break the client
        return {}


def _coerce_float(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def get_vllm_config() -> dict:
    """Resolve the vLLM endpoint settings (no hardcoded production values).

    ``base_url``, ``model`` and ``api_key`` describe HOW to reach the
    REMOTE GPU server; ``timeout_seconds`` and ``max_tokens`` come from
    the declared configuration and are applied to every request.
    """
    defaults = _config_defaults()
    return {
        # Env var first, then configs/config.yaml, then the local default.
        "base_url": (
            os.getenv("VLLM_BASE_URL", "").strip()
            or str(defaults.get("base_url") or _DEFAULT_BASE_URL).strip()
        ),
        "model": (
            os.getenv("VLLM_MODEL", "").strip()
            or str(defaults.get("model") or _DEFAULT_MODEL).strip()
        ),
        # The API key is ONLY ever read from the environment: a secret must
        # never be picked up from a file that is committed to the repo.
        "api_key": os.getenv("VLLM_API_KEY", "").strip(),
        "timeout_seconds": _coerce_float(
            defaults.get("timeout_seconds"), _DEFAULT_TIMEOUT_SECONDS
        ),
        "max_tokens": _coerce_int(defaults.get("max_tokens"), _DEFAULT_MAX_TOKENS),
    }


def is_configured() -> bool:
    """هل الإعدادات جاهزة فعلًا عشان نتصل بالـserver؟"""
    config = get_vllm_config()
    # من غير موديل مفيش داعي نتصل، والنقطة دي بتخلّي الاستخدام واضح
    # إنه "مش متظبط لسه" بدل ما يرمي error غريب.
    return bool(config["model"])


def create_async_client():
    """نعمل async OpenAI-compatible client للـvLLM.

    الـvLLM server بيرصد OpenAI-compatible API، عشان كده بنستخدم الـclient
    بتاع OpenAI مباشرة من غير ما نبني حاجة من الصفر.
    """
    # بنستورد جوّا الدالة عشان المشروع الأساسي ميتحمّلش الـopenai
    # وهو مش محتاجه (الاستخدام ده للـserving بس).
    from openai import AsyncOpenAI

    config = get_vllm_config()
    return AsyncOpenAI(
        base_url=config["base_url"],
        # لو الـkey فاضي، نمرّره None عشان الـlocal server ميطلبوش.
        api_key=config["api_key"] or "not-needed",
        # api_key=config["api_key"] or None,
        # Bound the wait for the remote GPU server (configs/config.yaml
        # vllm.timeout_seconds) so an unavailable endpoint fails fast
        # instead of hanging the BentoML worker on the SDK default.
        timeout=config["timeout_seconds"],
    )


async def generate(messages: list[dict], usage: dict | None = None) -> str:
    """نبعت الرسائل للـvLLM server ونرجع الرد كـtext (غير مجزّأ).

    ده المسار الـnon-streaming: بيرجّع الإجابة كاملة مرة واحدة.
    لو محتاج الرد ينزل token-by-token استخدم generate_stream() تحت.

    usage (اختياري): لو المستدعي مرّر dict فاضي، بنعمّله بـtoken usage
    الحقيقي من الـresponse (prompt_tokens / completion_tokens) لما الـserver
    يبعتها. مبنعدّش الـchunks كـtokens أبدًا - والـstreaming مالوش usage هنا.
    """
    if not is_configured():
        raise RuntimeError(
            "VLLM_MODEL is not set (neither the environment nor "
            "configs/config.yaml provides a model). Set it in .env "
            "before using this client."
        )

    client = create_async_client()
    config = get_vllm_config()

    # استدعاء واحد بسيط. مفيش retry ولا connection pool هنا
    # عشان المرحلة دي مجرد تجهيز ومش بنطلب تعقيد زيادة.
    response = await client.chat.completions.create(
        model=config["model"],
        messages=messages,
        stream=False,
        # Cap the generation length at the configured budget
        # (configs/config.yaml vllm.max_tokens).
        max_tokens=config["max_tokens"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # Token usage الحقيقي من الـserver (لو بعته). مبنعدّش chunks ولا نخمّن.
    if usage is not None:
        _capture_usage(response, usage)

    # بنرجّع أول choice بس - الرد النهائي كنص عادي.
    return response.choices[0].message.content or ""


def _capture_usage(response, usage: dict) -> None:
    """ننسخ الـtoken usage من الـresponse للـdict اللي المستدعي مرّره.

    ملاحظة مهمة: القيم دي بتيجي من الـprovider metadata بس. لو الـserver
    مبعتهاش، بنسيب الـdict زي ما هو (فاضي/ناقص) والمستدعي يتعامل معاها
    كـ"مش متاحة" - مش كتقدير مصحوب بكلمة "exact".
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return
    prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
    completion_tokens = getattr(usage_obj, "completion_tokens", None)
    if prompt_tokens is not None:
        usage["input_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        usage["output_tokens"] = int(completion_tokens)


# ==========================================================
# Streaming: الرد ينزل من vLLM token-by-token
# ----------------------------------------------------------
# الـvLLM بيرصد OpenAI-compatible API، فبنستخدم نفس الـclient
# بتاعنا (create_async_client) وبس بنفعّل stream=True.
# ==========================================================
async def generate_stream(messages: list[dict]):
    """ينزّل الرد من vLLM كـchunks صغيرة (async generator).

    بيطرح كل token/text chunk لوحده، عشان نقدر نبعته للعميل
    على طول ونقيس TTFT و TBT من غير ما نستنى الإجابة كلها.

    مش بنشيل generate() الـnon-streaming - الاتنين موجودين مع بعض.
    """
    if not is_configured():
        raise RuntimeError(
            "VLLM_MODEL is not set (neither the environment nor "
            "configs/config.yaml provides a model). Set it in .env "
            "before using this client."
        )

    client = create_async_client()
    config = get_vllm_config()

    # نفس استدعاء generate() بالظبط، الفرق stream=True بس.
    response = await client.chat.completions.create(
        model=config["model"],
        messages=messages,
        stream=True,
        # Qwen3 modes: نطفّي الـ"thinking" عشان الرد يبدأ فورًا
        # وميتأخرش ورا reasoning text مخفي (وبيأثر على TTFT).
        max_tokens=config["max_tokens"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # كل chunk بيرجّع list of choices؛ ممكن تكون فاضية في الـkeep-alives.
    async for chunk in response:
        if not chunk.choices:
            continue

        # delta.content بتبقى None في أول chunks (role) وبتبقى نص بعدين.
        delta = chunk.choices[0].delta.content

        if delta:
            yield delta


def build_vllm_messages(question: str, context: str) -> list[dict]:
    """نجهّز الـmessages للـvLLM server من السؤال والـcontext.

    ملاحظة: الدالة دي مش مستخدمة من الـRAG chain حاليًا.
    سببها إن الـchain عنده existing prompt في generation/prompts.py
    وفيه legal RAG instructions مهمة، فبنستخدمه هو بدل ما نبني system
    prompt عام من هنا. الدالة مسجّلة بس للاستخدام التاني/التجريبي.
    """
    system = "You are an AI assistant that answers using only the given context."
    user = f"Context:\n{context}\n\nQuestion:\n{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ==========================================================
# Adapter: LangChain PromptValue  ->  OpenAI-style messages
# ----------------------------------------------------------
# ده أهم جزء في الملف. الـexisting prompt (generation/prompts.py)
# بيرجّع PromptValue، واحنا عايزين نبعت النص النهائي اللي خرج منه
# زي ما هو للـvLLM، من غير إعادة بناء أي prompt جديد.
# ==========================================================
# ربط أنواع رسائل LangChain بأنواع OpenAI.
# (LangChain بيقول human/ai/system، والـOpenAI API عايز user/assistant/system)
_LANGCHAIN_TO_OPENAI_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
}


def prompt_value_to_messages(prompt_value) -> list[dict]:
    """نحوّل الناتج النهائي من الـexisting prompt لرسائل OpenAI-compatible.

    ده بالظبط اللي طلبته: ناخد final_prompt_text ونحوّلّه لـ:
        {"role": "user", "content": final_prompt_text}
    مبنغيرش ولا حرف من الـprompt اللي الـRAG بيبنيه.
    """
    # PromptValue ليه to_messages() و to_string()، فبنستخدم to_messages()
    # عشان نحافظ على الأدوار (system/human) لو الـprompt اتغيّر بعدين.
    return [
        {
            # لو اللِّغة مش معروفة، نعاملها كرسالة مستخدم (الأمان أكتر).
            "role": _LANGCHAIN_TO_OPENAI_ROLE.get(
                getattr(message, "type", "human"), "user"
            ),
            "content": message.content,
        }
        for message in prompt_value.to_messages()
    ]


def _run_async(coroutine):
    """ننفّذ async function من سياق sync بأمان.

    الـRAG chain بتاع المشروع بيتشغل بـ .invoke() (sync)، فمحتاجين
    نشغّل generate() دي. النقطة الحساسة: لو اتعلّقنا جوّا event loop
    شغال (زي الـBentoML async endpoint) هنوقع بـ
    "cannot be called from a running event loop"، عشان كده بنفحص الأول.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # مفيش loop شغال → ننفّذ العادي.
        return asyncio.run(coroutine)

    # في سياق async فعلًا → نلفّها في thread منفصلة عشان مانحبسش
    # الـloop الرئيسي، والخدمة تفضل تستقبل طلبات مع بعض.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def generate_from_prompt(prompt_value, usage: dict | None = None) -> str:
    """نمرّر الـPromptValue الناتج من الـexisting prompt للـvLLM ونرجع الرد.

    دي الدالة اللي الـRAG chain بيستدعيها بعد ما الـcontext والسؤال
    اتجمّعوا في الـprompt.

    usage (اختياري): لو اتمرّر dict فاضي، بنسجّل فيه token usage الحقيقي
    من الـresponse. غير كده مالوش أي تأثير على النتيجة.
    """
    messages = prompt_value_to_messages(prompt_value)
    # TODO: Streaming will be implemented after the remote vLLM integration is ready.
    # TODO: TTFT and TBT will be measured during the streaming phase.
    return _run_async(generate(messages, usage=usage))


def build_vllm_llm():
    """نرجّع "LLM" قابل للاستخدام جوّا الـLCEL chain (PromptValue -> str).

    الاستبدال الوحيد في الـpipeline: بدل llm_module.create_llm() (Gemini)
    بنستدعي الدالة دي، وكل حاجة تانية (retrieval, format_docs, prompt)
    زي ما هي بالظبط.
    """
    # بنستورد جوّا الدالة عشان الـservice package ميتحمّلش في كل مرة
    # يتم استيراد chain.py (وفي الاختبارات).
    from langchain_core.runnables import RunnableLambda

    # RunnableLambda بتاخد الـPromptValue (مخرج الـprompt) وترجّع str،
    # وبكده StrOutputParser اللي بعدها يشتغل عادي من غير أي تغيير.
    return RunnableLambda(generate_from_prompt)


# ==========================================================
# Streaming adapter: ناتج الـexisting prompt  ->  tokens
# ----------------------------------------------------------
# ده نظير stream للمسار الـnon-streaming:
#     generate_from_prompt()   -> str واحدة   (موجود فوق)
#     stream_from_prompt()     -> tokens       (جديد)
#
# الفرق الوحيد إن ده بيستخدم generate_stream() بدل generate().
# الـretrieval والـprompt زي ما هم تمامًا.
# ==========================================================
async def stream_from_prompt(prompt_value):
    """نمرّر ناتج الـexisting prompt للـvLLM ونطرح الـtokens واحدة واحدة.

    مرجع الاستخدام: bentoml_service.ask() بتعمل `async for` عليها
    وبتبعت كل chunk للعميل على طول.
    """
    # نفس التحويل تمامًا اللي بيستخدمه المسار الـnon-streaming،
    # فمش بنعيد بناء أي prompt هنا.
    messages = prompt_value_to_messages(prompt_value)

    async for token in generate_stream(messages):
        yield token
