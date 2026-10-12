"""Prometheus metrics for the Egyptian Civil Code RAG services.

These are the REAL application metrics, exported through a Prometheus
``/metrics`` endpoint. Prometheus scrapes this endpoint; it does NOT read
this Python file as data. The ``start_http_server`` call (or the FastAPI /
BentoML integration) is what actually exposes the numbers.

Cardinality rules
-----------------
Labels are kept deliberately LOW-cardinality. Never put a request id, a full
question, raw PII, or an arbitrary trace id in a Prometheus label: that would
explode the time-series count. Only bounded values (service name, HTTP
endpoint path, coarse status) are allowed.

Token accounting
----------------
Token counters are fed from the vLLM ``usage`` metadata when the server
provides it. Streamed chunks are NOT counted as tokens. If usage metadata is
unavailable, the caller must not silently pretend a value is exact: use
``record_token_estimate`` so it is clearly labelled as an estimate.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ==========================================================
# Requests
# ==========================================================
rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG requests handled.",
    ["service", "endpoint", "status"],
)
rag_request_errors_total = Counter(
    "rag_request_errors_total",
    "Total RAG requests that failed.",
    ["service", "endpoint", "error_type"],
)
rag_request_duration_seconds = Histogram(
    "rag_request_duration_seconds",
    "RAG request duration in seconds.",
    ["service", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

# ==========================================================
# Tokens (from vLLM usage metadata, NOT streamed chunk counts)
# ==========================================================
rag_input_tokens_total = Counter(
    "rag_input_tokens_total",
    "Input (prompt) tokens consumed, from provider usage metadata.",
    ["service", "model"],
)
rag_output_tokens_total = Counter(
    "rag_output_tokens_total",
    "Output (completion) tokens generated, from provider usage metadata.",
    ["service", "model"],
)
rag_token_estimates_total = Counter(
    "rag_token_estimates_total",
    "Token counts recorded as an ESTIMATE (exact usage metadata unavailable).",
    ["service", "model", "direction"],
)

# ==========================================================
# Embedding drift
# ==========================================================
rag_embedding_drift_total = Counter(
    "rag_embedding_drift_total",
    "Potential query embedding drift events (max similarity below threshold).",
    ["service"],
)
rag_embedding_drift_errors_total = Counter(
    "rag_embedding_drift_errors_total",
    "Embedding drift DETECTOR failures (could not score the query). "
    "Distinguishes a broken detector from an actual drift event: drift "
    "means 'scored below threshold', an error means 'not scored at all'.",
    ["service", "error_type"],
)
rag_embedding_similarity = Gauge(
    "rag_embedding_similarity",
    "Most recent maximum cosine similarity of a query to the reference set.",
    ["service"],
)

# ==========================================================
# Guardrails
# ==========================================================
rag_guardrail_actions_total = Counter(
    "rag_guardrail_actions_total",
    "Guardrail actions taken (detections, masking, blocks).",
    ["service", "stage", "action", "kind"],
)
rag_guardrail_blocked_total = Counter(
    "rag_guardrail_blocked_total",
    "Requests blocked by a guardrail.",
    ["service", "stage"],
)

# ==========================================================
# RAGAS faithfulness evaluation
# ==========================================================
rag_ragas_evaluations_total = Counter(
    "rag_ragas_evaluations_total",
    "Faithfulness evaluations attempted, by outcome.",
    ["service", "status"],  # status: success | failure
)
rag_ragas_faithfulness = Gauge(
    "rag_ragas_faithfulness",
    "Most recent RAGAS faithfulness score (0.0 - 1.0).",
    ["service"],
)


# ==========================================================
# Cost (OPTIONAL, opt-in only)
# ----------------------------------------------------------
# The counter below is incremented ONLY when pricing is EXPLICITLY
# configured by the operator via environment variables:
#   MODEL_PRICING_INPUT_PER_1K_USD
#   MODEL_PRICING_OUTPUT_PER_1K_USD
# (price in USD per 1,000 tokens). If either variable is unset, no cost
# number is ever produced - the metric simply stays at zero and the
# Grafana cost panel shows $0. We invent NO provider prices here.
# ==========================================================
rag_estimated_cost_usd_total = Counter(
    "rag_estimated_cost_usd_total",
    "Estimated generation cost in USD, recorded ONLY when explicit "
    "per-1k-token pricing is configured by the operator.",
    ["service", "model"],
)


# ==========================================================
# Helper functions (keep callers from touching labels directly)
# ==========================================================
def record_cost_from_usage(service, model, input_tokens=None, output_tokens=None):
    """Accumulate estimated USD cost using EXPLICIT operator pricing only.

    Reads MODEL_PRICING_INPUT_PER_1K_USD / MODEL_PRICING_OUTPUT_PER_1K_USD.
    If either price is unset or unparsable, NOTHING is recorded (the cost
    metric stays at zero). No provider price is ever invented here.
    Returns the recorded USD amount (0.0 when pricing is not configured).
    """
    import os

    costs = []
    for tokens, env_name in (
        (input_tokens, "MODEL_PRICING_INPUT_PER_1K_USD"),
        (output_tokens, "MODEL_PRICING_OUTPUT_PER_1K_USD"),
    ):
        if not tokens:
            continue
        try:
            price = float(os.getenv(env_name, "").strip())
        except (TypeError, ValueError, AttributeError):
            continue
        if price > 0:
            costs.append(tokens * price / 1000.0)
    if not costs:
        return 0.0
    total = sum(costs)
    rag_estimated_cost_usd_total.labels(service=service, model=model).inc(total)
    return total


def record_request(service, endpoint, status, duration_seconds=None):
    """Record a completed request (count + optional duration)."""
    rag_requests_total.labels(service=service, endpoint=endpoint, status=status).inc()
    if duration_seconds is not None:
        rag_request_duration_seconds.labels(service=service, endpoint=endpoint).observe(
            duration_seconds
        )


def record_error(service, endpoint, error_type):
    """Record a failed request."""
    rag_request_errors_total.labels(
        service=service, endpoint=endpoint, error_type=error_type
    ).inc()


def record_tokens(service, model, input_tokens=None, output_tokens=None):
    """Record exact token usage from provider metadata.

    Only pass values that came from the provider's ``usage`` object. Do not
    pass a streamed chunk count here; use ``record_token_estimate`` for that.
    """
    if input_tokens:
        rag_input_tokens_total.labels(service=service, model=model).inc(input_tokens)
    if output_tokens:
        rag_output_tokens_total.labels(service=service, model=model).inc(output_tokens)


def record_token_estimate(service, model, direction, count):
    """Record a clearly-labelled ESTIMATE of token usage (not exact)."""
    if count:
        rag_token_estimates_total.labels(
            service=service, model=model, direction=direction
        ).inc(count)


def record_drift(service, max_similarity=None, drifted=False):
    """Record a drift evaluation for a query."""
    if drifted:
        rag_embedding_drift_total.labels(service=service).inc()
    if max_similarity is not None:
        rag_embedding_similarity.labels(service=service).set(max_similarity)


def record_drift_error(service, error_type):
    """Record that the drift DETECTOR failed (no score was produced).

    This is deliberately separate from :func:`record_drift`: a detector
    error must never be counted as a drift event, because it says nothing
    about the query. It is the only signal that drift monitoring itself
    is broken (missing baseline, embedding failure, unreadable report).
    """
    rag_embedding_drift_errors_total.labels(
        service=service, error_type=str(error_type)
    ).inc()


def record_guardrail(service, stage, action, kind):
    """Record a guardrail action (stage=input/output, action=allow/mask/block)."""
    rag_guardrail_actions_total.labels(
        service=service, stage=stage, action=action, kind=kind
    ).inc()
    if action == "block":
        rag_guardrail_blocked_total.labels(service=service, stage=stage).inc()


def record_ragas(service, status, score=None):
    """Record a faithfulness evaluation outcome and, on success, its score."""
    rag_ragas_evaluations_total.labels(service=service, status=status).inc()
    if status == "success" and score is not None:
        rag_ragas_faithfulness.labels(service=service).set(score)
