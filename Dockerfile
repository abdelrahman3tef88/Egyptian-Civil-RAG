FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

COPY configs/ ./configs/
COPY src/ ./src/
COPY service/ ./service/
# Root-level packages imported at STARTUP by service/bentoml_service.py:
#   from guardrails   import pii_detector
#   from monitoring   import metrics, evaluate_embedding_drift
#   from observability import langfuse_tracing
# Without these the BentoML service fails at import time with
# ModuleNotFoundError before it can serve a single request.
COPY guardrails/ ./guardrails/
COPY monitoring/ ./monitoring/
COPY observability/ ./observability/
COPY data/processed/ ./data/processed/
COPY artifacts/ ./artifacts/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# BentoML service port
EXPOSE 3000

# Default: run BentoML service. Override with uvicorn for FastAPI gateway.
CMD ["bentoml", "serve", "service.bentoml_service:RAGService", "--host", "0.0.0.0", "--port", "3000"]
