FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY artifacts/ ./artifacts/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

EXPOSE 8000

CMD ["uvicorn", "rag_project.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
