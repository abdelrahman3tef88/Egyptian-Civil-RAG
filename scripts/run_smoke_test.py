"""Smoke-test runner: POST each smoke question to the FastAPI /ask endpoint.

1. Loads the 10 deterministically-selected questions from
   data/smoke_test_questions.json (produced by prepare_smoke_test.py).
2. Sends ONE HTTP POST per question to ``{base_url}/ask`` with body
   ``{"question": "..."}`` (never the whole list as one batch request).
3. Waits for each response and validates the HTTP status and the response
   schema: the body MUST contain ``answer`` (str) and ``sources`` (list).
4. Records the answer, sources, duration, status, and safe error details.
5. Saves a JSON report under reports/ and prints a pass/fail summary.

Config (env vars): SMOKE_BASE_URL (default http://127.0.0.1:8000),
SMOKE_TIMEOUT (default 120), SMOKE_QUESTIONS, SMOKE_REPORT.

No raw PII is printed or saved: answers are truncated and the runner never
echoes secrets. The HTTP call goes through ``_post`` so tests can mock it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "smoke_test_questions.json"
DEFAULT_REPORT = ROOT / "reports" / "smoke_test_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post(url: str, payload: dict, timeout: float) -> requests.Response:
    """Single POST of one question. Isolated so tests can mock it."""
    return requests.post(url, json=payload, timeout=timeout)


def _truncate(text: str, limit: int = 80) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def validate_response(body) -> list:
    """Return a list of schema problems for a /ask response body."""
    if not isinstance(body, dict):
        return ["response body is not a JSON object"]
    problems = []
    answer = body.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        problems.append("missing or empty 'answer' string")
    sources = body.get("sources")
    if not isinstance(sources, list):
        problems.append("missing 'sources' list")
    elif sources and not all(isinstance(s, str) for s in sources):
        problems.append("'sources' must be a list of strings")
    return problems


def run(
    base_url: str,
    questions_path: Path = DEFAULT_QUESTIONS,
    report_path: Path = DEFAULT_REPORT,
    timeout: float = 120.0,
    post=_post,
) -> dict:
    """Run the smoke test and return a summary dict (also writes the report)."""
    if not Path(questions_path).exists():
        raise FileNotFoundError(
            f"Smoke questions file not found: {questions_path}. "
            f"Run scripts/prepare_smoke_test.py first."
        )

    dataset = json.load(open(questions_path, encoding="utf-8"))
    questions = dataset["questions"] if isinstance(dataset, dict) else dataset
    endpoint = base_url.rstrip("/") + "/ask"

    results = []
    passed = 0
    for item in questions:
        question = item["question"]
        entry = {
            "id": item.get("id"),
            "question": _truncate(question),
            "timestamp": _utc_now(),
        }
        start = time.perf_counter()
        try:
            response = post(endpoint, {"question": question}, timeout)
            duration = time.perf_counter() - start
            entry["duration_seconds"] = round(duration, 3)
            entry["status_code"] = response.status_code

            if response.status_code != 200:
                entry["status"] = "fail"
                entry["error"] = f"HTTP {response.status_code}"
            else:
                body = response.json()
                problems = validate_response(body)
                if problems:
                    entry["status"] = "fail"
                    entry["error"] = "; ".join(problems)
                else:
                    entry["status"] = "pass"
                    entry["answer"] = _truncate(body["answer"], 300)
                    entry["sources"] = body["sources"]
                    passed += 1
        except requests.exceptions.RequestException as error:
            entry["duration_seconds"] = round(time.perf_counter() - start, 3)
            entry["status"] = "error"
            # Only the exception TYPE is recorded, never a stack/secret.
            entry["error"] = type(error).__name__
        results.append(entry)

    total = len(questions)
    summary = {
        "timestamp": _utc_now(),
        "base_url": base_url,
        "endpoint": endpoint,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "overall": "PASS" if passed == total else "FAIL",
        "results": results,
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def main():
    base_url = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8000")
    timeout = float(os.getenv("SMOKE_TIMEOUT", "120"))
    questions_path = Path(os.getenv("SMOKE_QUESTIONS", str(DEFAULT_QUESTIONS)))
    report_path = Path(os.getenv("SMOKE_REPORT", str(DEFAULT_REPORT)))

    try:
        summary = run(base_url, questions_path, report_path, timeout)
    except FileNotFoundError as error:
        print(f"STOP: {error}")
        sys.exit(1)

    print(
        f"\nSmoke test summary: {summary['overall']} "
        f"({summary['passed']}/{summary['total']} passed)"
    )
    for entry in summary["results"]:
        print(
            f"  [{entry['status']:>4}] {entry['id']} "
            f"({entry.get('duration_seconds', '?')}s) {entry.get('error', '')}"
        )
    print(f"Report: {report_path}")
    sys.exit(0 if summary["overall"] == "PASS" else 2)


if __name__ == "__main__":
    main()
