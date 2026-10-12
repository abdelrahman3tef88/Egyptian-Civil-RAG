"""Offline unit tests for guardrails, drift, metrics, langfuse, smoke runner.

These tests are deliberately light: they do NOT import bentoml, chromadb, or
any heavy embedding/LLM model. Everything runs on the CPU with synthetic
data or mocks, so the whole suite is safe to run without a GPU or the remote
vLLM server.
"""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from guardrails import pii_detector as pii  # noqa: E402


# ==========================================================
# PII guardrail tests: allow / mask / block / false-positive
# ==========================================================
class TestPIIDetection:
    def test_allow_when_no_pii(self):
        scan = pii.detect("ما هي أركان عقد البيع في القانون المدني؟")
        assert scan.has_pii is False
        assert pii.decide(scan) == pii.ALLOW

    def test_detect_egyptian_mobile(self):
        scan = pii.detect("رقمي هو 01012345678 للتواصل")
        assert scan.kinds == [pii.EGYPTIAN_MOBILE]

    def test_detect_mobile_with_country_code(self):
        scan = pii.detect("call +201234567890 now")
        assert scan.kinds == [pii.EGYPTIAN_MOBILE]

    def test_detect_egyptian_national_id_blocks(self):
        scan = pii.detect("الرقم القومي 28005211500123 هنا")
        assert pii.EGYPTIAN_NATIONAL_ID in scan.kinds
        assert pii.decide(scan) == pii.BLOCK

    def test_detect_email(self):
        scan = pii.detect("reach me at legal@example.com please")
        assert scan.kinds == [pii.EMAIL]

    def test_mask_mobile(self):
        masked = pii.mask("اتصل على 01012345678")
        assert "01012345678" not in masked
        assert "[REDACTED_PHONE]" in masked

    def test_mask_email(self):
        masked = pii.mask("mail: legal@example.com")
        assert "legal@example.com" not in masked
        assert "[REDACTED_EMAIL]" in masked

    def test_mask_national_id(self):
        masked = pii.mask("id 28005211500123")
        assert "28005211500123" not in masked
        assert "[REDACTED_NATIONAL_ID]" in masked

    def test_false_positive_short_digits(self):
        # A short digit run must NOT be treated as a national id or mobile.
        scan = pii.detect("المادة 123 من القانون")
        assert scan.has_pii is False

    def test_false_positive_article_number(self):
        scan = pii.detect("ما نص المادة 456؟")
        assert scan.has_pii is False

    def test_policy_override_block_mobile(self):
        scan = pii.detect("رقمي 01012345678")
        assert pii.decide(scan, {"egyptian_mobile": "block"}) == pii.BLOCK

    def test_policy_override_allow_email(self):
        scan = pii.detect("legal@example.com")
        assert pii.decide(scan, {"email": "allow"}) == pii.ALLOW

    def test_no_raw_value_in_result(self):
        # The scan result must never carry the raw matched text.
        scan = pii.detect("28005211500123")
        serialized = repr(scan)
        assert "28005211500123" not in serialized


# ==========================================================
# Drift detection tests (synthetic vectors, no model load)
# ==========================================================
class TestDrift:
    def _load(self):
        from monitoring import evaluate_embedding_drift as drift

        return drift

    def test_cosine_identical_is_one(self):
        drift = self._load()
        assert drift.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_cosine_orthogonal_is_zero(self):
        drift = self._load()
        assert drift.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_max_reference_similarity_picks_best(self):
        drift = self._load()
        refs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        best, index = drift.max_reference_similarity([0.9, 0.1, 0], refs)
        assert index == 0
        assert best > 0.9

    def test_evaluate_query_flags_drift_below_threshold(self):
        drift = self._load()
        baseline = {
            "model": "BAAI/bge-m3",
            "reference_ids": ["ref_000", "ref_001"],
            "embeddings": [[1, 0, 0], [0, 1, 0]],
        }
        result = drift.evaluate_query([0, 0, 1], baseline, threshold=0.5)
        assert result["drift"] is True
        assert result["max_similarity"] == pytest.approx(0.0)

    def test_evaluate_query_no_drift_above_threshold(self):
        drift = self._load()
        baseline = {
            "model": "BAAI/bge-m3",
            "reference_ids": ["ref_000"],
            "embeddings": [[1, 0, 0]],
        }
        result = drift.evaluate_query([1, 0, 0], baseline, threshold=0.5)
        assert result["drift"] is False
        assert result["closest_reference_id"] == "ref_000"

    def test_report_has_no_raw_embedding(self):
        drift = self._load()
        baseline = {"model": "m", "reference_ids": ["a"], "embeddings": [[1, 2, 3]]}
        result = drift.evaluate_query([1, 2, 3], baseline, threshold=0.1)
        # The per-request report must not embed the raw vector.
        assert "embedding" not in result
        assert [1, 2, 3] != result

    def test_calibrate_threshold_midpoint(self):
        drift = self._load()
        result = drift.calibrate_threshold([0.8, 0.9], [0.2, 0.3])
        assert result["suggested_threshold"] == pytest.approx(0.55)
        assert "PROVISIONAL" in result["note"]

    def test_threshold_comes_from_config_not_a_dead_constant(self):
        """configs/config.yaml is the single source of truth for the threshold.

        A module-level ``_PROVISIONAL_THRESHOLD`` used to sit next to a
        different value, so the file contradicted the live configuration.
        """
        drift = self._load()
        assert not hasattr(drift, "_PROVISIONAL_THRESHOLD"), (
            "a hard-coded drift threshold must not shadow configs/config.yaml"
        )
        from rag_project.config.settings import get_drift_config

        configured = get_drift_config()["threshold"]
        # evaluate_query must use exactly the configured threshold.
        baseline = {
            "model": "m",
            "reference_ids": ["a"],
            "embeddings": [[1, 0, 0]],
        }
        result = drift.evaluate_query([1, 0, 0], baseline)
        assert result["threshold"] == pytest.approx(configured)

    def test_detector_failure_is_not_counted_as_drift(self):
        """A broken detector must be observable separately from drift itself.

        A drift event means 'scored below the threshold'. A detector error
        means 'not scored at all' - conflating them would either hide a
        broken monitor or manufacture false alarms.
        """
        from monitoring import metrics as m

        drifted = m.rag_embedding_drift_total.labels(service="svc-det")
        errored = m.rag_embedding_drift_errors_total.labels(
            service="svc-det", error_type="ValueError"
        )
        drift_before, error_before = drifted._value.get(), errored._value.get()

        m.record_drift("svc-det", max_similarity=0.2, drifted=True)
        m.record_drift_error("svc-det", "ValueError")

        assert drifted._value.get() == drift_before + 1
        assert errored._value.get() == error_before + 1

        # The two counters are independent: recording an error never
        # increments the drift counter.
        m.record_drift_error("svc-det", "RuntimeError")
        assert drifted._value.get() == drift_before + 1

    def test_drift_error_report_records_status_error(self, tmp_path):
        """A detector failure is written to the drift report, not swallowed."""
        drift = self._load()
        report_path = drift.append_drift_report(
            {
                "trace_id": "trace-det",
                "error_type": "ValueError",
                "status": "error",
            },
            path=tmp_path / "reports",
        )
        entries = [
            json.loads(line)
            for line in report_path.read_text(encoding="utf-8").splitlines()
        ]
        assert entries[0]["status"] == "error"
        assert entries[0]["error_type"] == "ValueError"
        # A failure must never be reported as a scored drift event.
        assert "drift" not in entries[0]
        assert "max_similarity" not in entries[0]


# ==========================================================
# Prometheus metrics tests
# ==========================================================
class TestMetrics:
    def test_counters_increment(self):
        from monitoring import metrics

        c = metrics.rag_requests_total.labels(
            service="fastapi", endpoint="/ask", status="200"
        )
        before = c._value.get()
        c.inc()
        assert c._value.get() == before + 1

    def test_histogram_observes_duration(self):
        from monitoring import metrics

        hist = metrics.rag_request_duration_seconds.labels(
            service="fastapi", endpoint="/ask"
        )
        hist.observe(0.5)
        assert hist._sum.get() > 0

    def test_token_estimate_is_labelled(self):
        from monitoring import metrics

        assert hasattr(metrics, "record_token_estimate")
        # Signature: (service, model, direction, count). direction marks the
        # estimate explicitly so it is never mistaken for an exact count.
        metrics.record_token_estimate(
            service="bentoml", model="qwen", direction="output", count=20
        )


# ==========================================================
# Cost metric: opt-in explicit pricing only, never invented
# ==========================================================
class TestCostMetric:
    def test_cost_requires_explicit_pricing(self, monkeypatch):
        from monitoring import metrics as mm

        monkeypatch.setenv("MODEL_PRICING_INPUT_PER_1K_USD", "0.01")
        monkeypatch.setenv("MODEL_PRICING_OUTPUT_PER_1K_USD", "0.02")
        before = mm.rag_estimated_cost_usd_total.labels(
            service="svc-cost-a", model="m"
        )._value.get()
        got = mm.record_cost_from_usage("svc-cost-a", "m", 1000, 1000)
        assert abs(got - 0.03) < 1e-9
        after = mm.rag_estimated_cost_usd_total.labels(
            service="svc-cost-a", model="m"
        )._value.get()
        assert abs(after - before - 0.03) < 1e-9

    def test_cost_records_nothing_without_pricing(self, monkeypatch):
        from monitoring import metrics as mm

        monkeypatch.delenv("MODEL_PRICING_INPUT_PER_1K_USD", raising=False)
        monkeypatch.delenv("MODEL_PRICING_OUTPUT_PER_1K_USD", raising=False)
        before = mm.rag_estimated_cost_usd_total.labels(
            service="svc-cost-b", model="m"
        )._value.get()
        assert mm.record_cost_from_usage("svc-cost-b", "m", 1000, 1000) == 0.0
        after = mm.rag_estimated_cost_usd_total.labels(
            service="svc-cost-b", model="m"
        )._value.get()
        assert after == before

    def test_token_estimate_records_output_estimate(self):
        from monitoring import metrics

        metrics.record_token_estimate(
            service="bentoml", model="qwen", direction="output", count=20
        )

    def test_drift_and_guardrail_counters_exist(self):
        from monitoring import metrics

        metrics.rag_embedding_drift_total.labels(service="fastapi").inc()
        metrics.rag_guardrail_actions_total.labels(
            service="fastapi", stage="input", action="mask", kind="egyptian_mobile"
        ).inc()


# ==========================================================
# Langfuse trace/score association tests (mocked client)
# ==========================================================
class TestLangfuse:
    def test_new_trace_id_returns_string(self):
        from observability import langfuse_tracing as lf

        assert isinstance(lf.new_trace_id(), str)

    def test_disabled_by_default_without_keys(self):
        from observability import langfuse_tracing as lf

        with mock.patch.dict(
            os.environ, {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}
        ):
            lf._client = None
            lf._client_ready = False
            assert lf.is_enabled() is False
            assert lf.get_client() is None

    def test_span_is_noop_when_disabled(self):
        from observability import langfuse_tracing as lf

        with mock.patch.object(lf, "get_client", return_value=None):
            ran = False
            with lf.span("retrieval", as_type="retriever"):
                ran = True
            assert ran is True

    def test_score_associates_with_correct_trace_id(self):
        from observability import langfuse_tracing as lf

        fake_client = mock.Mock()
        with mock.patch.object(lf, "get_client", return_value=fake_client):
            ok = lf.score_trace("trace-abc", "faithfulness", 0.87, comment="ok")
        assert ok is True
        _, kwargs = fake_client.create_score.call_args
        assert kwargs["trace_id"] == "trace-abc"
        assert kwargs["value"] == 0.87
        assert kwargs["name"] == "faithfulness"

    def test_score_failure_returns_false(self):
        from observability import langfuse_tracing as lf

        fake_client = mock.Mock()
        fake_client.create_score.side_effect = RuntimeError("langfuse down")
        with mock.patch.object(lf, "get_client", return_value=fake_client):
            ok = lf.score_trace("trace-abc", "faithfulness", 0.5)
        assert ok is False


# ==========================================================
# Smoke-test runner tests (mocked HTTP endpoint)
# ==========================================================
class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def _load_smoke_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_smoke", ROOT / "scripts" / "run_smoke_test.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSmokeRunner:
    def _questions(self, tmp_path):
        data = {
            "questions": [{"id": f"ci_{i:03d}", "question": f"q{i}"} for i in range(10)]
        }
        path = tmp_path / "smoke.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_validate_response_ok(self):
        mod = _load_smoke_module()
        assert mod.validate_response({"answer": "a", "sources": ["1"]}) == []

    def test_validate_response_missing_sources(self):
        mod = _load_smoke_module()
        problems = mod.validate_response({"answer": "a"})
        assert any("sources" in p for p in problems)

    def test_run_all_pass(self, tmp_path):
        mod = _load_smoke_module()
        qpath = self._questions(tmp_path)
        report = tmp_path / "report.json"

        def fake_post(url, payload, timeout):
            return _FakeResponse(200, {"answer": "جواب", "sources": ["المادة 1"]})

        summary = mod.run("http://x", qpath, report, 5.0, post=fake_post)
        assert summary["overall"] == "PASS"
        assert summary["passed"] == 10
        assert report.exists()

    def test_run_records_http_error(self, tmp_path):
        mod = _load_smoke_module()
        qpath = self._questions(tmp_path)
        report = tmp_path / "report.json"

        def fake_post(url, payload, timeout):
            return _FakeResponse(500, {"detail": "boom"})

        summary = mod.run("http://x", qpath, report, 5.0, post=fake_post)
        assert summary["overall"] == "FAIL"
        assert summary["results"][0]["status"] == "fail"

    def test_run_records_connection_error_type_only(self, tmp_path):
        mod = _load_smoke_module()
        qpath = self._questions(tmp_path)
        report = tmp_path / "report.json"
        import requests

        def fake_post(url, payload, timeout):
            raise requests.exceptions.ConnectionError("http://secret:9999")

        summary = mod.run("http://x", qpath, report, 5.0, post=fake_post)
        assert summary["overall"] == "FAIL"
        # Only the exception type is stored, never the URL/secret.
        assert summary["results"][0]["status"] == "error"
        assert "secret" not in summary["results"][0]["error"]
