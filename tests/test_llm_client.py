"""Tests for the remote vLLM client (service/llm_client.py).

These tests pin the contract of the ONLY component that talks to the
remote GPU server. No network, no GPU, no model: ``AsyncOpenAI`` is
replaced with a fake, so the request payload can be inspected directly.

What is verified here:
  * endpoint, model and credentials come from environment configuration,
  * configs/config.yaml supplies the declared defaults when the env is
    empty (it used to be ignored entirely, leaving timeout_seconds and
    max_tokens dead configuration),
  * the outgoing request carries the right model id, messages and
    max_tokens budget,
  * the client is created with the configured timeout,
  * streaming yields ``delta.content`` and skips keep-alive chunks,
  * connection/HTTP/parse failures PROPAGATE (they are never turned into
    an answer), and
  * secrets are never read from a committed file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service import llm_client as llc  # noqa: E402


# ==========================================================
# Doubles
# ==========================================================
class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.owner.response


class _FakeChat:
    def __init__(self, owner):
        self.completions = _FakeCompletions(owner)


class _FakeAsyncOpenAI:
    """Records its constructor kwargs and serves a canned response."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = _FakeChat(self)
        self.response = None
        _FakeAsyncOpenAI.instances.append(self)


class _FakePromptValue:
    """Minimal LangChain PromptValue stand-in."""

    def __init__(self, messages):
        self._messages = messages

    def to_messages(self):
        return self._messages

    def to_string(self):
        return "\n".join(m.content for m in self._messages)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate every test from a developer's real .env / VLLM_* variables."""
    for name in ("VLLM_BASE_URL", "VLLM_MODEL", "VLLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    _FakeAsyncOpenAI.instances = []
    yield


def _patch_client(monkeypatch, response=None):
    owner = SimpleNamespace()

    def factory(**kwargs):
        client = _FakeAsyncOpenAI(**kwargs)
        client.response = response
        return client

    monkeypatch.setattr(
        llc,
        "create_async_client",
        lambda: _FakeAsyncOpenAI(**{}) if False else factory(),
    )
    return owner


# ==========================================================
# Configuration resolution
# ==========================================================
class TestVllmConfigResolution:
    def test_base_url_and_model_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-host:8000/v1")
        monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        monkeypatch.setenv("VLLM_API_KEY", "sk-secret")

        config = llc.get_vllm_config()
        assert config["base_url"] == "http://gpu-host:8000/v1"
        assert config["model"] == "Qwen/Qwen3-8B-AWQ"
        assert config["api_key"] == "sk-secret"

    def test_config_yaml_supplies_the_declared_defaults(self, monkeypatch):
        """configs/config.yaml is the declared default, not dead weight."""
        config = llc.get_vllm_config()
        # vllm section of configs/config.yaml
        assert config["base_url"] == "http://127.0.0.1:8000/v1"
        assert config["model"] == "Qwen/Qwen3-8B-AWQ"
        assert config["timeout_seconds"] == pytest.approx(300.0)
        assert config["max_tokens"] == 4096

    def test_environment_overrides_config_yaml(self, monkeypatch):
        monkeypatch.setenv("VLLM_MODEL", "some/other-model")
        assert llc.get_vllm_config()["model"] == "some/other-model"

    def test_no_model_anywhere_is_not_configured(self, monkeypatch):
        monkeypatch.setattr(llc, "_config_defaults", lambda: {})
        assert llc.is_configured() is False

    def test_api_key_is_never_read_from_the_committed_config(self, monkeypatch):
        """The config file is in git; secrets must not be picked up there."""
        monkeypatch.setattr(
            llc, "_config_defaults", lambda: {"api_key": "leaked-from-yaml"}
        )
        assert llc.get_vllm_config()["api_key"] == ""
        monkeypatch.setenv("VLLM_API_KEY", "sk-from-env")
        assert llc.get_vllm_config()["api_key"] == "sk-from-env"

    def test_broken_config_does_not_break_the_client(self, monkeypatch):
        """A unreadable config file degrades to safe local defaults."""
        monkeypatch.setattr(
            "rag_project.config.settings.get_vllm_config",
            lambda: (_ for _ in ()).throw(OSError("config unreadable")),
        )
        config = llc.get_vllm_config()
        assert config["base_url"] == "http://127.0.0.1:8000/v1"
        assert config["timeout_seconds"] == pytest.approx(300.0)


# ==========================================================
# The outgoing request
# ==========================================================
class TestRequestPayload:
    def _fake_response(self, content="إجابة"):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    def test_generate_sends_the_configured_model_messages_and_budget(self, monkeypatch):
        monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        client = _FakeAsyncOpenAI()
        client.response = self._fake_response()
        monkeypatch.setattr(llc, "create_async_client", lambda: client)

        prompt = _FakePromptValue(
            [
                SimpleNamespace(type="system", content="SYSTEM"),
                SimpleNamespace(type="human", content="USER"),
            ]
        )
        answer = llc._run_async(llc.generate(llc.prompt_value_to_messages(prompt)))

        assert answer == "إجابة"
        call = client.chat.completions.calls[0]
        assert call["model"] == "Qwen/Qwen3-8B-AWQ"
        assert call["stream"] is False
        assert call["max_tokens"] == 4096
        assert call["messages"] == [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "USER"},
        ]
        # The existing legal prompt is forwarded verbatim - never rebuilt.
        assert "SYSTEM" in call["messages"][0]["content"]

    def test_generate_captures_exact_token_usage(self, monkeypatch):
        monkeypatch.setenv("VLLM_MODEL", "m")
        client = _FakeAsyncOpenAI()
        client.response = self._fake_response()
        monkeypatch.setattr(llc, "create_async_client", lambda: client)

        usage = {}
        llc._run_async(llc.generate([{"role": "user", "content": "q"}], usage))
        assert usage == {"input_tokens": 11, "output_tokens": 7}

    def test_missing_usage_object_leaves_the_dict_empty(self, monkeypatch):
        monkeypatch.setenv("VLLM_MODEL", "m")
        client = _FakeAsyncOpenAI()
        client.response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="a"))],
            usage=None,
        )
        monkeypatch.setattr(llc, "create_async_client", lambda: client)

        usage = {}
        llc._run_async(llc.generate([{"role": "user", "content": "q"}], usage))
        assert usage == {}, "a missing usage object must not be guessed at"

    def test_thinking_mode_is_disabled_for_low_ttft(self, monkeypatch):
        monkeypatch.setenv("VLLM_MODEL", "m")
        client = _FakeAsyncOpenAI()
        client.response = self._fake_response()
        monkeypatch.setattr(llc, "create_async_client", lambda: client)

        llc._run_async(llc.generate([{"role": "user", "content": "q"}]))
        extra = client.chat.completions.calls[0]["extra_body"]
        assert extra == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_streaming_uses_the_same_payload_with_stream_true(self, monkeypatch):
        monkeypatch.setenv("VLLM_MODEL", "m")

        class _Stream:
            def __aiter__(self):
                async def gen():
                    # A keep-alive chunk, a role-only chunk, then real text.
                    yield SimpleNamespace(choices=[])
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
                    )
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="المادة"))
                        ]
                    )
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=" 20"))]
                    )

                return gen()

        client = _FakeAsyncOpenAI()
        client.response = _Stream()
        monkeypatch.setattr(llc, "create_async_client", lambda: client)

        async def collect():
            return [
                chunk
                async for chunk in llc.generate_stream(
                    [{"role": "user", "content": "q"}]
                )
            ]

        chunks = llc._run_async(collect())
        assert chunks == ["المادة", " 20"]
        call = client.chat.completions.calls[0]
        assert call["stream"] is True
        assert call["max_tokens"] == 4096


# ==========================================================
# Client construction
# ==========================================================
class TestClientConstruction:
    def test_client_uses_the_configured_timeout(self, monkeypatch):
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return _FakeAsyncOpenAI()

        monkeypatch.setattr("openai.AsyncOpenAI", fake_openai)
        monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-host:8000/v1")

        llc.create_async_client()
        assert captured["base_url"] == "http://gpu-host:8000/v1"
        assert captured["timeout"] == pytest.approx(300.0)

    def test_empty_api_key_is_replaced_by_a_placeholder(self, monkeypatch):
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return _FakeAsyncOpenAI()

        monkeypatch.setattr("openai.AsyncOpenAI", fake_openai)
        llc.create_async_client()
        assert captured["api_key"] == "not-needed"

    def test_real_api_key_is_forwarded(self, monkeypatch):
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return _FakeAsyncOpenAI()

        monkeypatch.setattr("openai.AsyncOpenAI", fake_openai)
        monkeypatch.setenv("VLLM_API_KEY", "sk-live")
        llc.create_async_client()
        assert captured["api_key"] == "sk-live"


# ==========================================================
# Failure handling: propagate, never fabricate an answer
# ==========================================================
class TestFailurePropagation:
    def test_unavailable_endpoint_raises(self, monkeypatch):
        """An unreachable vLLM must surface, not become an answer."""

        def boom(**_kwargs):
            raise ConnectionError("vllm unreachable")

        monkeypatch.setattr(llc, "create_async_client", boom)
        monkeypatch.setenv("VLLM_MODEL", "m")

        with pytest.raises(ConnectionError):
            llc._run_async(llc.generate([{"role": "user", "content": "q"}]))

    def test_http_error_propagates(self, monkeypatch):
        """An upstream HTTP error is NOT swallowed or turned into an answer."""

        class _HttpError(Exception):
            pass

        def boom(**_kwargs):
            raise _HttpError("vLLM returned HTTP 500")

        monkeypatch.setattr(llc, "create_async_client", boom)
        monkeypatch.setenv("VLLM_MODEL", "m")

        with pytest.raises(_HttpError):
            llc._run_async(llc.generate([{"role": "user", "content": "q"}]))

    def test_invalid_response_propagates(self, monkeypatch):
        client = _FakeAsyncOpenAI()
        client.response = SimpleNamespace(choices=[])  # no usable choice
        monkeypatch.setattr(llc, "create_async_client", lambda: client)
        monkeypatch.setenv("VLLM_MODEL", "m")

        with pytest.raises(IndexError):
            llc._run_async(llc.generate([{"role": "user", "content": "q"}]))

    def test_unconfigured_model_fails_before_any_network_call(self, monkeypatch):
        monkeypatch.setattr(llc, "_config_defaults", lambda: {})
        monkeypatch.setattr(
            llc,
            "create_async_client",
            lambda: pytest.fail("no request may be made without a model"),
        )
        with pytest.raises(RuntimeError, match="VLLM_MODEL"):
            llc._run_async(llc.generate([{"role": "user", "content": "q"}]))

    def test_empty_answer_is_returned_as_empty_string(self, monkeypatch):
        """A legal empty answer is NOT an error."""
        monkeypatch.setenv("VLLM_MODEL", "m")
        client = _FakeAsyncOpenAI()
        client.response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        )
        monkeypatch.setattr(llc, "create_async_client", lambda: client)
        assert llc._run_async(llc.generate([{"role": "user", "content": "q"}])) == ""


# ==========================================================
# PromptValue -> OpenAI messages
# ==========================================================
class TestPromptConversion:
    def test_langchain_roles_map_to_openai_roles(self):
        prompt = _FakePromptValue(
            [
                SimpleNamespace(type="system", content="S"),
                SimpleNamespace(type="human", content="H"),
                SimpleNamespace(type="ai", content="A"),
            ]
        )
        assert llc.prompt_value_to_messages(prompt) == [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "H"},
            {"role": "assistant", "content": "A"},
        ]

    def test_unknown_role_defaults_to_user(self):
        prompt = _FakePromptValue([SimpleNamespace(type="weird", content="X")])
        assert llc.prompt_value_to_messages(prompt) == [
            {"role": "user", "content": "X"}
        ]

    def test_stream_from_prompt_uses_the_same_conversion(self, monkeypatch):
        captured = {}

        class _Stream:
            def __aiter__(self):
                async def gen():
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
                    )

                return gen()

        client = _FakeAsyncOpenAI()
        client.response = _Stream()
        monkeypatch.setattr(llc, "create_async_client", lambda: client)
        monkeypatch.setenv("VLLM_MODEL", "m")

        prompt = _FakePromptValue([SimpleNamespace(type="human", content="H")])

        async def collect():
            return [c async for c in llc.stream_from_prompt(prompt)]

        assert llc._run_async(collect()) == ["ok"]
        call = client.chat.completions.calls[0]
        captured.update(call)
        assert call["messages"] == [{"role": "user", "content": "H"}]
