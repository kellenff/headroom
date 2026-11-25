"""Startup checks for the operator's model-backend selections (proxy/server.py).

``_preflight_model_backends`` turns a selection that cannot work (an unknown
name, a malformed variable, a backend that fails to construct, or one that
would dial out under ``HEADROOM_OFFLINE``) into exit 78 with a message that
names the variable. Without it, each of these surfaced only as a quiet
degradation after startup: memory off, relevance on BM25, Kompress off.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from headroom.config import RelevanceScorerConfig
from headroom.models.backends import EMBEDDING_BACKENDS
from headroom.offline import guard_egress
from headroom.proxy import server
from headroom.proxy.models import ProxyConfig
from headroom.transforms import kompress_remote
from headroom.transforms.content_router import ContentRouterConfig
from headroom.transforms.kompress_backend import KOMPRESS_BACKENDS

SECRET = "sk-live-0123456789abcdef"

_ENV_FAMILIES = ("HEADROOM_EMBEDDER", "HEADROOM_RELEVANCE_EMBEDDER")
_SUFFIXES = ("PROVIDER", "MODEL", "BASE_URL", "API_KEY", "DIMENSION", "HEADERS", "OPTIONS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for prefix in _ENV_FAMILIES:
        for suffix in _SUFFIXES:
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for var in (
        "HEADROOM_EMBEDDER_RUNTIME",
        "HEADROOM_KOMPRESS_PROVIDER",
        "HEADROOM_KOMPRESS_PROVIDER_OPTIONS",
        "HEADROOM_KOMPRESS_ENDPOINT",
        "HEADROOM_OFFLINE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    EMBEDDING_BACKENDS._reset_for_tests()
    KOMPRESS_BACKENDS._reset_for_tests()
    kompress_remote._reset_provider_selection_for_tests()
    yield
    EMBEDDING_BACKENDS._reset_for_tests()
    KOMPRESS_BACKENDS._reset_for_tests()
    kompress_remote._reset_provider_selection_for_tests()


def _config(**overrides: Any) -> ProxyConfig:
    values: dict[str, Any] = {"optimize": True, "memory_enabled": True}
    values.update(overrides)
    return ProxyConfig(**values)


def _refused(config: ProxyConfig, capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        server._preflight_model_backends(config)
    assert excinfo.value.code == 78
    return capsys.readouterr().err


class _Recorder:
    """An embedding backend that records construction and close."""

    instances: list[_Recorder] = []

    def __init__(self, model: str | None = None, dimension: int | None = None, **options: Any):
        self.name = "recorder"
        self.model_name = model or "recorder-model"
        self.dimension = dimension or 4
        self.options = options
        self.closed = False
        _Recorder.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dimension for _ in texts]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recorder() -> type[_Recorder]:
    _Recorder.instances = []
    EMBEDDING_BACKENDS.register("recorder", _Recorder)
    return _Recorder


# ----------------------------------------------------------------- defaults


def test_nothing_selected_does_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy")
    server._preflight_model_backends(_config())
    assert not [r for r in caplog.records if "model_backend" in r.getMessage()]


def test_default_router_config_is_unchanged() -> None:
    router_config = ContentRouterConfig()
    server._apply_relevance_embedder_env(router_config)
    assert router_config.relevance == RelevanceScorerConfig()
    assert router_config.relevance.embedding_backend is None


def test_a_relevance_provider_reaches_the_router_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "openai")
    router_config = ContentRouterConfig()
    server._apply_relevance_embedder_env(router_config)
    assert router_config.relevance.embedding_backend == "env"
    # Everything else about relevance scoring stays as it was.
    assert router_config.relevance.tier == RelevanceScorerConfig().tier


def test_the_proxy_routers_carry_the_relevance_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from headroom.proxy.server import create_app
    from headroom.transforms.content_router import ContentRouter

    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "openai")
    app = create_app(ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False))
    proxy = app.state.proxy
    routers = [
        transform
        for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline)
        for transform in pipeline.transforms
        if isinstance(transform, ContentRouter)
    ]
    assert routers
    assert {router.config.relevance.embedding_backend for router in routers} == {"env"}


# ----------------------------------------------------------------- memory


def test_memory_unknown_provider_names_the_variable_not_the_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", SECRET)
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "8")
    message = _refused(_config(), capsys)
    assert "HEADROOM_EMBEDDER_PROVIDER" in message
    assert SECRET not in message


def test_memory_missing_dimension_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", SECRET)
    message = _refused(_config(), capsys)
    assert "HEADROOM_EMBEDDER_DIMENSION" in message
    assert SECRET not in message


def test_memory_backend_that_fails_to_construct_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A self-hosted server needs an explicit model; the constructor says so.
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("HEADROOM_EMBEDDER_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", SECRET)
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "8")
    message = _refused(_config(), capsys)
    assert "model is required" in message
    assert SECRET not in message


def test_memory_runtime_alias_is_checked_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_RUNTIME", "openai")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "8")
    # api.openai.com without a key.
    message = _refused(_config(), capsys)
    assert "api_key is required" in message


def test_memory_selection_is_ignored_when_memory_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "no-such-backend")
    server._preflight_model_backends(_config(memory_enabled=False))


def test_memory_selection_is_ignored_for_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "no-such-backend")
    server._preflight_model_backends(_config(memory_backend="qdrant-neo4j"))


def test_memory_valid_selection_is_constructed_once_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    recorder: type[_Recorder],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy")
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "recorder")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "4")
    monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", SECRET)
    server._preflight_model_backends(_config())
    assert len(recorder.instances) == 1
    assert recorder.instances[0].closed
    assert recorder.instances[0].dimension == 4
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "kind=memory_embedder backend=recorder" in logged
    assert SECRET not in logged


def test_memory_third_party_error_text_is_not_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(**_: Any) -> Any:
        raise RuntimeError(f"could not log in with {SECRET}")

    EMBEDDING_BACKENDS.register("broken", broken)
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "broken")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "4")
    message = _refused(_config(), capsys)
    assert "RuntimeError" in message
    assert SECRET not in message


def test_memory_remote_embedder_is_refused_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("HEADROOM_EMBEDDER_BASE_URL", "https://embed.example.com/v1")
    monkeypatch.setenv("HEADROOM_EMBEDDER_MODEL", "m")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "8")
    message = _refused(_config(), capsys)
    assert "HEADROOM_OFFLINE" in message
    assert "HEADROOM_EMBEDDER" in message


# ----------------------------------------------------------------- relevance


def test_relevance_selection_is_ignored_without_optimize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "no-such-backend")
    server._preflight_model_backends(_config(optimize=False, memory_enabled=False))


def test_relevance_unknown_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", SECRET)
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER" in message
    assert SECRET not in message


def test_relevance_memory_reuse_needs_a_memory_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "memory")
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_EMBEDDER_PROVIDER" in message


def test_relevance_valid_selection_is_constructed_once_and_closed(
    monkeypatch: pytest.MonkeyPatch, recorder: type[_Recorder]
) -> None:
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "recorder")
    server._preflight_model_backends(_config(memory_enabled=False))
    assert len(recorder.instances) == 1
    assert recorder.instances[0].closed


def test_relevance_remote_embedder_is_refused_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_MODEL", "m")
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_OFFLINE" in message
    assert "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER" in message


# ----------------------------------------------------------------- Kompress


class _Shortener:
    instances: list[_Shortener] = []

    def __init__(self, **options: Any) -> None:
        self.name = "shortener"
        self.options = options
        self.closed = False
        _Shortener.instances.append(self)

    def compress_text(
        self, content: str, *, target_ratio: float | None, question: str | None
    ) -> str:
        return content[: len(content) // 2]

    def close(self) -> None:
        self.closed = True


def test_kompress_unknown_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", SECRET)
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_KOMPRESS_PROVIDER" in message
    assert SECRET not in message


def test_kompress_malformed_options_are_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    KOMPRESS_BACKENDS.register("shortener", _Shortener)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "shortener")
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER_OPTIONS", "{not json " + SECRET)
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_KOMPRESS_PROVIDER_OPTIONS" in message
    assert SECRET not in message


def test_kompress_provider_and_endpoint_together_are_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    KOMPRESS_BACKENDS.register("shortener", _Shortener)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "shortener")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "http://127.0.0.1:9")
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_KOMPRESS_ENDPOINT" in message


def test_kompress_valid_provider_is_constructed_once_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Shortener.instances = []
    KOMPRESS_BACKENDS.register("shortener", _Shortener)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "shortener")
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER_OPTIONS", '{"level": 2}')
    server._preflight_model_backends(_config(memory_enabled=False))
    assert len(_Shortener.instances) == 1
    assert _Shortener.instances[0].options == {"level": 2}
    assert _Shortener.instances[0].closed


def test_kompress_third_party_error_text_is_not_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(**_: Any) -> Any:
        raise RuntimeError(f"bad token {SECRET}")

    KOMPRESS_BACKENDS.register("broken", broken)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "broken")
    message = _refused(_config(memory_enabled=False), capsys)
    assert "RuntimeError" in message
    assert SECRET not in message


def test_kompress_provider_that_dials_out_is_refused_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Remote(_Shortener):
        def __init__(self, **options: Any) -> None:
            guard_egress("example Kompress service", "kompress.example.com")
            super().__init__(**options)

    KOMPRESS_BACKENDS.register("remote-plugin", Remote)
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "remote-plugin")
    message = _refused(_config(memory_enabled=False), capsys)
    assert "HEADROOM_OFFLINE" in message
    assert "HEADROOM_KOMPRESS_PROVIDER" in message


def test_kompress_selection_is_ignored_without_optimize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    server._preflight_model_backends(_config(optimize=False, memory_enabled=False))


# ----------------------------------------------------------------- lifespan


def test_lifespan_refuses_a_bad_selection_before_serving(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("fastapi")
    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_HTTP2", "false")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    app = create_app(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pass

    # Driven directly: TestClient's portal turns the exit into a cancellation.
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(start())
    assert excinfo.value.code == 78
    assert "HEADROOM_KOMPRESS_PROVIDER" in capsys.readouterr().err
