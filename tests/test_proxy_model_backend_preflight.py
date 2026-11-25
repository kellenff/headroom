"""Startup checks for the operator's model-backend selections (proxy/server.py).

``_preflight_model_backends`` turns a selection that cannot work (an unknown
name, a malformed variable, a backend that fails to construct, or one that
would dial out under ``HEADROOM_OFFLINE``) into exit 78 with a message that
names the variable. Without it, each of these surfaced only as a quiet
degradation after startup: memory off, relevance on BM25, Kompress off.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import traceback
from pathlib import Path
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


@pytest.mark.parametrize("offline", [False, True], ids=["bad_selection", "offline_remote"])
def test_memory_selection_is_ignored_in_stateless_mode(
    monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    # Stateless mode builds no MemoryHandler, so nothing uses the embedder.
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")
    if offline:
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", SECRET)
        monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "8")
    config = _config(stateless=True)
    server._preflight_model_backends(config)
    assert not server._memory_handler_enabled(config)


def test_memory_backend_declaring_another_width_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recorder: type[_Recorder],
) -> None:
    # The memory factory refuses this when memory starts, and memory then
    # stays off until restart; the startup check refuses it first.
    class NativeWidth(_Recorder):
        def __init__(self, model: str | None = None, dimension: int | None = None, **options: Any):
            super().__init__(model=model, dimension=dimension, **options)
            self.dimension = 1024  # ignores the dimension it was given

    EMBEDDING_BACKENDS.register("native-width", NativeWidth)
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "native-width")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "768")
    monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", SECRET)
    message = _refused(_config(), capsys)
    assert "HEADROOM_EMBEDDER_DIMENSION" in message
    assert "HEADROOM_EMBEDDER_PROVIDER" in message
    assert SECRET not in message
    assert len(recorder.instances) == 1
    assert recorder.instances[0].closed


def test_memory_backend_without_an_int_width_is_accepted(
    monkeypatch: pytest.MonkeyPatch, recorder: type[_Recorder]
) -> None:
    class UnknownWidth(_Recorder):
        def __init__(self, model: str | None = None, dimension: int | None = None, **options: Any):
            super().__init__(model=model, dimension=dimension, **options)
            self.dimension = None  # type: ignore[assignment]

    EMBEDDING_BACKENDS.register("unknown-width", UnknownWidth)
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "unknown-width")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "768")
    server._preflight_model_backends(_config())
    assert recorder.instances[-1].closed


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


def test_relevance_selection_is_checked_without_optimize(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # /v1/compress runs the content routers, and so the relevance scorer,
    # whether or not proxied traffic is optimized.
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "no-such-backend")
    message = _refused(_config(optimize=False, memory_enabled=False), capsys)
    assert "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER" in message


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


def test_kompress_selection_is_checked_without_optimize(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # /v1/compress still reaches Kompress with optimize off.
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    message = _refused(_config(optimize=False, memory_enabled=False), capsys)
    assert "HEADROOM_KOMPRESS_PROVIDER" in message


_KOMPRESS_OFF = {
    "disable_kompress": {"disable_kompress": True},
    "both_providers_disabled": {
        "disable_kompress_anthropic": True,
        "disable_kompress_openai": True,
    },
    "selection_without_kompress": {"compressors": {"smart_crusher", "log"}},
    "per_provider_override_wins": {
        "compressors": {"kompress"},
        "disable_kompress_anthropic": True,
        "disable_kompress_openai": True,
    },
}


@pytest.mark.parametrize("overrides", list(_KOMPRESS_OFF.values()), ids=list(_KOMPRESS_OFF))
@pytest.mark.parametrize("with_endpoint", [False, True], ids=["provider", "provider_and_endpoint"])
def test_kompress_selection_is_ignored_when_kompress_is_off(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], with_endpoint: bool
) -> None:
    # Turning Kompress off is how an operator gets past a broken plugin.
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    if with_endpoint:
        monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "http://127.0.0.1:9")
    server._preflight_model_backends(_config(memory_enabled=False, **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"disable_kompress": True, "disable_kompress_openai": False},
        {"compressors": {"smart_crusher"}, "disable_kompress_anthropic": False},
        {"disable_kompress_anthropic": True},
    ],
    ids=["global_off_openai_on", "selection_off_anthropic_on", "one_provider_off"],
)
def test_kompress_selection_is_checked_while_one_provider_runs_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    overrides: dict[str, Any],
) -> None:
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    message = _refused(_config(memory_enabled=False, **overrides), capsys)
    assert "HEADROOM_KOMPRESS_PROVIDER" in message


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        *_KOMPRESS_OFF.values(),
        {"disable_kompress": True, "disable_kompress_openai": False},
        {"compressors": {"smart_crusher"}, "disable_kompress_anthropic": False},
        {"compressors": {"*"}},
        {"disable_kompress_openai": True},
    ],
)
def test_kompress_gate_matches_the_routers_the_proxy_builds(overrides: dict[str, Any]) -> None:
    """The startup gate and HeadroomProxy.__init__ must not drift apart."""
    from headroom.transforms.content_router import ContentRouter

    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, **overrides)
    proxy = server.HeadroomProxy(config)
    routers = [
        transform
        for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline, proxy._no_ccr_pipeline())
        for transform in pipeline.transforms
        if isinstance(transform, ContentRouter)
    ]
    assert routers
    runs_kompress = any(router.config.enable_kompress for router in routers)
    assert server._kompress_can_run(ProxyConfig(**overrides)) is runs_kompress


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


def _quiet_app_config(**overrides: Any) -> ProxyConfig:
    values: dict[str, Any] = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
    }
    values.update(overrides)
    return ProxyConfig(**values)


def _quiet_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_HTTP2", "false")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")


def test_lifespan_checks_backends_after_a_degraded_rust_core_turns_optimize_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _quiet_app_env(monkeypatch)
    monkeypatch.setattr(server, "_check_rust_core", lambda: ("disabled", "simulated"))
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    config = _quiet_app_config()
    app = server.create_app(config)

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(start())
    assert excinfo.value.code == 78
    assert config.optimize is False  # the degraded core did turn it off
    assert "HEADROOM_KOMPRESS_PROVIDER" in capsys.readouterr().err


class _StopLifespan(Exception):
    pass


def _stop_after_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def stop() -> None:
        raise _StopLifespan

    monkeypatch.setattr(server, "_configure_observability_or_refuse", stop)


@pytest.mark.parametrize("already_checked", [False, True], ids=["embedded", "after_run_server"])
def test_lifespan_checks_a_config_only_once(
    monkeypatch: pytest.MonkeyPatch, already_checked: bool
) -> None:
    _quiet_app_env(monkeypatch)
    _stop_after_preflight(monkeypatch)
    _Shortener.instances = []
    KOMPRESS_BACKENDS.register("shortener", _Shortener)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "shortener")
    config = _quiet_app_config()
    if already_checked:
        server._preflight_model_backends_once(config)
        assert len(_Shortener.instances) == 1
    app = server.create_app(config)

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(_StopLifespan):
        asyncio.run(start())
    assert len(_Shortener.instances) == 1


# ----------------------------------------------------------------- refusal tracebacks


def _raise_with_secret(**_: Any) -> Any:
    raise RuntimeError(f"cannot log in with {SECRET}")


class _DialsOutAfterAFailure(_Shortener):
    """A plugin that falls back to a remote service after a local error."""

    def __init__(self, **options: Any) -> None:
        try:
            raise RuntimeError(f"local model rejected {SECRET}")
        except RuntimeError:
            guard_egress("example Kompress service", "kompress.example.com")
        super().__init__(**options)


def _broken_memory(monkeypatch: pytest.MonkeyPatch) -> ProxyConfig:
    EMBEDDING_BACKENDS.register("broken", _raise_with_secret)
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "broken")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "4")
    monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", SECRET)
    return _config()


def _broken_relevance(monkeypatch: pytest.MonkeyPatch) -> ProxyConfig:
    EMBEDDING_BACKENDS.register("broken", _raise_with_secret)
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "broken")
    monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_API_KEY", SECRET)
    return _config(memory_enabled=False)


def _broken_kompress(monkeypatch: pytest.MonkeyPatch) -> ProxyConfig:
    KOMPRESS_BACKENDS.register("broken", _raise_with_secret)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "broken")
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER_OPTIONS", json.dumps({"token": SECRET}))
    return _config(memory_enabled=False)


def _offline_kompress(monkeypatch: pytest.MonkeyPatch) -> ProxyConfig:
    KOMPRESS_BACKENDS.register("dials-out", _DialsOutAfterAFailure)
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "dials-out")
    return _config(memory_enabled=False)


_BROKEN_SELECTIONS = {
    "memory": _broken_memory,
    "relevance": _broken_relevance,
    "kompress": _broken_kompress,
    "kompress_offline": _offline_kompress,
}


@pytest.mark.parametrize("select", list(_BROKEN_SELECTIONS.values()), ids=list(_BROKEN_SELECTIONS))
def test_a_refusal_traceback_carries_no_third_party_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], select: Any
) -> None:
    """Starlette sends ``traceback.format_exc()`` of the exit to uvicorn, which logs it."""
    config = select(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        server._preflight_model_backends(config)
    assert excinfo.value.code == 78
    assert SECRET not in "".join(traceback.format_exception(excinfo.value))
    assert SECRET not in capsys.readouterr().err


def test_the_lifespan_failure_message_carries_no_third_party_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What uvicorn logs is the ``lifespan.startup.failed`` message Starlette sends."""
    _quiet_app_env(monkeypatch)
    _broken_kompress(monkeypatch)
    app = server.create_app(_quiet_app_config())
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    with pytest.raises(SystemExit):
        asyncio.run(app.router.lifespan({"type": "lifespan", "app": app}, receive, send))
    failed = [message for message in sent if message["type"] == "lifespan.startup.failed"]
    assert failed
    assert "SystemExit" in failed[0]["message"]
    assert SECRET not in failed[0]["message"]


# ----------------------------------------------------------------- launch


def _run_server_without_uvicorn(
    monkeypatch: pytest.MonkeyPatch, config: ProxyConfig, **kwargs: Any
) -> list[tuple[Any, ...]]:
    """Run ``run_server`` up to the point it hands off to uvicorn."""
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(server, "_run_uvicorn", lambda *args: calls.append(args))
    monkeypatch.setattr(server, "create_app", lambda config=None: object())
    monkeypatch.setattr("headroom.agent_savings.seed_proxy_env_defaults", lambda: [])
    # Recorded so the payload run_server writes is removed afterwards.
    monkeypatch.setenv(server._MULTI_WORKER_CONFIG_ENV, "")
    server.run_server(config, print_banner=False, **kwargs)
    return calls


@pytest.mark.parametrize("workers", [1, 2])
def test_run_server_refuses_before_uvicorn_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workers: int
) -> None:
    # Raised inside the ASGI lifespan, the exit became uvicorn's status 3,
    # and with workers > 1 the supervisor started the worker again forever.
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "no-such-backend")
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(SystemExit) as excinfo:
        calls = _run_server_without_uvicorn(monkeypatch, ProxyConfig(), workers=workers)
    assert excinfo.value.code == 78
    assert calls == []
    assert os.environ[server._MULTI_WORKER_CONFIG_ENV] == ""
    assert "HEADROOM_KOMPRESS_PROVIDER" in capsys.readouterr().err


def test_run_server_checks_a_single_worker_config_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _Shortener.instances = []
    KOMPRESS_BACKENDS.register("shortener", _Shortener)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "shortener")
    config = ProxyConfig()
    calls = _run_server_without_uvicorn(monkeypatch, config)
    assert len(calls) == 1
    assert len(_Shortener.instances) == 1
    # The lifespan's call for the same config does nothing.
    server._preflight_model_backends_once(config)
    assert len(_Shortener.instances) == 1


def test_run_server_checks_once_for_all_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    _Shortener.instances = []
    KOMPRESS_BACKENDS.register("shortener", _Shortener)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "shortener")
    calls = _run_server_without_uvicorn(monkeypatch, ProxyConfig(port=39111), workers=2)
    assert len(calls) == 1
    assert len(_Shortener.instances) == 1
    # Each worker rebuilds its config from the payload; the check travels with it.
    worker_config = server._proxy_config_from_env()
    assert worker_config.port == 39111
    server._preflight_model_backends_once(worker_config)
    assert len(_Shortener.instances) == 1


def test_an_unchecked_config_leaves_the_check_to_the_worker() -> None:
    payload = server._proxy_config_payload(ProxyConfig())
    assert server._MODEL_BACKENDS_CHECKED not in payload


def test_run_server_exports_offline_before_checking_for_workers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Remote(_Shortener):
        def __init__(self, **options: Any) -> None:
            guard_egress("example Kompress service", "kompress.example.com")
            super().__init__(**options)

    KOMPRESS_BACKENDS.register("remote-plugin", Remote)
    monkeypatch.setenv("HEADROOM_KOMPRESS_PROVIDER", "remote-plugin")
    # Recorded as unset, so the export run_server makes is removed afterwards.
    monkeypatch.setenv("HEADROOM_OFFLINE", "0")
    monkeypatch.delenv("HEADROOM_OFFLINE")
    with pytest.raises(SystemExit) as excinfo:
        _run_server_without_uvicorn(monkeypatch, ProxyConfig(offline=True), workers=2)
    assert excinfo.value.code == 78
    assert "HEADROOM_OFFLINE" in capsys.readouterr().err


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("workers", [1, 2])
def test_the_process_exits_78(tmp_path: Path, workers: int) -> None:
    """End to end: the operator's supervisor sees EX_CONFIG, not uvicorn's 3."""
    pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    repo_root = Path(server.__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")])),
            "HEADROOM_KOMPRESS_PROVIDER": "no-such-backend",
            "HEADROOM_STATELESS": "1",
            "HEADROOM_SKIP_UPSTREAM_CHECK": "1",
            "HEADROOM_REQUIRE_RUST_CORE": "false",
        }
    )
    command = [sys.executable, "-m", "headroom.proxy.server", "--port", str(_free_port())]
    if workers > 1:
        command += ["--workers", str(workers)]
    # Before the fix, two workers were started again until this timeout.
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 78, result.stderr[-2000:]
    assert result.stderr.count("FATAL: the Kompress provider") == 1
    assert "Traceback" not in result.stderr
