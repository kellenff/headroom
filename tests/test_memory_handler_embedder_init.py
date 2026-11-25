"""Proxy memory init with pluggable embedders: which failures are retried.

A configuration error (bad ``HEADROOM_EMBEDDER_*``, an embedder mismatch) or an
air-gap refusal cannot be fixed by retrying, so memory reports it once and stays
off; a transient failure (a database that cannot open yet) keeps today's
per-request retry. Neither ever propagates into a request.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.memory.factory import _reset_embedder_cache_for_tests
from headroom.models.backends import EMBEDDING_BACKENDS, register_embedding_backend
from headroom.offline import OfflineEgressBlocked
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

_EMBEDDER_VARS = (
    "HEADROOM_EMBEDDER_PROVIDER",
    "HEADROOM_EMBEDDER_RUNTIME",
    "HEADROOM_EMBEDDER_MODEL",
    "HEADROOM_EMBEDDER_BASE_URL",
    "HEADROOM_EMBEDDER_API_KEY",
    "HEADROOM_EMBEDDER_DIMENSION",
    "HEADROOM_EMBEDDER_HEADERS",
    "HEADROOM_EMBEDDER_OPTIONS",
    "HEADROOM_OFFLINE",
)


class SharedBackend:
    """A registry backend; one instance is shared by every memory backend."""

    name = "shared-fake"
    instances: list[SharedBackend] = []

    def __init__(self, model: str | None = None, dimension: int | None = None, **_: Any):
        self.model_name = model or "shared-model"
        self.dimension = dimension
        self.closed = 0
        SharedBackend.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text)), 0.0] for text in texts]

    def close(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    for var in _EMBEDDER_VARS:
        monkeypatch.delenv(var, raising=False)
    EMBEDDING_BACKENDS._reset_for_tests()
    _reset_embedder_cache_for_tests()
    SharedBackend.instances = []
    yield
    EMBEDDING_BACKENDS._reset_for_tests()
    _reset_embedder_cache_for_tests()


def _handler(tmp_path: Path) -> MemoryHandler:
    return MemoryHandler(MemoryConfig(enabled=True, db_path=str(tmp_path / "memory.db")))


def _count_inits(monkeypatch: pytest.MonkeyPatch, handler: MemoryHandler) -> list[int]:
    calls: list[int] = []
    original = handler._init_backend_locked

    async def counted() -> None:
        calls.append(1)
        await original()

    monkeypatch.setattr(handler, "_init_backend_locked", counted)
    return calls


async def test_a_configuration_error_is_reported_once_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")  # no DIMENSION
    handler = _handler(tmp_path)
    calls = _count_inits(monkeypatch, handler)

    with caplog.at_level(logging.ERROR, logger="headroom.proxy.memory_handler"):
        for _ in range(3):
            await handler._ensure_initialized()

    assert calls == [1]
    assert handler.initialized is False
    assert handler.backend is None
    assert "HEADROOM_EMBEDDER_DIMENSION" in (handler.init_error or "")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "HEADROOM_EMBEDDER_DIMENSION" in errors[0].getMessage()
    assert "disabled until restart" in errors[0].getMessage()
    assert not (tmp_path / "memory.db").exists()


async def test_a_backend_error_names_the_variable_that_chose_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for name, value in {
        "HEADROOM_EMBEDDER_PROVIDER": "openai",
        "HEADROOM_EMBEDDER_BASE_URL": "http://embeddings.invalid/v1",
        "HEADROOM_EMBEDDER_API_KEY": "sk-never-logged",
        "HEADROOM_EMBEDDER_DIMENSION": "3",
    }.items():
        monkeypatch.setenv(name, value)  # no MODEL, which this endpoint needs
    handler = _handler(tmp_path)

    with caplog.at_level(logging.ERROR, logger="headroom.proxy.memory_handler"):
        await handler._ensure_initialized()
        await handler._ensure_initialized()

    assert "model is required" in (handler.init_error or "")
    (error,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "HEADROOM_EMBEDDER_PROVIDER" in error.getMessage()
    assert "sk-never-logged" not in caplog.text
    assert not (tmp_path / "memory.db").exists()


async def test_an_embedder_mismatch_is_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_embedding_backend("shared-fake", SharedBackend)
    for name, value in {
        "HEADROOM_EMBEDDER_PROVIDER": "shared-fake",
        "HEADROOM_EMBEDDER_MODEL": "first-model",
        "HEADROOM_EMBEDDER_DIMENSION": "3",
    }.items():
        monkeypatch.setenv(name, value)
    writer = _handler(tmp_path)
    await writer._ensure_initialized()
    assert writer.initialized
    await writer.close()

    monkeypatch.setenv("HEADROOM_EMBEDDER_MODEL", "second-model")
    reader = _handler(tmp_path)
    calls = _count_inits(monkeypatch, reader)
    await reader._ensure_initialized()
    await reader._ensure_initialized()
    assert calls == [1]
    assert "shared-fake:first-model:3" in (reader.init_error or "")
    assert "shared-fake:second-model:3" in (reader.init_error or "")


async def test_a_transient_failure_is_retried_and_the_shared_embedder_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a failed init closes the half-built backend. That close must
    not close the process-wide embedder the retry is about to reuse."""
    register_embedding_backend("shared-fake", SharedBackend)
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "shared-fake")
    monkeypatch.setenv("HEADROOM_EMBEDDER_DIMENSION", "3")

    from headroom.memory.adapters import sqlite_graph

    real_graph_store = sqlite_graph.SQLiteGraphStore
    failures = iter([sqlite3.OperationalError("unable to open database file")])

    def flaky_graph_store(*args: Any, **kwargs: Any) -> Any:
        for error in failures:
            raise error
        return real_graph_store(*args, **kwargs)

    monkeypatch.setattr(sqlite_graph, "SQLiteGraphStore", flaky_graph_store)
    handler = _handler(tmp_path)
    calls = _count_inits(monkeypatch, handler)

    await handler._ensure_initialized()
    assert handler.initialized is False
    assert handler.init_error is None  # transient: not recorded as permanent

    await handler._ensure_initialized()
    assert calls == [1, 1]
    assert handler.initialized is True
    saved = await handler.backend.save_memory("retry works", user_id="u")
    assert saved.embedding is not None

    (backend,) = SharedBackend.instances
    assert backend.closed == 0
    await handler.close()


async def test_an_air_gap_refusal_turns_memory_off_without_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "HEADROOM_OFFLINE": "1",
        "HEADROOM_EMBEDDER_PROVIDER": "openai",
        "HEADROOM_EMBEDDER_MODEL": "m",
        "HEADROOM_EMBEDDER_BASE_URL": "https://embeddings.example/v1",
        "HEADROOM_EMBEDDER_DIMENSION": "3",
    }.items():
        monkeypatch.setenv(name, value)
    handler = _handler(tmp_path)
    calls = _count_inits(monkeypatch, handler)

    await handler._ensure_initialized()  # must not raise the BaseException
    await handler._ensure_initialized()

    assert calls == [1]
    assert handler.initialized is False
    assert "HEADROOM_OFFLINE" in (handler.init_error or "")
    assert "embeddings.example" in (handler.init_error or "")
    assert not (tmp_path / "memory.db").exists()


async def test_a_warmup_refusal_turns_memory_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RefusingEmbedder:
        async def embed(self, text: str) -> Any:
            raise OfflineEgressBlocked("memory embedding API", "embeddings.example")

    closed: list[str] = []

    async def close() -> None:
        closed.append("closed")

    handler = _handler(tmp_path)
    handler._initialized = True
    handler._backend = SimpleNamespace(
        _hierarchical_memory=SimpleNamespace(_embedder=RefusingEmbedder()), close=close
    )

    assert await handler.warmup_embedder() is False
    assert closed == ["closed"]
    assert handler.backend is None
    assert handler.initialized is False
    assert "HEADROOM_OFFLINE" in (handler.init_error or "")

    calls = _count_inits(monkeypatch, handler)
    await handler._ensure_initialized()
    assert calls == []


async def test_health_status_keeps_its_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")
    handler = _handler(tmp_path)
    await handler._ensure_initialized()
    assert set(handler.health_status()) == {
        "enabled",
        "backend",
        "initialized",
        "native_tool",
        "bridge_enabled",
    }
