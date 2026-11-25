"""Every process that opens a memory database picks the same embedder.

The proxy, the memory MCP server, the sync subprocess, ``headroom memory`` and
``headroom.memory.Memory`` each build their own LocalBackend. With
``HEADROOM_EMBEDDER_*`` unset each keeps its historical default; set, all of
them use it, so they can share one database (whose embedder identity check
would refuse the odd one out). ``Memory(embedder=...)`` lets library callers
bring their own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import headroom.cli.memory as memory_cli
from headroom.evals.memory.runner_v2 import LoCoMoEvaluatorV2
from headroom.memory.easy import Memory
from headroom.memory.embedder_identity import read_embedder_identity
from headroom.memory.factory import _reset_embedder_cache_for_tests
from headroom.memory.sync import _build_sync_backend

PROXY_ENV = {
    "HEADROOM_EMBEDDER_PROVIDER": "openai",
    "HEADROOM_EMBEDDER_MODEL": "nomic-embed-text",
    "HEADROOM_EMBEDDER_BASE_URL": "http://gpu-box:8080/v1",
    "HEADROOM_EMBEDDER_DIMENSION": "768",
}


@pytest.fixture(autouse=True)
def _no_embedder_env(monkeypatch: pytest.MonkeyPatch):
    for var in (*PROXY_ENV, "HEADROOM_EMBEDDER_RUNTIME", "HEADROOM_EMBEDDER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    _reset_embedder_cache_for_tests()
    yield
    _reset_embedder_cache_for_tests()


def _set_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROXY_ENV.items():
        monkeypatch.setenv(name, value)


def _assert_proxy_embedder(config: Any) -> None:
    assert config.embedder_backend == "external"
    assert config.embedder_backend_name == "openai"
    assert config.embedder_options == {
        "model": "nomic-embed-text",
        "base_url": "http://gpu-box:8080/v1",
    }
    assert config.vector_dimension == 768


# =============================================================================
# Sync subprocess
# =============================================================================


def test_sync_default_is_unchanged(tmp_path: Path) -> None:
    config = _build_sync_backend(str(tmp_path / "memory.db"))._config
    assert (config.embedder_backend, config.vector_dimension) == ("onnx", 384)


def test_sync_uses_the_proxy_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_proxy_env(monkeypatch)
    _assert_proxy_embedder(_build_sync_backend(str(tmp_path / "memory.db"))._config)


def test_sync_refuses_invalid_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("HEADROOM_EMBEDDER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_DIMENSION"):
        _build_sync_backend(str(tmp_path / "memory.db"))


# =============================================================================
# headroom memory repair-supersession
# =============================================================================


class _CapturingBackend:
    configs: list[Any] = []

    def __init__(self, config: Any) -> None:
        _CapturingBackend.configs.append(config)

    async def detach_supersession(self, old: str, new: str) -> tuple[str, str]:
        return old, new

    async def close(self) -> None:
        return None


@pytest.fixture
def captured_repair_configs(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    _CapturingBackend.configs = []
    monkeypatch.setattr("headroom.memory.backends.local.LocalBackend", _CapturingBackend)
    return _CapturingBackend.configs


def test_repair_default_is_unchanged(captured_repair_configs: list[Any]) -> None:
    asyncio.run(memory_cli._apply_supersession_repair("memory.db", "old", "new", 384))
    (config,) = captured_repair_configs
    assert (config.embedder_backend, config.vector_dimension) == ("local", 384)


def test_repair_uses_the_proxy_embedder(
    captured_repair_configs: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_proxy_env(monkeypatch)
    asyncio.run(memory_cli._apply_supersession_repair("memory.db", "old", "new", 768))
    _assert_proxy_embedder(captured_repair_configs[0])


def test_repair_refuses_a_dimension_the_stored_vectors_contradict(
    captured_repair_configs: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_proxy_env(monkeypatch)
    with pytest.raises(ValueError, match="HEADROOM_EMBEDDER_DIMENSION=768.*384 dimensions"):
        asyncio.run(memory_cli._apply_supersession_repair("memory.db", "old", "new", 384))
    assert captured_repair_configs == []


# =============================================================================
# Memory eval runner
# =============================================================================


def test_eval_runner_default_is_unchanged(tmp_path: Path) -> None:
    config = LoCoMoEvaluatorV2._default_backend(str(tmp_path / "memory.db"))._config
    assert (config.embedder_backend, config.vector_dimension) == ("local", 384)


def test_eval_runner_evaluates_the_configured_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_proxy_env(monkeypatch)
    _assert_proxy_embedder(LoCoMoEvaluatorV2._default_backend(str(tmp_path / "m.db"))._config)


# =============================================================================
# headroom.memory.Memory
# =============================================================================


def _topic_embedder(texts: list[str]) -> list[list[float]]:
    return [
        [1.0 if "python" in text.lower() else 0.0, 1.0 if "rust" in text.lower() else 0.0, 0.2]
        for text in texts
    ]


class _ClosableBackend:
    name = "mine"
    model_name = "my-model"
    dimension = 3

    def __init__(self) -> None:
        self.closed = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _topic_embedder(texts)

    def close(self) -> None:
        self.closed += 1


async def test_memory_runs_on_a_plain_function(tmp_path: Path) -> None:
    memory = Memory(db_path=tmp_path / "memory.db", embedder=_topic_embedder, vector_dimension=3)
    try:
        await memory.save("User likes Python", user_id="alice")
        await memory.save("User likes Rust", user_id="alice")
        results = await memory.search("python tooling", user_id="alice", top_k=1)
        assert [result.content for result in results] == ["User likes Python"]
    finally:
        await memory.close()
    assert read_embedder_identity(tmp_path / "memory.db") == "custom:callable:3"


async def test_memory_leaves_the_callers_embedder_open(tmp_path: Path) -> None:
    mine = _ClosableBackend()
    memory = Memory(db_path=tmp_path / "memory.db", embedder=mine, vector_dimension=3)
    await memory.save("User likes Python", user_id="alice")
    await memory.close()
    assert mine.closed == 0


async def test_an_explicit_embedder_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_EMBEDDER_PROVIDER", "openai")  # invalid: no dimension
    memory = Memory(db_path=tmp_path / "memory.db", embedder=_topic_embedder, vector_dimension=3)
    try:
        await memory.save("User likes Python", user_id="alice")
    finally:
        await memory.close()


async def test_without_an_embedder_memory_uses_the_proxy_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_proxy_env(monkeypatch)
    memory = Memory(db_path=tmp_path / "memory.db")
    await memory._ensure_initialized()
    _assert_proxy_embedder(memory._backend._config)


async def test_vector_dimension_alone_overrides_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_proxy_env(monkeypatch)
    memory = Memory(db_path=tmp_path / "memory.db", vector_dimension=1024)
    await memory._ensure_initialized()
    assert memory._backend._config.vector_dimension == 1024


async def test_memory_default_is_unchanged(tmp_path: Path) -> None:
    memory = Memory(db_path=tmp_path / "memory.db")
    await memory._ensure_initialized()
    config = memory._backend._config
    assert (config.embedder_backend, config.vector_dimension, config.embedder) == (
        "local",
        384,
        None,
    )


@pytest.mark.parametrize("kwargs", [{"embedder": _topic_embedder}, {"vector_dimension": 3}])
def test_embedder_options_are_for_the_local_backend_only(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="'local' backend only"):
        Memory(backend="qdrant-neo4j", **kwargs)
