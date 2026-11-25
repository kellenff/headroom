"""Memory on pluggable embedding backends (``headroom.models.backends``).

Covers the EXTERNAL embedder (registry backends), embedders injected by the
caller, the async adapter over a synchronous backend, embedder ownership, and
the embedding-space fingerprint that keeps processes from mixing models in one
database. Everything runs offline: backends here are fakes or loopback-free.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from headroom.memory.adapters.backend_embedder import BackendEmbedder
from headroom.memory.backends.local import (
    LocalBackend,
    LocalBackendConfig,
    resolve_embedder_backend,
)
from headroom.memory.config import EmbedderBackend, MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.embedder_identity import (
    IDENTITY_CHECK_ENV,
    EmbedderMismatchError,
    embedder_identity,
    identity_check_enabled,
    read_embedder_identity,
    verify_embedder_identity,
)
from headroom.memory.factory import (
    _EMBEDDER_CACHE,
    _create_embedder,
    _reset_embedder_cache_for_tests,
    create_memory_system,
    embedder_is_shared,
)
from headroom.models.backends import (
    EMBEDDING_BACKENDS,
    EmbeddingBackendResponseError,
    UnknownBackendError,
    register_embedding_backend,
)

FAKE = "fake-memory"


class FakeBackend:
    """A conforming synchronous backend. Two keyword "topics" become two axes."""

    name = FAKE
    instances: list[FakeBackend] = []

    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
        *,
        native_dimension: int | None = None,
        **options: Any,
    ) -> None:
        self.model_name = model or "fake-default"
        # A backend that ignores the requested width reports its own.
        self.dimension = native_dimension if native_dimension is not None else dimension
        self.requested_dimension = dimension
        self.received_model = model
        self.options = options
        self.calls: list[list[str]] = []
        self.threads: list[str] = []
        self.closed = 0
        FakeBackend.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.threads.append(threading.current_thread().name)
        width = self.dimension or 3
        rows = []
        for text in texts:
            row = [0.0] * width
            row[0] = 3.0 if "python" in text.lower() else 0.0
            row[1] = 4.0 if "rust" in text.lower() else 0.0
            row[-1] += 0.1
            rows.append(row)
        return rows

    def close(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def _clean_registry_and_cache(monkeypatch: pytest.MonkeyPatch):
    EMBEDDING_BACKENDS._reset_for_tests()
    _reset_embedder_cache_for_tests()
    FakeBackend.instances = []
    register_embedding_backend(FAKE, FakeBackend)
    monkeypatch.delenv(IDENTITY_CHECK_ENV, raising=False)
    yield
    EMBEDDING_BACKENDS._reset_for_tests()
    _reset_embedder_cache_for_tests()


def external(**overrides: Any) -> MemoryConfig:
    settings: dict[str, Any] = {
        "embedder_backend": EmbedderBackend.EXTERNAL,
        "embedder_backend_name": FAKE,
        "vector_dimension": 3,
    }
    settings.update(overrides)
    return MemoryConfig(**settings)


# =============================================================================
# MemoryConfig
# =============================================================================


class TestMemoryConfig:
    def test_external_needs_a_backend_name(self) -> None:
        with pytest.raises(ValueError, match="embedder_backend_name is required"):
            MemoryConfig(embedder_backend=EmbedderBackend.EXTERNAL)

    def test_an_injected_embedder_needs_no_name(self) -> None:
        config = MemoryConfig(embedder_backend=EmbedderBackend.EXTERNAL, embedder=lambda t: t)
        assert config.embedder is not None

    @pytest.mark.parametrize("spec", ["", "Bad Name", ":model"])
    def test_spec_is_validated(self, spec: str) -> None:
        with pytest.raises(ValueError):
            external(embedder_backend_name=spec)

    def test_the_model_may_not_be_given_twice_differently(self) -> None:
        with pytest.raises(ValueError, match="set the model in one place"):
            external(embedder_backend_name=f"{FAKE}:a", embedder_options={"model": "b"})
        # The same model in both places is not a contradiction.
        external(embedder_backend_name=f"{FAKE}:a", embedder_options={"model": "a"})

    def test_an_option_dimension_must_match_the_index(self) -> None:
        with pytest.raises(ValueError, match="contradicts vector_dimension"):
            external(embedder_options={"dimension": 8})
        external(embedder_options={"dimension": 3})

    def test_options_must_be_a_dict(self) -> None:
        with pytest.raises(TypeError, match="embedder_options"):
            external(embedder_options=[("api_key", "x")])

    def test_repr_hides_options_and_the_embedder(self) -> None:
        config = external(embedder_options={"api_key": "sk-secret-value"}, embedder=FakeBackend())
        assert "sk-secret-value" not in repr(config)
        assert "FakeBackend" not in repr(config)

    def test_the_embedder_is_not_compared(self) -> None:
        assert external(embedder=FakeBackend()) == external(embedder=FakeBackend())


# =============================================================================
# EXTERNAL embedders from the registry
# =============================================================================


class TestExternalEmbedder:
    @pytest.mark.parametrize(
        ("spec", "options", "expected"),
        [
            (f"{FAKE}:from-spec", {}, "from-spec"),
            (FAKE, {"model": "from-options"}, "from-options"),
            (FAKE, {}, None),
        ],
    )
    def test_model_comes_from_the_spec_or_options_only(
        self, spec: str, options: dict[str, Any], expected: str | None
    ) -> None:
        # embedder_model always holds a MiniLM default; it must never leak to
        # a backend that serves something else.
        config = external(
            embedder_backend_name=spec,
            embedder_options=options,
            embedder_model="all-MiniLM-L6-v2",
        )
        embedder = _create_embedder(config)
        assert isinstance(embedder, BackendEmbedder)
        (backend,) = FakeBackend.instances
        assert backend.received_model == expected
        assert backend.requested_dimension == 3

    def test_options_are_forwarded(self) -> None:
        _create_embedder(external(embedder_options={"base_url": "http://x/v1", "api_key": "k"}))
        assert FakeBackend.instances[0].options == {"base_url": "http://x/v1", "api_key": "k"}

    async def test_a_dimension_mismatch_fails_before_any_file_is_created(
        self, tmp_path: Path
    ) -> None:
        config = external(
            db_path=tmp_path / "memory.db",
            embedder_options={"native_dimension": 768},
            vector_dimension=384,
        )
        with pytest.raises(ValueError, match="768-dimensional vectors.*vector_dimension=384"):
            await create_memory_system(config)
        assert list(tmp_path.iterdir()) == []
        # The discarded backend is released, and nothing broken is cached.
        assert FakeBackend.instances[0].closed == 1
        assert _EMBEDDER_CACHE == {}

    def test_unknown_backend(self) -> None:
        with pytest.raises(UnknownBackendError, match="no-such-backend"):
            _create_embedder(external(embedder_backend_name="no-such-backend"))

    def test_same_settings_share_one_embedder(self) -> None:
        first = _create_embedder(external(embedder_options={"base_url": "http://a/v1"}))
        again = _create_embedder(external(embedder_options={"base_url": "http://a/v1"}))
        assert again is first
        assert len(FakeBackend.instances) == 1

    def test_the_model_is_one_key_wherever_it_is_given(self) -> None:
        in_spec = _create_embedder(external(embedder_backend_name=f"{FAKE}:m"))
        in_options = _create_embedder(external(embedder_options={"model": "m"}))
        assert in_options is in_spec

    @pytest.mark.parametrize(
        "override",
        [
            {"embedder_backend_name": f"{FAKE}:other-model"},
            {"vector_dimension": 4},
            {"embedder_options": {"base_url": "http://b/v1"}},
            {"embedder_options": {"base_url": "http://a/v1", "api_key": "second-key"}},
        ],
    )
    def test_any_difference_gets_its_own_embedder(self, override: dict[str, Any]) -> None:
        base = {"embedder_options": {"base_url": "http://a/v1"}}
        first = _create_embedder(external(**base))
        assert _create_embedder(external(**(base | override))) is not first

    def test_cache_key_holds_no_raw_option_values(self) -> None:
        _create_embedder(external(embedder_options={"api_key": "sk-raw-secret"}))
        assert all("sk-raw-secret" not in part for key in _EMBEDDER_CACHE for part in key)

    def test_builtin_embedders_keep_their_unchecked_dimension(self) -> None:
        # Ollama reports 768 against the 384 default index; refusing that up
        # front would break every existing Ollama configuration.
        embedder = _create_embedder(
            MemoryConfig(embedder_backend=EmbedderBackend.OLLAMA, embedder_model="nomic")
        )
        assert embedder.dimension == 768


# =============================================================================
# Embedders supplied by the caller
# =============================================================================


class AsyncEmbedder:
    """A memory ``Embedder`` the caller wrote themselves."""

    dimension = 3
    model_name = "my-async-model"
    max_tokens = 512

    def __init__(self) -> None:
        self.closed = 0

    async def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(text) for text in texts]

    async def close(self) -> None:
        self.closed += 1


class TestInjectedEmbedder:
    def test_a_sync_backend_is_wrapped_and_not_cached(self) -> None:
        backend = FakeBackend(dimension=3)
        first = _create_embedder(MemoryConfig(embedder=backend, vector_dimension=3))
        second = _create_embedder(MemoryConfig(embedder=backend, vector_dimension=3))
        assert isinstance(first, BackendEmbedder) and first.backend is backend
        assert second is not first
        assert _EMBEDDER_CACHE == {}

    async def test_a_plain_function_is_enough(self) -> None:
        seen: list[list[str]] = []

        def fn(texts: list[str]) -> list[list[float]]:
            seen.append(texts)
            return [[0.0, 2.0, 0.0] for _ in texts]

        embedder = _create_embedder(MemoryConfig(embedder=fn, vector_dimension=3))
        assert embedder.model_name == "callable"
        np.testing.assert_allclose(await embedder.embed("hello"), [0.0, 1.0, 0.0])
        assert seen == [["hello"]]

    def test_an_async_embedder_is_used_as_is(self) -> None:
        mine = AsyncEmbedder()
        assert _create_embedder(MemoryConfig(embedder=mine, vector_dimension=3)) is mine

    def test_a_partial_async_embedder_is_refused(self) -> None:
        class HalfAsync:
            name = "half"
            model_name = "half"
            dimension = 3

            async def embed(self, texts: list[str]) -> Any:
                return []

        with pytest.raises(TypeError, match="memory Embedder protocol"):
            _create_embedder(MemoryConfig(embedder=HalfAsync(), vector_dimension=3))

    def test_something_that_cannot_embed_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Expected an EmbeddingBackend"):
            _create_embedder(MemoryConfig(embedder=42, vector_dimension=3))

    @pytest.mark.parametrize("make", [lambda: FakeBackend(dimension=5), AsyncEmbedder])
    def test_a_declared_dimension_must_match_the_index(self, make: Any) -> None:
        with pytest.raises(ValueError, match="vector_dimension=4"):
            _create_embedder(MemoryConfig(embedder=make(), vector_dimension=4))

    def test_it_wins_over_embedder_backend(self) -> None:
        mine = AsyncEmbedder()
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.OLLAMA, embedder=mine, vector_dimension=3
        )
        assert _create_embedder(config) is mine


# =============================================================================
# BackendEmbedder
# =============================================================================


class TestBackendEmbedder:
    async def test_rows_are_normalised_ordered_float32_and_empty_texts_are_zero(self) -> None:
        backend = FakeBackend(dimension=3)
        embedder = BackendEmbedder(backend, dimension=3)
        rows = await embedder.embed_batch(["python", "", "rust", "   "])
        np.testing.assert_allclose(rows[0], np.array([3.0, 0.0, 0.1]) / np.hypot(3.0, 0.1))
        np.testing.assert_allclose(rows[1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(rows[2], np.array([0.0, 4.0, 0.1]) / np.hypot(4.0, 0.1))
        np.testing.assert_allclose(rows[3], [0.0, 0.0, 0.0])
        assert all(row.dtype == np.float32 for row in rows)
        assert backend.calls == [["python", "rust"]]  # empty texts never sent
        single = await embedder.embed("python")
        np.testing.assert_allclose(single, rows[0])
        await embedder.close()

    async def test_an_empty_batch_never_reaches_the_backend(self) -> None:
        backend = FakeBackend(dimension=3)
        assert await BackendEmbedder(backend, dimension=3).embed_batch([]) == []
        assert backend.calls == []

    async def test_rows_of_the_wrong_width_are_refused(self) -> None:
        backend = FakeBackend(dimension=None)  # embeds 3 wide
        with pytest.raises(EmbeddingBackendResponseError, match="expected 4"):
            await BackendEmbedder(backend, dimension=4).embed("python")

    async def test_calls_run_on_its_own_threads(self) -> None:
        backend = FakeBackend(dimension=3)
        embedder = BackendEmbedder(backend, dimension=3)
        await embedder.embed("python")
        assert backend.threads[0].startswith("backend-embed")
        await embedder.close()

    def test_reports_the_backend_model_and_window(self) -> None:
        backend = FakeBackend(model="m", dimension=3)
        embedder = BackendEmbedder(backend, dimension=3)
        assert embedder.model_name == "m"
        assert embedder.dimension == 3
        assert embedder.max_tokens == 8192
        backend.max_tokens = 256  # learnt later, e.g. when a model loads
        assert embedder.max_tokens == 256

    @pytest.mark.parametrize("dimension", [0, -1, True, 3.0])
    def test_dimension_must_be_a_positive_int(self, dimension: Any) -> None:
        with pytest.raises(ValueError, match="dimension"):
            BackendEmbedder(FakeBackend(), dimension=dimension)

    async def test_close_is_idempotent_and_a_later_embed_revives_it(self) -> None:
        backend = FakeBackend(dimension=3)
        embedder = BackendEmbedder(backend, dimension=3)
        await embedder.embed("python")
        await embedder.close()
        await embedder.close()
        assert backend.closed == 2  # the backend contract: close() is idempotent
        assert embedder._executor is None
        np.testing.assert_allclose(np.linalg.norm(await embedder.embed("rust")), 1.0)
        assert embedder._executor is not None
        await embedder.close()

    async def test_close_resubmits_queued_calls_instead_of_failing_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_EMBED_CONCURRENCY", "1")
        started, release = threading.Event(), threading.Event()

        class Blocking(FakeBackend):
            def embed(self, texts: list[str]) -> list[list[float]]:
                if texts == ["first"]:
                    started.set()
                    release.wait(5)
                return super().embed(texts)

        backend = Blocking(dimension=3)
        embedder = BackendEmbedder(backend, dimension=3)
        first = asyncio.create_task(embedder.embed("first"))
        await asyncio.to_thread(started.wait, 5)
        queued = asyncio.create_task(embedder.embed("second"))
        await asyncio.sleep(0.05)  # "second" waits behind "first" on the one worker
        await embedder.close()
        release.set()
        first_row, queued_row = await asyncio.gather(first, queued)
        assert first_row.shape == queued_row.shape == (3,)
        assert ["second"] in backend.calls
        await embedder.close()

    async def test_a_cancelled_caller_is_still_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_EMBED_CONCURRENCY", "1")
        started, release = threading.Event(), threading.Event()

        class Blocking(FakeBackend):
            def embed(self, texts: list[str]) -> list[list[float]]:
                if texts == ["first"]:
                    started.set()
                    release.wait(5)
                return super().embed(texts)

        backend = Blocking(dimension=3)
        embedder = BackendEmbedder(backend, dimension=3)
        first = asyncio.create_task(embedder.embed("first"))
        await asyncio.to_thread(started.wait, 5)
        queued = asyncio.create_task(embedder.embed("second"))
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await first
        assert ["second"] not in backend.calls
        await embedder.close()

    async def test_a_failing_backend_close_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class BadClose(FakeBackend):
            def close(self) -> None:
                raise RuntimeError("secret detail http://key@host")

        embedder = BackendEmbedder(BadClose(dimension=3), dimension=3)
        with caplog.at_level(logging.WARNING):
            await embedder.close()
        assert "RuntimeError" in caplog.text
        assert "secret detail" not in caplog.text

    def test_repr_does_not_use_the_backends_own(self) -> None:
        class Leaky(FakeBackend):
            def __repr__(self) -> str:
                return "Leaky(api_key='sk-leak')"

        assert "sk-leak" not in repr(BackendEmbedder(Leaky(dimension=3), dimension=3))


# =============================================================================
# Ownership: shared embedders are never closed by one memory system
# =============================================================================


class TestEmbedderOwnership:
    async def test_close_respects_owns_embedder(self) -> None:
        for owns, expected in ((True, 1), (False, 0)):
            embedder = AsyncEmbedder()
            memory = HierarchicalMemory(
                store=_Closable(),
                vector_index=_Closable(),
                text_index=_Closable(),
                embedder=embedder,
                owns_embedder=owns,
            )
            await memory.close()
            assert embedder.closed == expected

    async def test_create_never_owns_a_cached_or_injected_embedder(self, tmp_path: Path) -> None:
        cached = await HierarchicalMemory.create(external(db_path=tmp_path / "a.db"))
        mine = AsyncEmbedder()
        injected = await HierarchicalMemory.create(
            MemoryConfig(db_path=tmp_path / "b.db", embedder=mine, vector_dimension=3)
        )
        assert cached._owns_embedder is False
        assert injected._owns_embedder is False
        assert embedder_is_shared(cached.config, cached.embedder)
        await cached.close()
        await injected.close()
        assert FakeBackend.instances[0].closed == 0
        assert mine.closed == 0

    async def test_closing_one_backend_leaves_the_shared_embedder_working(
        self, tmp_path: Path
    ) -> None:
        """Regression: the proxy closes and rebuilds backends; a close must not
        break the other backends sharing the process-wide embedder."""
        config = LocalBackendConfig(
            db_path=str(tmp_path / "a.db"),
            embedder_backend="external",
            embedder_backend_name=FAKE,
            vector_dimension=3,
        )
        first = LocalBackend(config)
        await first.save_memory("User likes Python", user_id="u")
        await first.close()

        retried = LocalBackend(replace(config, db_path=str(tmp_path / "b.db")))
        await retried.save_memory("User likes Python", user_id="u")
        await retried.save_memory("User likes Rust", user_id="u")
        results = await retried.search_memories("python", user_id="u", top_k=1)
        assert results[0].memory.content == "User likes Python"
        await retried.close()

        (backend,) = FakeBackend.instances
        assert backend.closed == 0


class _Closable:
    async def close(self) -> None:
        return None


# =============================================================================
# Embedding-space fingerprint
# =============================================================================


class TestEmbedderIdentity:
    @pytest.mark.parametrize(
        ("config", "identity"),
        [
            (
                MemoryConfig(embedder_model="sentence-transformers/all-MiniLM-L6-v2"),
                "sentence-transformers:all-MiniLM-L6-v2:384",
            ),
            (
                MemoryConfig(embedder_model="all-mpnet-base-v2", vector_dimension=768),
                "sentence-transformers:all-mpnet-base-v2:768",
            ),
            (
                MemoryConfig(embedder_backend=EmbedderBackend.ONNX),
                "sentence-transformers:all-MiniLM-L6-v2:384",
            ),
            (
                MemoryConfig(
                    embedder_backend=EmbedderBackend.OPENAI,
                    embedder_model="text-embedding-3-small",
                    openai_api_key="k",
                    vector_dimension=1536,
                ),
                "openai:text-embedding-3-small:1536",
            ),
            (
                MemoryConfig(
                    embedder_backend=EmbedderBackend.OLLAMA,
                    embedder_model="nomic-embed-text",
                    vector_dimension=768,
                ),
                "ollama:nomic-embed-text:768",
            ),
        ],
    )
    def test_builtin_identities(self, config: MemoryConfig, identity: str) -> None:
        assert embedder_identity(config) == identity

    def test_external_identities(self) -> None:
        assert embedder_identity(external(embedder_backend_name=f"{FAKE}:m")) == f"{FAKE}:m:3"
        assert embedder_identity(external(embedder_options={"model": "m"})) == f"{FAKE}:m:3"
        assert embedder_identity(external()) == f"{FAKE}::3"

    def test_injected_identities(self) -> None:
        mine = AsyncEmbedder()
        config = MemoryConfig(embedder=mine, vector_dimension=3)
        assert embedder_identity(config) == "custom:my-async-model:3"
        fn_config = MemoryConfig(embedder=lambda texts: texts, vector_dimension=3)
        assert embedder_identity(fn_config, _create_embedder(fn_config)) == "custom:callable:3"

    def test_onnx_and_local_minilm_interoperate(self) -> None:
        local = MemoryConfig(embedder_model="all-MiniLM-L6-v2")
        onnx = MemoryConfig(embedder_backend=EmbedderBackend.ONNX)
        assert embedder_identity(local) == embedder_identity(onnx)

    def test_first_open_records_and_later_opens_compare(self, tmp_path: Path) -> None:
        db = tmp_path / "memory.db"
        verify_embedder_identity(db, "a:m:3")
        assert read_embedder_identity(db) == "a:m:3"
        verify_embedder_identity(db, "a:m:3")
        with pytest.raises(EmbedderMismatchError) as excinfo:
            verify_embedder_identity(db, "b:m:3")
        error = excinfo.value
        assert isinstance(error, ValueError)
        assert (error.recorded, error.current) == ("a:m:3", "b:m:3")
        message = str(error)
        assert "'a:m:3'" in message and "'b:m:3'" in message
        assert "HEADROOM_EMBEDDER_" in message and "fresh memory directory" in message
        assert read_embedder_identity(db) == "a:m:3"

    def test_an_existing_database_without_a_record_adopts_the_identity(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "memory.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO memories VALUES ('m1')")
        assert read_embedder_identity(db) is None
        verify_embedder_identity(db, "a:m:3")
        assert read_embedder_identity(db) == "a:m:3"
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT id FROM memories").fetchall() == [("m1",)]

    def test_in_memory_databases_are_skipped(self) -> None:
        verify_embedder_identity(":memory:", "a:m:3")
        assert read_embedder_identity(":memory:") is None

    def test_disabled_check_writes_nothing_and_only_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = tmp_path / "memory.db"
        verify_embedder_identity(db, "a:m:3", enforce=False)
        assert not db.exists()
        verify_embedder_identity(db, "a:m:3")
        with caplog.at_level(logging.WARNING):
            verify_embedder_identity(db, "b:m:3", enforce=False)
        assert "'a:m:3'" in caplog.text and "'b:m:3'" in caplog.text
        assert read_embedder_identity(db) == "a:m:3"

    @pytest.mark.parametrize(
        ("value", "enabled"),
        [(None, True), ("1", True), ("", True), ("0", False), ("false", False), ("OFF", False)],
    )
    def test_env_switch(self, value: str | None, enabled: bool) -> None:
        env = {} if value is None else {IDENTITY_CHECK_ENV: value}
        assert identity_check_enabled(env) is enabled


class TestLocalBackendIdentity:
    def _external(self, db: Path, **overrides: Any) -> LocalBackendConfig:
        settings: dict[str, Any] = {
            "db_path": str(db),
            "embedder_backend": "external",
            "embedder_backend_name": f"{FAKE}:written-model",
            "vector_dimension": 3,
            "graph_persist": False,
        }
        settings.update(overrides)
        return LocalBackendConfig(**settings)

    async def test_a_database_refuses_another_embedder_before_any_write(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "memory.db"
        writer = LocalBackend(self._external(db))
        await writer.save_memory("User likes Python", user_id="u")
        await writer.close()
        assert read_embedder_identity(db) == f"{FAKE}:written-model:3"

        other = LocalBackend(self._external(db, embedder_backend_name=f"{FAKE}:other-model"))
        with pytest.raises(EmbedderMismatchError, match="other-model"):
            await other.save_memory("must not land", user_id="u")
        assert other._initialized is False
        with sqlite3.connect(db) as conn:
            contents = [row[0] for row in conn.execute("SELECT content FROM memories")]
        assert contents == ["User likes Python"]

    async def test_a_proxy_written_database_refuses_the_default_embedder(
        self, tmp_path: Path
    ) -> None:
        """The MCP server and sync default to ONNX MiniLM; a database the proxy
        filled through a custom embedder must not be opened with it."""
        db = tmp_path / "memory.db"
        proxy = LocalBackend(self._external(db))
        await proxy.save_memory("User likes Python", user_id="u")
        await proxy.close()

        mcp_default = LocalBackend(LocalBackendConfig(db_path=str(db), embedder_backend="onnx"))
        with pytest.raises(EmbedderMismatchError) as excinfo:
            await mcp_default._ensure_initialized()
        assert excinfo.value.recorded == f"{FAKE}:written-model:3"
        assert excinfo.value.current == "sentence-transformers:all-MiniLM-L6-v2:384"

    @pytest.mark.parametrize("via", ["config", "env"])
    async def test_the_escape_hatch_opens_it_anyway(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via: str
    ) -> None:
        db = tmp_path / "memory.db"
        writer = LocalBackend(self._external(db))
        await writer.save_memory("User likes Python", user_id="u")
        await writer.close()

        overrides: dict[str, Any] = {"embedder_backend_name": f"{FAKE}:other-model"}
        if via == "config":
            overrides["enforce_embedder_identity"] = False
        else:
            monkeypatch.setenv(IDENTITY_CHECK_ENV, "0")
        reader = LocalBackend(self._external(db, **overrides))
        await reader._ensure_initialized()
        assert reader._initialized is True
        await reader.close()
        assert read_embedder_identity(db) == f"{FAKE}:written-model:3"


# =============================================================================
# LocalBackendConfig
# =============================================================================


class TestLocalBackendConfig:
    @pytest.mark.parametrize(
        ("value", "member"),
        [
            ("local", EmbedderBackend.LOCAL),
            ("ONNX", EmbedderBackend.ONNX),
            (" OpenAI ", EmbedderBackend.OPENAI),
            ("ollama", EmbedderBackend.OLLAMA),
            ("External", EmbedderBackend.EXTERNAL),
            (EmbedderBackend.EXTERNAL, EmbedderBackend.EXTERNAL),
        ],
    )
    def test_embedder_backend_mapping(self, value: Any, member: EmbedderBackend) -> None:
        assert resolve_embedder_backend(value) is member
        LocalBackendConfig(embedder_backend=value, embedder_backend_name=FAKE)

    @pytest.mark.parametrize("value", ["sentence-transformers", "", None, 3])
    def test_an_unknown_embedder_backend_is_refused(self, value: Any) -> None:
        # It used to become "local" silently, swapping the model in use.
        with pytest.raises(ValueError, match="'local', 'onnx', 'openai', 'ollama', 'external'"):
            LocalBackendConfig(embedder_backend=value)

    def test_external_needs_a_valid_backend_name_at_construction(self) -> None:
        with pytest.raises(ValueError, match="embedder_backend_name is required"):
            LocalBackendConfig(embedder_backend="external")
        with pytest.raises(ValueError, match="Invalid backend name"):
            LocalBackendConfig(embedder_backend="external", embedder_backend_name="Bad Name")
        LocalBackendConfig(embedder_backend="external", embedder=AsyncEmbedder())

    async def test_new_fields_reach_memory_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, MemoryConfig] = {}

        async def fake_create(config: MemoryConfig) -> Any:
            seen["config"] = config
            raise RuntimeError("stop after capturing the config")

        monkeypatch.setattr("headroom.memory.HierarchicalMemory.create", fake_create)
        mine = AsyncEmbedder()
        backend = LocalBackend(
            LocalBackendConfig(
                db_path=str(tmp_path / "memory.db"),
                embedder_backend=EmbedderBackend.EXTERNAL,
                embedder_backend_name=f"{FAKE}:m",
                embedder_options={"base_url": "http://x/v1"},
                embedder=mine,
            )
        )
        with pytest.raises(RuntimeError, match="stop after"):
            await backend._ensure_initialized()
        config = seen["config"]
        assert config.embedder_backend is EmbedderBackend.EXTERNAL
        assert config.embedder_backend_name == f"{FAKE}:m"
        assert config.embedder_options == {"base_url": "http://x/v1"}
        assert config.embedder is mine

    def test_repr_hides_options_and_the_embedder(self) -> None:
        config = LocalBackendConfig(
            embedder_options={"api_key": "sk-secret-value"}, embedder=AsyncEmbedder()
        )
        assert "sk-secret-value" not in repr(config)
        assert "AsyncEmbedder" not in repr(config)
