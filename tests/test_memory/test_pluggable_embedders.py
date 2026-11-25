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
    clear_embedder_identity,
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


class Masked:
    """A credential holder whose repr hides its value, like pydantic's SecretStr."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "Masked('**********')"


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

    def test_options_whose_repr_masks_them_never_share_an_embedder(self) -> None:
        """Regression: the key used repr() for non-JSON options, and a masked
        credential's repr is the same for every value, so tenant B got tenant
        A's cached client and key."""
        first = _create_embedder(external(embedder_options={"api_key": Masked("tenant-a")}))
        second = _create_embedder(external(embedder_options={"api_key": Masked("tenant-b")}))
        assert repr(Masked("tenant-a")) == repr(Masked("tenant-b"))
        assert second is not first
        assert [b.options["api_key"].value for b in FakeBackend.instances] == [
            "tenant-a",
            "tenant-b",
        ]
        # Not cached, so each memory system owns (and closes) its own.
        assert _EMBEDDER_CACHE == {}
        assert not embedder_is_shared(external(), first)

    def test_a_pydantic_secret_is_not_cached(self) -> None:
        secret = pytest.importorskip("pydantic").SecretStr
        first = _create_embedder(external(embedder_options={"api_key": secret("a")}))
        second = _create_embedder(external(embedder_options={"api_key": secret("b")}))
        assert second is not first
        assert second.backend.options["api_key"].get_secret_value() == "b"

    async def test_an_uncached_embedder_is_owned_and_closed(self, tmp_path: Path) -> None:
        memory = await HierarchicalMemory.create(
            external(db_path=tmp_path / "m.db", embedder_options={"api_key": Masked("a")})
        )
        assert memory._owns_embedder is True
        await memory.close()
        assert FakeBackend.instances[0].closed == 1

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

    async def test_a_name_that_is_not_a_registry_name_is_never_shown(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: a plugin's self-chosen ``name`` (here an endpoint with a
        key) reached the WARNING line and the repr unvalidated."""

        class EndpointNamed(FakeBackend):
            name = "https://gw.example/v1?key=sk-live-SECRET"

            def close(self) -> None:
                raise RuntimeError("boom")

        embedder = BackendEmbedder(EndpointNamed(dimension=3), dimension=3)
        with caplog.at_level(logging.WARNING):
            await embedder.close()
        assert "RuntimeError" in caplog.text
        assert "EndpointNamed" in repr(embedder) and "EndpointNamed" in caplog.text
        assert "sk-live-SECRET" not in caplog.text
        assert "sk-live-SECRET" not in repr(embedder)
        # A registry-style name is still shown as is.
        assert f"backend={FAKE!r}" in repr(BackendEmbedder(FakeBackend(dimension=3), dimension=3))


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
        # Without the built embedder only the configured model is known.
        assert embedder_identity(external()) == f"{FAKE}::3"

    def test_external_identity_is_the_model_the_backend_serves(self) -> None:
        """Regression: an unnamed model was recorded as '', so two backend
        defaults of the same width shared one identity."""
        config = external()
        assert embedder_identity(config, _create_embedder(config)) == f"{FAKE}:fake-default:3"

    @pytest.mark.parametrize("options", [{}, {"model": "sentence-transformers/all-MiniLM-L6-v2"}])
    def test_the_sentence_transformers_backend_shares_the_minilm_identity(
        self, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
    ) -> None:
        """Regression: the same MiniLM through the sentence-transformers backend
        got another identity than ONNX and LOCAL, so it was refused."""
        from headroom.models.config import ML_MODEL_DEFAULTS

        monkeypatch.setattr(ML_MODEL_DEFAULTS, "sentence_transformer", "all-MiniLM-L6-v2")
        config = external(
            embedder_backend_name="sentence-transformers",
            embedder_options=options,
            vector_dimension=384,
        )
        onnx = embedder_identity(MemoryConfig(embedder_backend=EmbedderBackend.ONNX))
        assert embedder_identity(config, _create_embedder(config)) == onnx

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
        assert "clear_embedder_identity()" in message and f"{IDENTITY_CHECK_ENV}=0" in message
        assert error.stored_dimension is None
        assert read_embedder_identity(db) == "a:m:3"

    def test_a_wrong_record_can_be_cleared(self, tmp_path: Path) -> None:
        db = tmp_path / "memory.db"
        assert clear_embedder_identity(db) is None
        assert not db.exists()
        verify_embedder_identity(db, "a:m:3")
        assert clear_embedder_identity(db) == "a:m:3"
        assert read_embedder_identity(db) is None
        assert clear_embedder_identity(db) is None
        verify_embedder_identity(db, "b:m:3")
        assert read_embedder_identity(db) == "b:m:3"

    @staticmethod
    def _old_database(db: Path, width: int | None) -> None:
        """A memory store as releases before identities wrote it: no record."""
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, embedding BLOB)")
            blob = None if width is None else np.ones(width, dtype=np.float32).tobytes()
            conn.execute("INSERT INTO memories VALUES ('m1', ?)", (blob,))

    def test_stored_vectors_of_another_width_are_refused_and_not_recorded(
        self, tmp_path: Path
    ) -> None:
        """Regression: the first opener's identity was recorded over stored
        vectors it could not have written, locking out the embedder that did."""
        db = tmp_path / "memory.db"
        self._old_database(db, width=16)
        with pytest.raises(EmbedderMismatchError) as excinfo:
            verify_embedder_identity(db, "a:m:3")
        error = excinfo.value
        assert error.stored_dimension == 16
        assert (error.recorded, error.current) == ("unrecorded::16", "a:m:3")
        assert "16-wide vectors" in str(error) and "Nothing was recorded" in str(error)
        assert read_embedder_identity(db) is None
        verify_embedder_identity(db, "writer:m:16")
        assert read_embedder_identity(db) == "writer:m:16"

    @pytest.mark.parametrize("width", [3, None])
    def test_stored_vectors_of_the_same_width_or_none_are_adopted(
        self, tmp_path: Path, width: int | None
    ) -> None:
        db = tmp_path / "memory.db"
        self._old_database(db, width=width)
        verify_embedder_identity(db, "a:m:3")
        assert read_embedder_identity(db) == "a:m:3"

    def test_a_disabled_check_warns_about_stored_vectors_too(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = tmp_path / "memory.db"
        self._old_database(db, width=16)
        with caplog.at_level(logging.WARNING):
            verify_embedder_identity(db, "a:m:3", enforce=False)
        assert "'unrecorded::16'" in caplog.text
        assert read_embedder_identity(db) is None

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

    @staticmethod
    def _from_env(db: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> LocalBackend:
        """The backend a proxy, MCP server or CLI builds from HEADROOM_EMBEDDER_*."""
        from headroom.memory.embedder_env import local_backend_embedder_kwargs_from_env

        for name, value in env.items():
            monkeypatch.setenv(name, value)
        kwargs = local_backend_embedder_kwargs_from_env()
        # Each open stands for another process: nothing is shared through the cache.
        _reset_embedder_cache_for_tests()
        return LocalBackend(LocalBackendConfig(db_path=str(db), graph_persist=False, **kwargs))

    async def test_two_sentence_transformer_defaults_are_two_identities(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: with no model named, both processes recorded
        'sentence-transformers::384' and shared one database, although
        HEADROOM_SENTENCE_TRANSFORMER made them serve different models."""
        from headroom.models.config import ML_MODEL_DEFAULTS

        db = tmp_path / "memory.db"
        env = {"HEADROOM_EMBEDDER_PROVIDER": "sentence-transformers"}
        env["HEADROOM_EMBEDDER_DIMENSION"] = "384"
        # HEADROOM_SENTENCE_TRANSFORMER is read once, at import.
        monkeypatch.setattr(ML_MODEL_DEFAULTS, "sentence_transformer", "all-MiniLM-L6-v2")
        first = self._from_env(db, monkeypatch, **env)
        await first._ensure_initialized()
        await first.close()
        assert read_embedder_identity(db) == "sentence-transformers:all-MiniLM-L6-v2:384"

        monkeypatch.setattr(ML_MODEL_DEFAULTS, "sentence_transformer", "example-org/other-embedder")
        second = self._from_env(db, monkeypatch, **env)
        with pytest.raises(EmbedderMismatchError) as excinfo:
            await second._ensure_initialized()
        assert excinfo.value.current == "sentence-transformers:example-org/other-embedder:384"

    @pytest.mark.parametrize("model", [None, "sentence-transformers/all-MiniLM-L6-v2"])
    async def test_an_onnx_database_opens_with_the_sentence_transformers_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str | None
    ) -> None:
        from headroom.models.config import ML_MODEL_DEFAULTS

        monkeypatch.setattr(ML_MODEL_DEFAULTS, "sentence_transformer", "all-MiniLM-L6-v2")
        db = tmp_path / "memory.db"
        onnx = LocalBackend(
            LocalBackendConfig(db_path=str(db), embedder_backend="onnx", graph_persist=False)
        )
        await onnx._ensure_initialized()
        await onnx.close()

        env = {"HEADROOM_EMBEDDER_PROVIDER": "sentence-transformers"}
        env["HEADROOM_EMBEDDER_DIMENSION"] = "384"
        if model is not None:
            env["HEADROOM_EMBEDDER_MODEL"] = model
        backend = self._from_env(db, monkeypatch, **env)
        await backend._ensure_initialized()
        assert backend._initialized is True
        await backend.close()
        assert read_embedder_identity(db) == "sentence-transformers:all-MiniLM-L6-v2:384"

    async def test_an_unrecorded_database_is_not_claimed_by_a_default_opener(
        self, tmp_path: Path
    ) -> None:
        """Regression: a database written before identities existed (here by a
        16-wide embedder) was stamped by whichever process opened it first,
        e.g. the MCP server on its ONNX default, locking out its writer."""
        db = tmp_path / "memory.db"
        writer_config = self._external(db, vector_dimension=16)
        # As older releases wrote it: no record at all.
        old = LocalBackend(replace(writer_config, enforce_embedder_identity=False))
        await old.save_memory("User likes Python", user_id="u")
        await old.close()
        with sqlite3.connect(db) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "headroom_embedder_identity" not in tables

        mcp_default = LocalBackend(
            LocalBackendConfig(db_path=str(db), embedder_backend="onnx", graph_persist=False)
        )
        with pytest.raises(EmbedderMismatchError) as excinfo:
            await mcp_default._ensure_initialized()
        assert excinfo.value.stored_dimension == 16
        assert read_embedder_identity(db) is None

        writer = LocalBackend(writer_config)
        results = await writer.search_memories("python", user_id="u", top_k=1)
        assert results[0].memory.content == "User likes Python"
        await writer.close()
        assert read_embedder_identity(db) == f"{FAKE}:written-model:16"


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
