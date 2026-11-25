"""Relevance scoring on pluggable embedding backends.

Covers the built-in ``fastembed`` backend, ``EmbeddingScorer(backend=...)`` and
its circuit breaker, the hybrid scorer's fallback when its embedding half
fails, ``create_scorer(embedding_backend=...)``,
``RelevanceScorerConfig.embedding_backend`` and the resolver the router and
the proxy share, and one end-to-end run against a loopback OpenAI-compatible
server. fastembed itself is stubbed: no test here downloads a model.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from headroom.config import RelevanceScorerConfig
from headroom.models.backends import (
    EMBEDDING_BACKENDS,
    BackendLoadError,
    CallableEmbeddingBackend,
    EmbeddingBackendCallError,
    EmbeddingBackendResponseError,
    UnknownBackendError,
    create_embedding_backend,
    embed_texts,
    register_embedding_backend,
)
from headroom.offline import OFFLINE_ENV, OfflineEgressBlocked
from headroom.relevance import (
    BM25Scorer,
    EmbeddingBackendCircuitOpenError,
    EmbeddingScorer,
    FastEmbedBackend,
    HybridScorer,
    create_scorer,
    embedding,
)
from headroom.relevance.backend_config import (
    RELEVANCE_OPENAI_MAX_RETRIES,
    RELEVANCE_OPENAI_TIMEOUT_SECONDS,
    describe_relevance_embedding_backend,
    resolve_relevance_embedding_backend,
)
from headroom.relevance.base import RelevanceScore, RelevanceScorer

# Words the fake embedders count, one dimension each, plus a constant so no
# vector is zero. "error" and "timeout" share an axis on purpose: a query for
# one finds records with the other, which BM25 alone cannot do.
_AXES = (("error", "timeout"), ("widget",), ("render",))


def _keyword_vectors(texts: list[str]) -> list[list[float]]:
    rows = []
    for text in texts:
        lower = text.lower()
        rows.append([float(sum(lower.count(word) for word in axis)) for axis in _AXES] + [0.01])
    return rows


@pytest.fixture(autouse=True)
def _clean_embedding_registry() -> Iterator[None]:
    EMBEDDING_BACKENDS._reset_for_tests()
    yield
    EMBEDDING_BACKENDS._reset_for_tests()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Drive the circuit breaker's clock: ``clock[0]`` is "now"."""
    now = [1000.0]
    monkeypatch.setattr(embedding, "_monotonic", lambda: now[0])
    return now


class _Recorder:
    """A backend function that records every batch and can be told to fail."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_with: BaseException | None = None

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail_with is not None:
            raise self.fail_with
        return _keyword_vectors(texts)


# =============================================================================
# FastEmbedBackend
# =============================================================================


class _FakeTextEmbedding:
    """Stands in for a loaded fastembed ``TextEmbedding``."""

    def __init__(self, width: int = 384) -> None:
        self.width = width
        self.embed_calls: list[tuple[list[str], dict[str, Any]]] = []

    def embed(self, texts: list[str], **kwargs: Any) -> Iterator[np.ndarray]:
        self.embed_calls.append((list(texts), kwargs))
        for text in texts:
            # Deterministic, text-dependent and never zero.
            yield np.array(
                [1.0 + (len(text) * (j + 1)) % 11 for j in range(self.width)], dtype=np.float32
            )


@pytest.fixture
def fastembed_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import fastembed`` succeed; the loaders are stubbed per test."""
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=object))


@pytest.fixture
def no_arena_shrink(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    shrunk: list[Any] = []
    monkeypatch.setattr(embedding, "_shrink_arena_after_each_run", shrunk.append)
    return shrunk


class TestFastEmbedBackend:
    def test_registry_builds_the_default_model_lazily(
        self, fastembed_installed: None, no_arena_shrink: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _FakeTextEmbedding()
        loads: list[str] = []
        monkeypatch.setattr(embedding, "_load_default_model", lambda: loads.append("d") or model)
        monkeypatch.setattr(
            embedding, "_load_text_embedding", lambda kwargs: pytest.fail("not the default")
        )

        backend = create_embedding_backend("fastembed")

        assert isinstance(backend, FastEmbedBackend)
        assert backend.name == "fastembed"
        assert backend.model_name == embedding.DEFAULT_MODEL_NAME
        assert backend.dimension == embedding.DEFAULT_MODEL_DIMENSION == 384
        assert loads == [], "building the backend must not load the model"

        out = embed_texts(backend, ["alpha", "  ", "beta"])

        assert out.shape == (3, 384) and out.dtype == np.float32
        assert not out[1].any(), "blank texts come back as zero rows"
        assert loads == ["d"]
        assert model.embed_calls == [
            (["alpha", "beta"], {"batch_size": embedding.EMBED_BATCH_SIZE})
        ]
        assert no_arena_shrink == [model]

        embed_texts(backend, ["gamma"])
        assert loads == ["d"], "the model is loaded once"
        assert no_arena_shrink == [model], "the arena hook is installed once"

    def test_the_default_model_spelled_out_uses_the_pinned_loader(
        self, fastembed_installed: None, no_arena_shrink: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loads: list[str] = []
        monkeypatch.setattr(
            embedding, "_load_default_model", lambda: loads.append("d") or _FakeTextEmbedding()
        )
        backend = create_embedding_backend(f"fastembed:{embedding.DEFAULT_MODEL_NAME}")
        backend.embed(["x"])
        assert loads == ["d"]

    def test_other_models_use_fastembed_resolution_and_learn_their_width(
        self, fastembed_installed: None, no_arena_shrink: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict[str, Any]] = []
        model = _FakeTextEmbedding(width=8)
        monkeypatch.setattr(
            embedding, "_load_text_embedding", lambda kwargs: seen.append(kwargs) or model
        )
        monkeypatch.setattr(embedding, "_load_default_model", lambda: pytest.fail("not default"))

        backend = create_embedding_backend("fastembed:intfloat/e5-small-v2")

        assert backend.model_name == "intfloat/e5-small-v2"
        assert backend.dimension is None
        assert embed_texts(backend, ["a", "b"]).shape == (2, 8)
        assert seen == [{"model_name": "intfloat/e5-small-v2"}]
        assert backend.dimension == 8

        model.width = 4  # a model that changed width under us is refused
        with pytest.raises(EmbeddingBackendResponseError, match="4-dimensional"):
            embed_texts(backend, ["c"])

    def test_a_declared_dimension_is_checked(
        self, fastembed_installed: None, no_arena_shrink: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            embedding, "_load_text_embedding", lambda kwargs: _FakeTextEmbedding(width=8)
        )
        with pytest.raises(EmbeddingBackendResponseError, match="8-dimensional"):
            embed_texts(FastEmbedBackend("some/model", dimension=16), ["a"])
        assert embed_texts(FastEmbedBackend("some/model", dimension=8), ["a"]).shape == (1, 8)

    def test_the_default_model_width_cannot_be_overridden(self, fastembed_installed: None) -> None:
        assert FastEmbedBackend(dimension=384).dimension == 384
        with pytest.raises(ValueError, match="384-dimensional"):
            FastEmbedBackend(dimension=768)
        with pytest.raises(BackendLoadError, match="384-dimensional"):
            create_embedding_backend("fastembed", dimension=768)
        with pytest.raises(ValueError, match="positive integer"):
            FastEmbedBackend("some/model", dimension=0)

    def test_missing_fastembed_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(EmbeddingScorer, "is_available", classmethod(lambda cls: False))
        with pytest.raises(BackendLoadError, match="requires fastembed"):
            create_embedding_backend("fastembed")

    def test_concurrent_first_calls_load_the_model_once(
        self, fastembed_installed: None, no_arena_shrink: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loads: list[int] = []
        gate = threading.Event()

        def slow_load() -> _FakeTextEmbedding:
            loads.append(1)
            gate.wait(5)
            return _FakeTextEmbedding()

        monkeypatch.setattr(embedding, "_load_default_model", slow_load)
        backend = FastEmbedBackend()
        threads = [threading.Thread(target=backend.embed, args=(["x"],)) for _ in range(6)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        gate.set()
        for thread in threads:
            thread.join(5)
        assert loads == [1]

    def test_an_offline_cold_cache_is_a_model_unavailable_error_with_its_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real loader, refused by the air gap: the caller hears which model
        is unavailable and how to seed it, through embed_texts, not just a class."""

        class _ColdTextEmbedding:
            def __init__(self, **kwargs: object) -> None:
                raise OSError("not cached locally")

        monkeypatch.setitem(
            sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_ColdTextEmbedding)
        )
        monkeypatch.setenv(OFFLINE_ENV, "1")
        backend = FastEmbedBackend("some/model")

        with pytest.raises(BackendLoadError) as excinfo:
            embed_texts(backend, ["x"])

        assert isinstance(excinfo.value, RuntimeError)
        assert "'some/model' is unavailable" in str(excinfo.value)
        assert OFFLINE_ENV in str(excinfo.value)
        assert "Pre-seed the fastembed cache" in str(excinfo.value)

    def test_scores_match_the_fastembed_scorer_for_the_same_model(
        self, fastembed_installed: None, no_arena_shrink: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting "fastembed" explicitly is the default scorer, not another model."""
        monkeypatch.setattr(embedding, "_load_default_model", lambda: _FakeTextEmbedding())
        items = ["first record", "second, longer record", "third"]

        via_backend = EmbeddingScorer(backend="fastembed").score_batch(items, "query")
        built_in = EmbeddingScorer().score_batch(items, "query")

        assert len({s.score for s in built_in}) == 3, "the fixture must tell records apart"
        assert [s.score for s in via_backend] == pytest.approx([s.score for s in built_in])


# =============================================================================
# EmbeddingScorer(backend=...)
# =============================================================================


class TestEmbeddingScorerWithABackend:
    def test_scores_through_the_backend(self) -> None:
        fn = _Recorder()
        backend = CallableEmbeddingBackend(fn, model_name="kw")
        scorer = EmbeddingScorer(backend=backend)

        assert scorer.backend is backend
        assert scorer.model_name == "kw"
        scores = scorer.score_batch(
            ["connection timeout on db", "render the widget", "error: disk full"], "show errors"
        )

        assert scores[0].score > 0.9 and scores[2].score > 0.9
        assert scores[1].score < 0.1
        assert fn.calls == [
            ["connection timeout on db", "render the widget", "error: disk full", "show errors"]
        ]
        single = scorer.score("timeout talking to the api", "any errors?")
        assert single.score > 0.9
        assert single.reason.startswith("Embedding: semantic similarity")

    def test_accepts_a_function_or_a_registry_spec(self) -> None:
        assert EmbeddingScorer(backend=_keyword_vectors).score("error", "timeout").score > 0.9

        built: list[dict[str, Any]] = []

        def factory(**options: Any) -> CallableEmbeddingBackend:
            built.append(options)
            return CallableEmbeddingBackend(_keyword_vectors, model=options.get("model"))

        register_embedding_backend("kw", factory)
        scorer = EmbeddingScorer(backend="kw:v1")
        assert built == [{"model": "v1"}]
        assert scorer.model_name == "v1"

    def test_a_bad_spec_fails_at_construction(self) -> None:
        with pytest.raises(UnknownBackendError):
            EmbeddingScorer(backend="no-such-backend")

    def test_empty_inputs_never_reach_the_backend(self) -> None:
        fn = _Recorder()
        scorer = EmbeddingScorer(backend=fn)

        assert scorer.score_batch([], "q") == []
        assert scorer.score_batch(["a"], "")[0].reason == "Embedding: empty context"
        assert scorer.score("", "q").reason == "Embedding: empty input"
        assert fn.calls == []

        scores = scorer.score_batch(["   ", "error here"], "timeout")
        assert scores[0].score == 0.0, "a blank record scores zero"
        assert fn.calls == [["error here", "timeout"]], "blank texts are not sent"

        assert scorer.score_batch(["error"], "   ")[0].score == 0.0

    def test_model_name_and_backend_together_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            EmbeddingScorer("BAAI/bge-small-en-v1.5", backend=_keyword_vectors)

    @pytest.mark.parametrize("cooldown", [0, -1, float("inf"), float("nan"), True, "30"])
    def test_the_cooldown_must_be_a_positive_number(self, cooldown: Any) -> None:
        with pytest.raises(ValueError, match="backend_cooldown_seconds"):
            EmbeddingScorer(backend=_keyword_vectors, backend_cooldown_seconds=cooldown)

    def test_without_a_backend_nothing_changes(self) -> None:
        scorer = EmbeddingScorer()
        assert scorer.backend is None
        assert scorer.model_name == embedding.DEFAULT_MODEL_NAME

    def test_the_fastembed_model_is_loaded_once_under_concurrency(
        self, fastembed_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loads: list[int] = []
        gate = threading.Event()

        def slow_load() -> object:
            loads.append(1)
            gate.wait(5)
            return object()

        monkeypatch.setattr(embedding, "_load_default_model", slow_load)
        scorer = EmbeddingScorer()
        threads = [threading.Thread(target=scorer._get_model) for _ in range(6)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        gate.set()
        for thread in threads:
            thread.join(5)
        assert loads == [1]


class TestCircuitBreaker:
    def test_a_failure_opens_it_and_calls_fail_fast_until_the_cooldown_ends(
        self, clock: list[float], caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.relevance.embedding")
        fn = _Recorder()
        scorer = EmbeddingScorer(
            backend=CallableEmbeddingBackend(fn, name="kw-remote"), backend_cooldown_seconds=30
        )
        fn.fail_with = ConnectionError("POST https://user:sk-secret@host/v1 refused")

        with pytest.raises(EmbeddingBackendCallError):
            scorer.score_batch(["a"], "q")
        assert len(fn.calls) == 1

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "'kw-remote'" in message and "EmbeddingBackendCallError" in message
        assert "30s" in message
        assert "sk-secret" not in caplog.text, "a backend's error text is never logged"

        # Inside the cooldown: refused without contacting the backend.
        clock[0] += 29
        with pytest.raises(EmbeddingBackendCircuitOpenError, match="retrying in 1s"):
            scorer.score_batch(["a"], "q")
        assert len(fn.calls) == 1

        # After it: exactly one call probes. Still failing reopens it quietly.
        clock[0] += 2
        with pytest.raises(EmbeddingBackendCallError):
            scorer.score_batch(["a"], "q")
        assert len(fn.calls) == 2
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1, (
            "one WARNING per opening, not per failed probe"
        )
        with pytest.raises(EmbeddingBackendCircuitOpenError):
            scorer.score_batch(["a"], "q")
        assert len(fn.calls) == 2

        # A successful probe closes it and says so.
        clock[0] += 31
        fn.fail_with = None
        assert scorer.score_batch(["error"], "timeout")[0].score > 0.9
        assert any("recovered" in r.getMessage() for r in caplog.records)
        assert scorer.score_batch(["error"], "timeout")[0].score > 0.9
        assert len(fn.calls) == 4

        # A new outage is a new opening, so it warns again.
        fn.fail_with = ConnectionError("down again")
        with pytest.raises(EmbeddingBackendCallError):
            scorer.score_batch(["a"], "q")
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2

    def test_only_one_call_probes_while_the_others_keep_failing_fast(
        self, clock: list[float]
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls: list[int] = []
        failing = [True]

        def backend_fn(texts: list[str]) -> list[list[float]]:
            calls.append(1)
            if failing[0]:
                raise ConnectionError("down")
            entered.set()
            release.wait(5)
            return _keyword_vectors(texts)

        scorer = EmbeddingScorer(backend=backend_fn, backend_cooldown_seconds=10)
        with pytest.raises(EmbeddingBackendCallError):
            scorer.score_batch(["a"], "q")
        clock[0] += 11
        failing[0] = False

        probe = threading.Thread(target=scorer.score_batch, args=(["a"], "q"))
        probe.start()
        assert entered.wait(5)
        with pytest.raises(EmbeddingBackendCircuitOpenError, match="a retry is in flight"):
            scorer.score_batch(["a"], "q")
        release.set()
        probe.join(5)
        assert len(calls) == 2
        scorer.score_batch(["a"], "q")
        assert len(calls) == 3, "the probe's success closed the breaker"

    def test_an_air_gap_refusal_propagates_and_is_not_a_backend_failure(
        self, clock: list[float]
    ) -> None:
        fn = _Recorder()
        scorer = EmbeddingScorer(backend=fn, backend_cooldown_seconds=10)
        fn.fail_with = OfflineEgressBlocked("relevance embeddings", "api.example.com")

        with pytest.raises(OfflineEgressBlocked):
            scorer.score_batch(["a"], "q")
        with pytest.raises(OfflineEgressBlocked):
            scorer.score_batch(["a"], "q")
        assert len(fn.calls) == 2, "a refusal does not open the breaker"

    def test_a_refused_probe_gives_its_slot_back(self, clock: list[float]) -> None:
        fn = _Recorder()
        scorer = EmbeddingScorer(backend=fn, backend_cooldown_seconds=10)
        fn.fail_with = ConnectionError("down")
        with pytest.raises(EmbeddingBackendCallError):
            scorer.score_batch(["a"], "q")

        clock[0] += 11
        fn.fail_with = OfflineEgressBlocked("relevance embeddings", "api.example.com")
        with pytest.raises(OfflineEgressBlocked):
            scorer.score_batch(["a"], "q")

        fn.fail_with = None
        assert scorer.score_batch(["error"], "timeout")[0].score > 0.9, (
            "a probe ended by a BaseException must not wedge the breaker half-open"
        )

    def test_bad_responses_open_it_too(self, clock: list[float]) -> None:
        calls: list[int] = []

        def wrong_count(texts: list[str]) -> list[list[float]]:
            calls.append(1)
            return [[1.0, 0.0]]

        scorer = EmbeddingScorer(backend=wrong_count)
        with pytest.raises(EmbeddingBackendResponseError):
            scorer.score_batch(["a", "b"], "q")
        with pytest.raises(EmbeddingBackendCircuitOpenError):
            scorer.score_batch(["a", "b"], "q")
        assert calls == [1]


# =============================================================================
# HybridScorer fallback
# =============================================================================


class _FailingEmbeddingScorer(RelevanceScorer):
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def score(self, item: str, context: str) -> RelevanceScore:
        self.calls += 1
        raise self.exc

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        self.calls += 1
        raise self.exc


def _bm25_only_hybrid() -> HybridScorer:
    scorer = HybridScorer(bm25_scorer=BM25Scorer())
    scorer.embedding = None
    scorer._embedding_available = False
    return scorer


class TestHybridFallback:
    ITEMS = ['{"id": 1, "user": "alice"}', '{"id": 2, "user": "bob"}', "unrelated log line"]
    QUERY = "find user alice"

    def test_a_failing_backend_call_gets_the_no_embedding_result(
        self, caplog: pytest.LogCaptureFixture, clock: list[float]
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="headroom.relevance")
        fn = _Recorder()
        fn.fail_with = RuntimeError("upstream said sk-secret")
        hybrid = HybridScorer(
            bm25_scorer=BM25Scorer(), embedding_scorer=EmbeddingScorer(backend=fn)
        )
        expected = _bm25_only_hybrid()

        assert hybrid.score_batch(self.ITEMS, self.QUERY) == expected.score_batch(
            self.ITEMS, self.QUERY
        )
        # The breaker is open now: these fail without calling the backend.
        assert hybrid.score(self.ITEMS[0], self.QUERY) == expected.score(self.ITEMS[0], self.QUERY)
        hybrid.score_batch(self.ITEMS, self.QUERY)

        assert len(fn.calls) == 1
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "headroom.relevance.hybrid"
        ]
        assert len(warnings) == 1, "warned once, not per call"
        assert "EmbeddingBackendCallError" in warnings[0].getMessage()
        assert "sk-secret" not in caplog.text
        assert hybrid.has_embedding_support() is True

    def test_a_custom_embedding_scorers_failure_still_reaches_the_caller(self) -> None:
        failing = _FailingEmbeddingScorer(RuntimeError("custom scorer failed"))
        hybrid = HybridScorer(bm25_scorer=BM25Scorer(), embedding_scorer=failing)
        with pytest.raises(RuntimeError, match="custom scorer failed"):
            hybrid.score_batch(self.ITEMS, self.QUERY)
        with pytest.raises(RuntimeError, match="custom scorer failed"):
            hybrid.score(self.ITEMS[0], self.QUERY)
        assert failing.calls == 2

    @pytest.mark.parametrize(
        "failure",
        [
            RuntimeError("model download failed"),
            OfflineEgressBlocked("fastembed weight download", "huggingface.co"),
        ],
        ids=["load-error", "air-gap"],
    )
    def test_the_default_fastembed_scorers_failure_still_reaches_the_caller(
        self,
        fastembed_installed: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        failure: BaseException,
    ) -> None:
        """Nothing configured: the default scorer raises exactly as before backends existed."""
        caplog.set_level(logging.WARNING, logger="headroom.relevance")
        loads: list[str] = []

        def failing_load(model_name: str) -> Any:
            loads.append(model_name)
            raise failure

        monkeypatch.setattr(embedding, "_load_fastembed_model", failing_load)
        hybrid = create_scorer("hybrid")
        assert isinstance(hybrid, HybridScorer)
        assert isinstance(hybrid.embedding, EmbeddingScorer)
        assert hybrid.embedding.backend is None

        with pytest.raises(RuntimeError):
            hybrid.score_batch(self.ITEMS, self.QUERY)
        with pytest.raises(RuntimeError):
            hybrid.score(self.ITEMS[0], self.QUERY)
        assert len(loads) == 2
        assert not caplog.records, "a propagated failure is the caller's to report"

    def test_an_air_gap_refusal_is_not_degraded(self) -> None:
        failing = _FailingEmbeddingScorer(OfflineEgressBlocked("relevance", "api.example.com"))
        hybrid = HybridScorer(bm25_scorer=BM25Scorer(), embedding_scorer=failing)
        with pytest.raises(OfflineEgressBlocked):
            hybrid.score_batch(self.ITEMS, self.QUERY)
        with pytest.raises(OfflineEgressBlocked):
            hybrid.score(self.ITEMS[0], self.QUERY)

    def test_a_dead_backend_costs_one_call_per_cooldown(self, clock: list[float]) -> None:
        fn = _Recorder()
        fn.fail_with = ConnectionError("down")
        hybrid = create_scorer("hybrid", embedding_backend=fn)
        expected = _bm25_only_hybrid().score_batch(self.ITEMS, self.QUERY)

        for _ in range(5):
            assert hybrid.score_batch(self.ITEMS, self.QUERY) == expected
        assert len(fn.calls) == 1

        clock[0] += embedding.BACKEND_COOLDOWN_SECONDS + 1
        fn.fail_with = None
        fused = hybrid.score_batch(self.ITEMS, self.QUERY)
        assert len(fn.calls) == 2
        assert all(score.reason.startswith("Hybrid (α=") for score in fused)


# =============================================================================
# create_scorer(embedding_backend=...)
# =============================================================================


class TestCreateScorer:
    def test_embedding_tier_needs_no_fastembed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(EmbeddingScorer, "is_available", classmethod(lambda cls: False))
        scorer = create_scorer("embedding", embedding_backend=_keyword_vectors)
        assert isinstance(scorer, EmbeddingScorer)
        assert scorer.backend is not None
        assert scorer.score("error", "timeout").score > 0.9

    def test_hybrid_tier_wraps_a_backend_scorer_and_forwards_its_options(self) -> None:
        scorer = create_scorer("HYBRID", embedding_backend=_keyword_vectors, alpha=0.7)
        assert isinstance(scorer, HybridScorer)
        assert isinstance(scorer.embedding, EmbeddingScorer)
        assert scorer.embedding.backend is not None
        assert scorer.base_alpha == 0.7
        assert scorer.has_embedding_support()

    def test_the_default_tier_is_hybrid(self) -> None:
        assert isinstance(create_scorer(embedding_backend=_keyword_vectors), HybridScorer)

    def test_bm25_with_a_backend_is_refused(self) -> None:
        with pytest.raises(ValueError, match="bm25"):
            create_scorer("bm25", embedding_backend=_keyword_vectors)

    def test_a_backend_and_an_embedding_scorer_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            create_scorer(
                "hybrid",
                embedding_backend=_keyword_vectors,
                embedding_scorer=EmbeddingScorer(backend=_keyword_vectors),
            )

    def test_an_unknown_tier_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown scorer tier"):
            create_scorer("semantic", embedding_backend=_keyword_vectors)

    def test_without_a_backend_the_fastembed_requirement_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(EmbeddingScorer, "is_available", classmethod(lambda cls: False))
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            create_scorer("embedding")


# =============================================================================
# RelevanceScorerConfig.embedding_backend
# =============================================================================


class TestRelevanceScorerConfig:
    def test_the_default_is_none(self) -> None:
        config = RelevanceScorerConfig()
        assert config.embedding_backend is None
        assert asdict(config)["embedding_backend"] is None

    @pytest.mark.parametrize(
        "value", ["openai", "openai:text-embedding-3-small", "my-plugin:a:b", "env", " ENV "]
    )
    def test_valid_values(self, value: str) -> None:
        assert RelevanceScorerConfig(embedding_backend=value).embedding_backend == value

    @pytest.mark.parametrize("value", ["", "   ", "Not A Name", "env:some-model", ":model"])
    def test_malformed_values_are_refused(self, value: str) -> None:
        with pytest.raises(ValueError):
            RelevanceScorerConfig(embedding_backend=value)

    def test_an_instance_is_refused_with_the_alternative_named(self) -> None:
        with pytest.raises(TypeError, match="create_scorer"):
            RelevanceScorerConfig(embedding_backend=CallableEmbeddingBackend(_keyword_vectors))  # type: ignore[arg-type]

    def test_a_named_backend_needs_an_embedding_tier(self) -> None:
        with pytest.raises(ValueError, match="bm25"):
            RelevanceScorerConfig(tier="bm25", embedding_backend="openai")
        # "env" may ride along on any tier; bm25 never reads it.
        assert RelevanceScorerConfig(tier="bm25", embedding_backend="env").tier == "bm25"

    def test_the_names_are_not_resolved_at_config_time(self) -> None:
        """Plugins may not be importable where the config is built."""
        assert RelevanceScorerConfig(embedding_backend="not-installed").embedding_backend

    def test_the_router_namespace_is_stable_and_tracks_the_backend(self) -> None:
        code = (
            "from headroom.config import RelevanceScorerConfig;"
            "from headroom.transforms.content_router import ContentRouterConfig, _config_namespace;"
            "print(_config_namespace(ContentRouterConfig("
            "relevance=RelevanceScorerConfig(embedding_backend='openai:text-embedding-3-small'))))"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("HEADROOM_")}
        seen = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={**env, "PYTHONHASHSEED": seed},
            ).stdout.strip()
            for seed in ("1", "2")
        }
        assert len(seen) == 1

        from headroom.transforms.content_router import ContentRouterConfig, _config_namespace

        default = _config_namespace(ContentRouterConfig())
        configured = _config_namespace(
            ContentRouterConfig(relevance=RelevanceScorerConfig(embedding_backend="openai"))
        )
        assert default != configured, "a different scorer must not share cached results"


# =============================================================================
# resolve_relevance_embedding_backend
# =============================================================================


@pytest.fixture
def fake_factory() -> list[dict[str, Any]]:
    built: list[dict[str, Any]] = []

    def factory(**options: Any) -> CallableEmbeddingBackend:
        built.append(dict(options))
        return CallableEmbeddingBackend(_keyword_vectors, name="fake-rel", model=options["model"])

    register_embedding_backend("fake-rel", factory)
    return built


class TestResolveRelevanceEmbeddingBackend:
    def test_nothing_selected(self) -> None:
        assert resolve_relevance_embedding_backend(None) is None
        assert resolve_relevance_embedding_backend("env", env={}) is None

    def test_a_spec_is_built_with_no_other_options(
        self, fake_factory: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A spec never consults the environment.
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "nope")
        backend = resolve_relevance_embedding_backend("fake-rel:m1")
        assert backend is not None and backend.model_name == "m1"
        assert fake_factory == [{"model": "m1"}]

    def test_env_reads_the_relevance_family(self, fake_factory: list[dict[str, Any]]) -> None:
        env = {
            "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "fake-rel",
            "HEADROOM_RELEVANCE_EMBEDDER_MODEL": "m2",
            "HEADROOM_RELEVANCE_EMBEDDER_OPTIONS": '{"region": "eu"}',
        }
        backend = resolve_relevance_embedding_backend("env", env=env)
        assert backend is not None and backend.model_name == "m2"
        assert fake_factory == [{"model": "m2", "region": "eu"}]

    def test_env_defaults_to_os_environ(
        self, fake_factory: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "fake-rel")
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_MODEL", "m3")
        backend = resolve_relevance_embedding_backend("env")
        assert backend is not None and backend.model_name == "m3"

    def test_an_unknown_provider_names_the_variable(self) -> None:
        env = {"HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "opanai"}
        with pytest.raises(UnknownBackendError, match="HEADROOM_RELEVANCE_EMBEDDER_PROVIDER"):
            resolve_relevance_embedding_backend("env", env=env)
        with pytest.raises(UnknownBackendError):
            resolve_relevance_embedding_backend("opanai")

    def test_the_builtin_openai_backend_gets_request_path_bounds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(OFFLINE_ENV, raising=False)
        env = {
            "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "openai",
            "HEADROOM_RELEVANCE_EMBEDDER_MODEL": "m",
            "HEADROOM_RELEVANCE_EMBEDDER_BASE_URL": "http://127.0.0.1:9/v1",
        }
        backend: Any = resolve_relevance_embedding_backend("env", env=env)
        assert (
            (backend.timeout, backend.max_retries)
            == (
                RELEVANCE_OPENAI_TIMEOUT_SECONDS,
                RELEVANCE_OPENAI_MAX_RETRIES,
            )
            == (5.0, 0)
        )

        explicit = {
            **env,
            "HEADROOM_RELEVANCE_EMBEDDER_OPTIONS": '{"timeout": 2.5, "max_retries": 1}',
        }
        backend = resolve_relevance_embedding_backend("env", env=explicit)
        assert (backend.timeout, backend.max_retries) == (2.5, 1)

    def test_a_replaced_openai_registration_gets_no_extra_options(self) -> None:
        built: list[dict[str, Any]] = []

        def factory(**options: Any) -> CallableEmbeddingBackend:
            built.append(options)
            return CallableEmbeddingBackend(_keyword_vectors)

        register_embedding_backend("openai", factory, replace=True)
        resolve_relevance_embedding_backend("openai:m")
        EmbeddingScorer(backend="openai:m")
        create_scorer("hybrid", embedding_backend="openai:m")
        assert built == [{"model": "m"}] * 3

    def test_a_spec_gets_the_same_bounds_through_every_entry_point(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(OFFLINE_ENV, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
        spec = "openai:m"
        hybrid: Any = create_scorer("hybrid", embedding_backend=spec)
        embedding_tier: Any = create_scorer("embedding", embedding_backend=spec)
        backends: list[Any] = [
            resolve_relevance_embedding_backend(spec),
            EmbeddingScorer(backend=spec).backend,
            embedding_tier.backend,
            hybrid.embedding.backend,
        ]
        assert [(b.timeout, b.max_retries) for b in backends] == [
            (RELEVANCE_OPENAI_TIMEOUT_SECONDS, RELEVANCE_OPENAI_MAX_RETRIES)
        ] * 4

        # An instance is the caller's choice and is used as given.
        instance = create_embedding_backend(spec)
        assert EmbeddingScorer(backend=instance).backend is instance

    def test_an_air_gapped_process_refuses_a_remote_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OFFLINE_ENV, "1")
        env = {
            "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "openai",
            "HEADROOM_RELEVANCE_EMBEDDER_MODEL": "m",
            "HEADROOM_RELEVANCE_EMBEDDER_BASE_URL": "http://127.0.0.1:9/v1",
        }
        with pytest.raises(OfflineEgressBlocked):
            resolve_relevance_embedding_backend("env", env=env)

    def test_labels_never_raise_and_carry_no_model(self) -> None:
        assert describe_relevance_embedding_backend(None) == "fastembed (default)"
        assert describe_relevance_embedding_backend("env") == "HEADROOM_RELEVANCE_EMBEDDER_*"
        assert describe_relevance_embedding_backend("openai:secret-model") == "openai"
        assert describe_relevance_embedding_backend("Not A Name") == "<invalid>"


# =============================================================================
# End to end: a loopback OpenAI-compatible server behind create_scorer
# =============================================================================


@pytest.fixture
def keyword_embedding_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, Any]]:
    """An OpenAI-compatible /v1/embeddings server that embeds with _keyword_vectors."""
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = SimpleNamespace(requests=[], status=200)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            if state.status == 200:
                vectors = _keyword_vectors(body["input"])
                payload: Any = {
                    "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
                }
            else:
                payload = {"error": {"message": "overloaded"}}
            raw = json.dumps(payload).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hybrid_scoring_against_a_loopback_openai_compatible_server(
    keyword_embedding_server: tuple[str, Any], clock: list[float]
) -> None:
    url, state = keyword_embedding_server
    backend = resolve_relevance_embedding_backend(
        "env",
        env={
            "HEADROOM_RELEVANCE_EMBEDDER_PROVIDER": "openai",
            "HEADROOM_RELEVANCE_EMBEDDER_MODEL": "keywords-v1",
            "HEADROOM_RELEVANCE_EMBEDDER_BASE_URL": url,
        },
    )
    scorer = create_scorer("hybrid", embedding_backend=backend, adaptive=False, alpha=0.5)
    records = [
        "2024-01-01 worker 3 request timeout after 30s",
        "2024-01-01 widget render complete",
        "",
        "2024-01-01 ERROR could not open socket",
    ]

    scores = scorer.score_batch(records, "show me the errors")

    # One request; the blank record was not sent; the model id came from env.
    assert len(state.requests) == 1
    assert state.requests[0]["model"] == "keywords-v1"
    assert state.requests[0]["input"] == [
        records[0],
        records[1],
        records[3],
        "show me the errors",
    ]
    # "timeout" shares no token with the query, so only the embedding half
    # can lift it over the widget record.
    assert scores[0].score > scores[1].score
    assert scores[3].score > scores[1].score
    assert all(score.reason.startswith("Hybrid (α=0.50)") for score in scores)

    # The server falls over: one request (no retries on this path), BM25 for
    # the call, and nothing more is sent until the cooldown ends.
    state.status = 503
    expected = _bm25_only_hybrid().score_batch(records, "show me the errors")
    assert scorer.score_batch(records, "show me the errors") == expected
    assert scorer.score_batch(records, "show me the errors") == expected
    assert len(state.requests) == 2

    state.status = 200
    clock[0] += embedding.BACKEND_COOLDOWN_SECONDS + 1
    assert scorer.score_batch(records, "show me the errors") == scores
    assert len(state.requests) == 3
