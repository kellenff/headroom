"""The ContentRouter's relevance split on a configured embedding backend.

``RelevanceScorerConfig.embedding_backend`` reaches the router's background
prewarm, which builds ``create_scorer(tier, embedding_backend=...)``. Covered
here: the default stays the fastembed path and never reads the environment,
a spec and ``"env"`` select the backend, a broken or air-gapped backend leaves
the router on BM25 and says so, and a backend-scored split is record-capped.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import headroom.offline as offline_module
import headroom.transforms.content_router as content_router_module
from headroom.config import RelevanceScorerConfig
from headroom.models.backends import (
    EMBEDDING_BACKENDS,
    CallableEmbeddingBackend,
    register_embedding_backend,
)
from headroom.offline import OFFLINE_ENV
from headroom.relevance import BM25Scorer, EmbeddingScorer, HybridScorer, embedding
from headroom.relevance.backend_config import RELEVANCE_BACKEND_MAX_RECORDS
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

ROUTER_LOGGER = "headroom.transforms.content_router"


def _keyword_vectors(texts: list[str]) -> list[list[float]]:
    return [
        [float(t.lower().count("error") + t.lower().count("timeout")), float("widget" in t), 0.01]
        for t in texts
    ]


class _FakeBackends:
    """A registered ``fake-rel`` backend that records what it builds and embeds."""

    def __init__(self) -> None:
        self.built: list[dict[str, Any]] = []
        self.batches: list[list[str]] = []
        self.fail_construction: Exception | None = None
        self.fail_embed: Exception | None = None
        self.gate: threading.Event | None = None

    def factory(self, **options: Any) -> CallableEmbeddingBackend:
        self.built.append(dict(options))
        if self.fail_construction is not None:
            raise self.fail_construction
        return CallableEmbeddingBackend(self._embed, name="fake-rel", model=options.get("model"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail_embed is not None:
            raise self.fail_embed
        return _keyword_vectors(texts)


@pytest.fixture
def fake() -> Iterator[_FakeBackends]:
    EMBEDDING_BACKENDS._reset_for_tests()
    backends = _FakeBackends()
    register_embedding_backend("fake-rel", backends.factory)
    yield backends
    EMBEDDING_BACKENDS._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_relevance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PROVIDER",
        "MODEL",
        "BASE_URL",
        "API_KEY",
        "DIMENSION",
        "HEADERS",
        "OPTIONS",
    ):
        monkeypatch.delenv(f"HEADROOM_RELEVANCE_EMBEDDER_{name}", raising=False)
    monkeypatch.delenv(OFFLINE_ENV, raising=False)


def _router(embedding_backend: str | None, **config: Any) -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            relevance=RelevanceScorerConfig(embedding_backend=embedding_backend), **config
        )
    )


def _backend_of(scorer: Any) -> Any:
    inner = scorer.embedding if isinstance(scorer, HybridScorer) else scorer
    return inner.backend if isinstance(inner, EmbeddingScorer) else None


class TestPrewarmSelection:
    def test_the_default_never_reads_the_environment(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Library compress() builds the default router: an operator's proxy
        variables in the same process must not send its tool output anywhere."""
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "fake-rel")
        router = ContentRouter(ContentRouterConfig())

        router._prewarm_relevance_scorer("hybrid")

        assert fake.built == []
        assert _backend_of(router._relevance_scorer) is None
        assert router._relevance_max_records() == 0

    def test_a_spec_selects_the_backend(
        self, fake: _FakeBackends, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
        router = _router("fake-rel:m1")

        router._prewarm_relevance_scorer("hybrid")

        assert fake.built == [{"model": "m1"}], "a spec carries no other options"
        scorer = router._relevance_scorer
        assert isinstance(scorer, HybridScorer)
        assert _backend_of(scorer).model_name == "m1"
        assert fake.batches == [["warmup", "warmup"]], "warmed off the request thread"
        assert "uses embedding backend 'fake-rel'" in caplog.text

    def test_the_embedding_tier_selects_the_backend_too(self, fake: _FakeBackends) -> None:
        router = _router("fake-rel", relevance_split=True)
        router._prewarm_relevance_scorer("embedding")
        assert isinstance(router._relevance_scorer, EmbeddingScorer)
        assert _backend_of(router._relevance_scorer) is not None

    def test_env_reads_the_relevance_variables_when_the_scorer_is_built(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        router = _router("env")
        # Set after the router exists: the variables are read at prewarm.
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "fake-rel")
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_MODEL", "from-env")

        router._prewarm_relevance_scorer("hybrid")

        assert fake.built == [{"model": "from-env"}]
        assert _backend_of(router._relevance_scorer).model_name == "from-env"

    def test_env_with_nothing_selected_is_the_default_path(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(EmbeddingScorer, "is_available", classmethod(lambda cls: False))
        router = _router("env")
        router._prewarm_relevance_scorer("hybrid")
        assert fake.built == []
        scorer = router._relevance_scorer
        assert isinstance(scorer, HybridScorer) and not scorer.has_embedding_support()
        assert router._relevance_max_records() == 0

    def test_the_request_thread_serves_bm25_until_the_backend_is_warm(
        self, fake: _FakeBackends
    ) -> None:
        fake.gate = threading.Event()  # hold the warm-up embed
        router = _router("fake-rel:m")
        assert isinstance(router._get_relevance_scorer(), BM25Scorer)
        assert isinstance(router._get_relevance_scorer(), BM25Scorer)
        fake.gate.set()
        for thread in threading.enumerate():
            if thread.name == "relevance-prewarm":
                thread.join(10)
        assert _backend_of(router._get_relevance_scorer()) is not None

    def test_bm25_tier_with_env_never_reads_it(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "fake-rel")
        router = ContentRouter(
            ContentRouterConfig(
                relevance=RelevanceScorerConfig(tier="bm25", embedding_backend="env")
            )
        )
        assert isinstance(router._get_relevance_scorer(), BM25Scorer)
        assert fake.built == []


class TestDefaultFastembedPrewarm:
    """No backend configured: the fastembed path behaves as it did before."""

    @pytest.fixture
    def fastembed_stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=object))
        monkeypatch.setattr(embedding, "_shrink_arena_after_each_run", lambda model: None)

    def test_a_model_that_fails_to_load_leaves_bm25_in_place(
        self, fastembed_stub: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hybrid scorer now absorbs embedding failures, so warming through
        it would hide this one and swap in a scorer whose model never loads,
        retrying the load on every request. The embedding half is warmed itself."""
        loads: list[int] = []

        def broken_load() -> Any:
            loads.append(1)
            raise RuntimeError("download failed")

        monkeypatch.setattr(embedding, "_load_default_model", broken_load)
        router = ContentRouter(ContentRouterConfig())
        router._relevance_scorer = BM25Scorer()

        router._prewarm_relevance_scorer("hybrid")

        assert type(router._relevance_scorer) is BM25Scorer
        assert loads == [1]

    def test_a_model_that_loads_is_swapped_in(
        self, fastembed_stub: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Model:
            def embed(self, texts: list[str], **_kw: Any) -> Iterator[list[float]]:
                yield from _keyword_vectors(texts)

        monkeypatch.setattr(embedding, "_load_default_model", Model)
        router = ContentRouter(ContentRouterConfig())

        router._prewarm_relevance_scorer("hybrid")

        scorer = router._relevance_scorer
        assert isinstance(scorer, HybridScorer) and scorer.has_embedding_support()
        assert _backend_of(scorer) is None
        assert router._relevance_max_records() == 0


class TestPrewarmFailures:
    def test_a_backend_that_cannot_be_built_is_reported_and_bm25_stays(
        self, fake: _FakeBackends, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=ROUTER_LOGGER)
        fake.fail_construction = ValueError("bad api_key sk-secret-value")
        router = _router("fake-rel:m")
        router._relevance_scorer = BM25Scorer()

        router._prewarm_relevance_scorer("hybrid")

        assert isinstance(router._relevance_scorer, BM25Scorer)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "fake-rel" in message and "BackendLoadError" in message
        assert "stays on BM25" in message
        assert "sk-secret-value" not in caplog.text
        assert not router._relevance_uses_backend

    def test_a_misspelt_provider_is_a_warning_not_a_debug_line(
        self,
        fake: _FakeBackends,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Before backends, every prewarm failure went to DEBUG: a typo in the
        operator's selection would silently leave relevance on BM25."""
        caplog.set_level(logging.DEBUG, logger=ROUTER_LOGGER)
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "opanai")
        router = _router("env")
        router._relevance_scorer = BM25Scorer()

        router._prewarm_relevance_scorer("hybrid")

        assert isinstance(router._relevance_scorer, BM25Scorer)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "HEADROOM_RELEVANCE_EMBEDDER_*" in warnings[0].getMessage()
        assert "UnknownBackendError" in warnings[0].getMessage()

    def test_an_air_gap_refusal_is_reported_and_bm25_stays(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=ROUTER_LOGGER)
        monkeypatch.setattr(offline_module, "_REPORTED_REFUSALS", set())
        monkeypatch.setenv(OFFLINE_ENV, "1")
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_PROVIDER", "openai")
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_MODEL", "m")
        monkeypatch.setenv("HEADROOM_RELEVANCE_EMBEDDER_BASE_URL", "http://127.0.0.1:9/v1")
        hooked: list[Any] = []
        monkeypatch.setattr(threading, "excepthook", hooked.append)
        router = _router("env")

        # Through the real thread: an escaped BaseException would reach the hook.
        assert isinstance(router._get_relevance_scorer(), BM25Scorer)
        for thread in threading.enumerate():
            if thread.name == "relevance-prewarm":
                thread.join(10)

        assert hooked == []
        assert isinstance(router._relevance_scorer, BM25Scorer)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert OFFLINE_ENV in warnings[0].getMessage()
        assert not router._relevance_uses_backend

    def test_a_failing_warm_up_still_swaps_the_backend_in(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend that is down when the proxy starts is retried after the
        cooldown, not locked out for the life of the process."""
        now = [500.0]
        monkeypatch.setattr(embedding, "_monotonic", lambda: now[0])
        fake.fail_embed = ConnectionError("not up yet")
        router = _router("fake-rel:m")

        router._prewarm_relevance_scorer("hybrid")

        scorer = router._relevance_scorer
        assert _backend_of(scorer) is not None
        assert router._relevance_uses_backend
        assert len(fake.batches) == 1
        # During the cooldown requests score with BM25 without calling it...
        scorer.score_batch(["error one", "widget two"], "errors")
        assert len(fake.batches) == 1
        # ...and once it has passed, the backend is back in use.
        fake.fail_embed = None
        now[0] += embedding.BACKEND_COOLDOWN_SECONDS + 1
        fused = scorer.score_batch(["error one", "widget two"], "errors")
        assert len(fake.batches) == 2
        assert fused[0].reason.startswith("Hybrid (α=")


class TestRecordCap:
    @staticmethod
    def _records(count: int) -> str:
        return "".join(f"record {i} widget status ok\n\n" for i in range(count - 1)) + (
            "record final error: timeout\n"
        )

    @staticmethod
    def _warm(router: ContentRouter, tier: str = "hybrid") -> None:
        router._prewarm_relevance_scorer(tier)
        router._relevance_scorer_tried = True
        # Any signature: the Kompress side of the router may grow parameters.
        router._try_ml_compressor = lambda *_args, **_kwargs: ("[TAIL]", 1)  # type: ignore[method-assign]

    def test_a_backend_scored_split_is_capped_when_unbounded(self, fake: _FakeBackends) -> None:
        router = _router("fake-rel:m")
        self._warm(router)
        assert router._relevance_max_records() == RELEVANCE_BACKEND_MAX_RECORDS == 512
        fake.batches.clear()

        too_many = self._records(RELEVANCE_BACKEND_MAX_RECORDS + 10)
        assert router._relevance_split_compress(too_many, "log", "timeout errors") is None
        assert fake.batches == [], "an output over the cap is never sent to the backend"

        fits = self._records(40)
        router._relevance_split_compress(fits, "log", "timeout errors")
        assert len(fake.batches) == 1 and len(fake.batches[0]) == 41

    def test_an_explicit_cap_wins(self, fake: _FakeBackends) -> None:
        router = _router("fake-rel:m", relevance_max_records=5)
        self._warm(router)
        assert router._relevance_max_records() == 5

    def test_the_cap_reaches_the_planner(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        monkeypatch.setattr(
            content_router_module,
            "plan_relevance_split",
            lambda content, query, scorer, **kw: (
                seen.append(kw["max_records"]) or [(True, content)]
            ),
        )
        router = _router("fake-rel:m")
        self._warm(router)
        router._relevance_split_compress(self._records(3), "log", "errors")
        assert seen == [RELEVANCE_BACKEND_MAX_RECORDS]

    def test_without_a_backend_there_is_still_no_cap(
        self, fake: _FakeBackends, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(EmbeddingScorer, "is_available", classmethod(lambda cls: False))
        router = ContentRouter(ContentRouterConfig())
        self._warm(router)
        assert router._relevance_max_records() == 0
