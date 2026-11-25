"""Embedding-based relevance scorer for Headroom SDK.

This module provides semantic relevance scoring using `fastembed`
(BAAI/bge-small-en-v1.5 by default — 33M params, 384 dims, ~67 MB
quantized ONNX). The Rust scorer (fastembed-rs crate) loads the same
pinned Qdrant/bge-small-en-v1.5-onnx-Q snapshot, so both languages run
the same ONNX file and tokenizer. On the same ONNX Runtime build the two
agree within 1e-6 per component; across ORT versions they drift by up to
~4e-4, so they are close, not byte-equal. The gated parity fixture in
tests/parity/fixtures/embedding/ bounds that drift.

Key features:
- Semantic understanding ("errors" matches "failed", "issues")
- Handles paraphrases and synonyms
- ONNX-backed inference (no PyTorch / no CUDA required)
- ~2-3x faster than sentence-transformers' all-MiniLM-L6-v2
- Outranks all-MiniLM-L6-v2 by ~6 MTEB points

Install with: pip install "headroom-ai[relevance]"

Pluggable backends: ``EmbeddingScorer(backend=...)`` scores with any
:class:`~headroom.models.backends.EmbeddingBackend` instead (an
OpenAI-compatible server, a plugin, a plain function). This module also holds
the built-in ``fastembed`` backend, :class:`FastEmbedBackend`, which loads the
same model through the same loaders as the scorer: keeping both here keeps the
fastembed construction sites to the two the air-gap allowlist accounts for.

History: this module previously wrapped `sentence-transformers`
(PyTorch). Switched to fastembed in Stage 3c.1 of the Rust port to:
1. Run the same ONNX model as the Rust embedding scorer (both call
   into ONNX Runtime over the identical pinned ONNX file).
2. Remove the torch dependency from the relevance/ path
   (Phase 6: "drop torch from Python").
3. Get a better default model (bge-small-en-v1.5 vs MiniLM-L6-v2).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headroom.log_safety import describe_class, describe_exception
from headroom.models.backends import (
    BackendError,
    BackendLoadError,
    EmbeddingBackend,
    EmbeddingBackendResponseError,
    as_embedding_backend,
    embed_texts,
)
from headroom.models.backends.embedding import _backend_label, _positive_int_or_none
from headroom.offline import OfflineEgressBlocked, guard_egress
from headroom.onnx_runtime import _resolve_revision, hf_hub_download_local_first

from .backend_config import create_relevance_embedding_backend
from .base import RelevanceScore, RelevanceScorer

# Texts per ONNX Runtime run. fastembed pads each batch to its longest text, and
# its default batch of 256 grows ORT's CPU arena (which never shrinks) to fit
# the biggest batch a proxy ever scores: 2.5 GB retained in a long-lived macOS
# proxy. 16 holds ~10x less and runs faster, with less padding per batch.
EMBED_BATCH_SIZE = 16


def _shrink_arena_after_each_run(model: Any) -> None:  # reaches into fastembed internals
    """Have ORT hand its free CPU arena regions back after every run.

    The batch cap bounds one run, but concurrent relevance splits share the
    session and still stack its arena: ~2.2 GB retained after four parallel
    128-record splits in a macOS proxy, ~80 MB with arena shrinkage, same speed.
    Only on macOS under ``MallocLargeCache=0``, which the proxy CLI's malloc
    re-exec sets: with libmalloc's large cache on, the freed regions stay dirty
    and shrinking re-grows the arena (~3.8 GB). Other platforms are unmeasured.
    """
    model._headroom_arena_shrink_checked = True
    if sys.platform != "darwin" or os.environ.get("MallocLargeCache") != "0":
        return
    try:
        import onnxruntime as ort

        session = model.model.model  # TextEmbedding -> OnnxTextEmbedding -> InferenceSession
        options = ort.RunOptions()
        options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
        run = session.run
        session.run = lambda names, feed, run_options=None: run(names, feed, run_options or options)
    except Exception as exc:  # noqa: BLE001 - memory only, the arena keeps its high-water mark
        logger.debug("ORT arena shrinkage unavailable: %s", exc)


# numpy is an optional dependency - import lazily
_numpy = None


def _get_numpy():
    """Lazily import numpy."""
    global _numpy
    if _numpy is None:
        try:
            import numpy as np

            _numpy = np
        except ImportError as e:
            raise ImportError(
                "numpy is required for EmbeddingScorer. "
                'Install with: pip install "headroom-ai[relevance]"'
            ) from e
    return _numpy


if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


@contextmanager
def _hf_hub_forced_offline() -> Iterator[None]:
    """Force the HuggingFace stack offline for the duration of the block.

    huggingface_hub reads HF_HUB_OFFLINE once, at import, into
    ``constants.HF_HUB_OFFLINE``; if it was imported before this call the env
    var alone changes nothing. Force both, and restore both on exit.
    """
    from huggingface_hub import constants as hf_constants

    previous = os.environ.get("HF_HUB_OFFLINE")
    previous_constant = hf_constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = previous_constant
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _load_text_embedding(kwargs: dict[str, Any]) -> TextEmbedding:
    """Build a fastembed ``TextEmbedding``, cache first and network second.

    ``onnx_runtime.hf_hub_download_local_first`` can hang the air-gap guard on
    an explicit ``local_files_only=True`` attempt because ``hf_hub_download``
    takes that flag. fastembed's constructor does not, so the same cache-hit /
    network-fallback split has to be made here: try the load with the whole
    HuggingFace stack forced offline, and only reach for the guard when that
    fails, which is exactly the case where a socket would have been opened.

    This keeps the behaviour an air-gapped deployment actually wants — a
    pre-seeded cache still loads under ``HEADROOM_OFFLINE`` — while making a
    cold cache refuse instead of quietly dialling huggingface.co. A plain
    unconditional guard would have broken the pre-seeded case, which is the
    reason this path was left unguarded before.

    The ``HF_HUB_OFFLINE`` flip is process-global for the duration of the
    call. That is acceptable here because loading is one-shot and memoised by
    the caller, and because the value it forces is *stricter* than whatever it
    replaces, so a concurrent HuggingFace call can only be refused locally,
    never sent somewhere it would not otherwise go.
    """
    from fastembed import TextEmbedding

    try:
        with _hf_hub_forced_offline():
            return TextEmbedding(**kwargs)
    except Exception as cache_miss:  # noqa: BLE001 - any local-lookup failure
        logger.debug("fastembed cache lookup failed, falling back to network: %s", cache_miss)

    guard_egress(
        f"fastembed weight download for {kwargs['model_name']}",
        "huggingface.co",
    )
    return TextEmbedding(**kwargs)


# Default model name. Same model the Rust embedding scorer loads.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# The HF repo fastembed resolves DEFAULT_MODEL_NAME to, and every file a
# fastembed load of it reads. Its revision is pinned in
# ``onnx_runtime._PINNED_REVISIONS`` (HEADROOM_HF_PIN=off floats it). The Rust
# scorer loads the same snapshot: keep both equal to DEFAULT_MODEL_REPO and
# DEFAULT_MODEL_FILES in crates/headroom-core/src/relevance/embedding.rs.
DEFAULT_MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
DEFAULT_MODEL_FILES = (
    "model_optimized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def _resolve_default_model_snapshot() -> str:
    """Return the local directory holding the pinned default-model snapshot.

    fastembed cannot pin this itself: ``TextEmbedding`` accepts ``revision``
    but drops it before ``snapshot_download`` (fastembed 0.9.0), so passing it
    left the download floating on the repo's ``main``. Each file is resolved
    here through ``hf_hub_download_local_first`` instead, which applies the
    pinned SHA, tries the cache first and guards the network fallback.

    The files land in the standard HuggingFace hub cache, which the Rust
    scorer searches too, rather than fastembed's default under the system temp
    dir, which macOS purges after a few idle days.

    Raises:
        OfflineEgressBlocked: ``HEADROOM_OFFLINE`` is set and a file is not cached.
        RuntimeError: the files resolved to more than one snapshot, which only
            happens when ``main`` moved mid-download with pinning turned off.
    """
    paths = [
        Path(hf_hub_download_local_first(DEFAULT_MODEL_REPO, name)) for name in DEFAULT_MODEL_FILES
    ]
    snapshot_dirs = {path.parent for path in paths}
    if len(snapshot_dirs) != 1:
        raise RuntimeError(
            f"{DEFAULT_MODEL_REPO} files resolved to more than one snapshot: "
            f"{sorted(str(d) for d in snapshot_dirs)}"
        )
    return str(snapshot_dirs.pop())


def _default_model_seed_hint() -> str:
    """How to pre-seed DEFAULT_MODEL_NAME on a host that cannot download it.

    Names the exact snapshot the load looks up, because a fastembed cache
    seeded for an older Headroom no longer satisfies it.
    """
    revision = _resolve_revision(DEFAULT_MODEL_REPO, None) or "main"
    return (
        f"Pre-seed the HuggingFace hub cache with {DEFAULT_MODEL_REPO}@{revision} "
        f"(`hf download {DEFAULT_MODEL_REPO} {' '.join(DEFAULT_MODEL_FILES)} "
        f"--revision {revision}`), or use the BM25-only scorer. A fastembed cache "
        "seeded for an older Headroom is not read for this model."
    )


def _load_default_model() -> TextEmbedding:
    """Load DEFAULT_MODEL_NAME from its pinned snapshot, with no fastembed lookup.

    Given ``specific_model_path`` fastembed loads that directory without
    touching the Hub; the construction still runs with the Hub forced offline
    so that a fastembed that stopped honouring it fails here instead of
    downloading an unpinned copy.
    """
    from fastembed import TextEmbedding

    snapshot = _resolve_default_model_snapshot()
    with _hf_hub_forced_offline():
        return TextEmbedding(model_name=DEFAULT_MODEL_NAME, specific_model_path=snapshot)


def _load_fastembed_model(model_name: str) -> TextEmbedding:
    """Load ``model_name`` with the loader that owns it.

    The single dispatch both the scorer and :class:`FastEmbedBackend` go
    through, so neither grows a fastembed construction site of its own (the
    air-gap allowlist counts the two above).

    Raises:
        OfflineEgressBlocked: ``HEADROOM_OFFLINE`` is set and the weights are
            not cached.
    """
    if model_name == DEFAULT_MODEL_NAME:
        return _load_default_model()
    # Other catalog models are not pinned: fastembed resolves them on the
    # repo's main, cache first.
    return _load_text_embedding({"model_name": model_name})


def _model_unavailable_message(model_name: str, blocked: OfflineEgressBlocked) -> str:
    """What to tell a caller whose model the air gap kept from downloading.

    Translated at the boundary that owns the degradation, the way the Kompress
    and ONNX loaders do. Public model weights are not data leaving the box, so
    what the caller needs to hear is "this model is not available here and
    why", not an air-gap type it has never seen.
    """
    if model_name == DEFAULT_MODEL_NAME:
        seed_hint = _default_model_seed_hint()
    else:
        seed_hint = "Pre-seed the fastembed cache on this host, or use the BM25-only scorer."
    return f"Embedding model {model_name!r} is unavailable: {str(blocked).rstrip('.')}. {seed_hint}"


# Output width of DEFAULT_MODEL_NAME, known without loading it.
DEFAULT_MODEL_DIMENSION = 384

# How long a failed embedding backend is left alone before one call retries it.
BACKEND_COOLDOWN_SECONDS = 30.0

# Module-level indirection over the clock so tests can drive the circuit
# breaker without monkeypatching stdlib ``time.monotonic`` for every thread in
# the process. Same pattern as proxy.helpers and savings_tracker.
_monotonic = time.monotonic


class EmbeddingBackendCircuitOpenError(BackendError):
    """An embedding backend failed recently, so the call was refused without contacting it.

    Raised by :class:`EmbeddingScorer` while its circuit breaker is open. A
    :class:`~headroom.relevance.HybridScorer` treats it like any other failure
    of its embedding half and scores with BM25 alone.
    """


class _CircuitBreaker:
    """Fail fast for ``cooldown`` seconds after a failure, then let one call probe.

    Closed: every call goes through. A failure opens it: calls are refused
    until the cooldown ends, after which exactly one call (the probe) goes
    through while the others keep being refused. The probe's success closes
    the breaker; its failure starts a new cooldown.

    A remote backend on the request path is the reason this exists: without
    it a dead endpoint costs every request its full timeout.
    """

    def __init__(self, cooldown: float) -> None:
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._open_until: float | None = None  # None while closed
        self._probing = False

    def admit(self) -> tuple[float | None, bool]:
        """Decide one call: ``(seconds left to wait or None, whether it is the probe)``.

        A call may go through when the first element is ``None``. A probe must
        end in :meth:`succeeded`, :meth:`failed` or :meth:`end_probe`.
        """
        with self._lock:
            if self._open_until is None:
                return None, False
            remaining = self._open_until - _monotonic()
            if remaining > 0 or self._probing:
                return max(remaining, 0.0), False
            self._probing = True
            return None, True

    def succeeded(self) -> bool:
        """Close the breaker. True when it was open, i.e. the backend recovered."""
        with self._lock:
            was_open = self._open_until is not None
            self._open_until = None
            self._probing = False
            return was_open

    def failed(self) -> bool:
        """Start a cooldown. True when this opened a closed breaker."""
        with self._lock:
            opened = self._open_until is None
            self._open_until = _monotonic() + self.cooldown
            self._probing = False
            return opened

    def end_probe(self) -> None:
        """Give back the probe slot; call it after every probe.

        A no-op once :meth:`succeeded` or :meth:`failed` settled the probe. A
        probe that ended in a BaseException settles neither; this keeps it from
        wedging the breaker, which stays open with its cooldown spent so the
        next call probes.
        """
        with self._lock:
            self._probing = False


def _cooldown_seconds(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"backend_cooldown_seconds must be a positive number of seconds, got {value!r}"
        )
    return float(value)


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (numpy array).
        b: Second vector (numpy array).

    Returns:
        Cosine similarity in range [-1, 1], clamped to [0, 1].
    """
    np = _get_numpy()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [0, 1] since we only care about positive similarity
    return max(0.0, min(1.0, similarity))


class EmbeddingScorer(RelevanceScorer):
    """Semantic relevance scorer using fastembed (ONNX-backed).

    Default model: BAAI/bge-small-en-v1.5 (33M params, 384 dims).
    Auto-downloads its pinned snapshot from HuggingFace Hub on first
    use (~67 MB quantized ONNX).

    Example:
        scorer = EmbeddingScorer()
        score = scorer.score(
            '{"status": "failed", "error": "connection refused"}',
            "show me the errors"
        )
        # score.score > 0.5 (semantic match between "failed"/"error" and "errors")

    With ``backend`` the scorer embeds through that
    :class:`~headroom.models.backends.EmbeddingBackend` instead of fastembed:

        scorer = EmbeddingScorer(backend="openai:text-embedding-3-small")

    A backend is called through :func:`~headroom.models.backends.embed_texts`,
    so empty texts are never sent and every response is validated. A failing
    backend opens a circuit breaker: for ``backend_cooldown_seconds`` calls
    raise :class:`EmbeddingBackendCircuitOpenError` without contacting it,
    then one call retries it. That bounds what a dead remote endpoint costs
    the request path to one timeout per cooldown.

    Note:
        Without a backend, requires fastembed: pip install "headroom-ai[relevance]"
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_model: bool = True,  # Kept for API compatibility
        *,
        backend: EmbeddingBackend | str | Callable[[list[str]], Any] | None = None,
        backend_cooldown_seconds: float = BACKEND_COOLDOWN_SECONDS,
    ):
        """Initialize embedding scorer.

        Args:
            model_name: Sentence-embedding model name. Default
                "BAAI/bge-small-en-v1.5". See fastembed's catalog for
                supported models (BGE, E5, MiniLM, jina, etc.). Not used
                with ``backend``, which names its own model.
            cache_model: Deprecated, models are always cached via
                fastembed's HF Hub cache.
            backend: Embed with this instead of fastembed: an
                ``EmbeddingBackend``, a registry spec (``"name"`` or
                ``"name:model"``) or a function ``fn(list[str]) -> rows``.
                A spec is constructed here, so a bad one fails now. A spec
                naming the built-in ``openai`` backend gets the request-path
                bounds (a 5 s timeout, no retries); pass an instance to
                choose others.
            backend_cooldown_seconds: How long a failed backend is left
                alone before one call retries it.

        Raises:
            ValueError: Both ``model_name`` and ``backend`` were given, or the
                cooldown is not a positive number.
        """
        self.cache_model = cache_model
        self._model: TextEmbedding | None = None
        self._model_lock = threading.Lock()
        self._backend: EmbeddingBackend | None = None
        if backend is None:
            self.model_name = model_name or DEFAULT_MODEL_NAME
            return
        if model_name is not None:
            # Picking one silently would score with a model the caller did not ask for.
            raise ValueError(
                "Pass model_name or backend, not both; give a registry spec its model as "
                "'name:model'"
            )
        self._breaker = _CircuitBreaker(_cooldown_seconds(backend_cooldown_seconds))
        # A spec gets the request-path bounds RelevanceScorerConfig's specs get.
        self._backend = (
            create_relevance_embedding_backend(backend)
            if isinstance(backend, str)
            else as_embedding_backend(backend)
        )
        self._backend_label = _backend_label(self._backend)
        self.model_name = self._backend.model_name

    @property
    def backend(self) -> EmbeddingBackend | None:
        """The embedding backend this scorer uses, or ``None`` for fastembed."""
        return self._backend

    @classmethod
    def is_available(cls) -> bool:
        """Check if fastembed is installed.

        Returns:
            True if the package is available.
        """
        try:
            import fastembed  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_model(self) -> TextEmbedding:
        """Get or load the fastembed text embedding model.

        Returns:
            Loaded TextEmbedding model.

        Raises:
            RuntimeError: If fastembed is not installed.
        """
        if not self.is_available():
            raise RuntimeError(
                "EmbeddingScorer requires fastembed. "
                'Install with: pip install "headroom-ai[relevance]"'
            )

        if self._model is None:
            # Locked so concurrent first requests (parallel relevance splits)
            # load the model once rather than once each.
            with self._model_lock:
                if self._model is None:
                    try:
                        self._model = _load_fastembed_model(self.model_name)
                    except OfflineEgressBlocked as blocked:
                        raise RuntimeError(
                            _model_unavailable_message(self.model_name, blocked)
                        ) from None
        return self._model

    def _encode(self, texts: list[str]):
        """Encode texts to embeddings via the backend, or fastembed without one.

        fastembed's `embed` returns an iterator yielding numpy arrays
        (one per text). We materialize to a list/np.array for the
        cosine-similarity downstream.

        Args:
            texts: List of texts to encode.

        Returns:
            numpy array of embeddings, shape (len(texts), embedding_dim).
        """
        if self._backend is not None:
            return self._encode_with_backend(texts)
        np = _get_numpy()
        model = self._get_model()
        if not getattr(model, "_headroom_arena_shrink_checked", False):
            _shrink_arena_after_each_run(model)
        embeddings = list(model.embed(texts, batch_size=EMBED_BATCH_SIZE))
        return np.array(embeddings)

    def _encode_with_backend(self, texts: list[str]) -> Any:
        """Embed through the backend, behind the circuit breaker.

        Raises:
            EmbeddingBackendCircuitOpenError: The backend failed less than a
                cooldown ago (or another call is already retrying it).
            Exception: Whatever :func:`embed_texts` raised; it opens the breaker.
        """
        assert self._backend is not None
        remaining, probe = self._breaker.admit()
        if remaining is not None:
            wait = (
                f"retrying in {math.ceil(remaining)}s" if remaining > 0 else "a retry is in flight"
            )
            raise EmbeddingBackendCircuitOpenError(
                f"Embedding backend {self._backend_label!r} failed recently; {wait}"
            )
        try:
            matrix = embed_texts(self._backend, texts)
        except Exception as exc:
            self._backend_failed(exc)
            raise
        else:
            if self._breaker.succeeded():
                logger.info(
                    "Relevance embedding backend %r recovered; embedding scores resume",
                    self._backend_label,
                )
            return matrix
        finally:
            # OfflineEgressBlocked (a BaseException) is a policy refusal, not a
            # backend failure: it neither opens nor closes the breaker, and it
            # propagates. Only the probe slot it may hold is given back.
            if probe:
                self._breaker.end_probe()

    def _backend_failed(self, exc: Exception) -> None:
        """Open the breaker and report it: one WARNING per opening, detail at DEBUG.

        The WARNING names the backend and the exception's class only: a
        third-party backend's message can quote its endpoint, key or input.
        """
        if self._breaker.failed():
            logger.warning(
                "Relevance embedding backend %r failed (%s); not calling it again for %.0fs, "
                "then one call retries it. Details are logged at DEBUG.",
                self._backend_label,
                describe_class(type(exc)),
                self._breaker.cooldown,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Relevance embedding backend %r failed: %s",
                self._backend_label,
                describe_exception(exc),
            )

    def score(self, item: str, context: str) -> RelevanceScore:
        """Score item relevance to context using embeddings.

        Args:
            item: Item text.
            context: Query context.

        Returns:
            RelevanceScore with embedding-based similarity.
        """
        if not item or not context:
            return RelevanceScore(score=0.0, reason="Embedding: empty input")

        embeddings = self._encode([item, context])
        similarity = _cosine_similarity(embeddings[0], embeddings[1])

        return RelevanceScore(
            score=similarity,
            reason=f"Embedding: semantic similarity {similarity:.2f}",
        )

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        """Score multiple items efficiently using batch encoding.

        Encodes items + context in a single fastembed call. Mirrors the
        Rust scorer's batch behavior so both languages do the same
        amount of work for the same input.

        Args:
            items: List of items to score.
            context: Query context.

        Returns:
            List of RelevanceScore objects.
        """
        if not items:
            return []

        if not context:
            return [RelevanceScore(score=0.0, reason="Embedding: empty context") for _ in items]

        # Encode all texts in one batch
        all_texts = items + [context]
        embeddings = self._encode(all_texts)

        # Last embedding is the context
        context_emb = embeddings[-1]
        item_embs = embeddings[:-1]

        # Compute similarities
        results = []
        for emb in item_embs:
            similarity = _cosine_similarity(emb, context_emb)
            results.append(
                RelevanceScore(
                    score=similarity,
                    reason=f"Embedding: {similarity:.2f}",
                )
            )

        return results


# Convenience function for checking availability without instantiation
def embedding_available() -> bool:
    """Check if embedding scorer is available.

    Returns:
        True if fastembed is installed.
    """
    return EmbeddingScorer.is_available()


class FastEmbedBackend:
    """The built-in ``fastembed`` embedding backend: a local ONNX model.

    Registered as ``"fastembed"`` in :data:`~headroom.models.backends.EMBEDDING_BACKENDS`.
    It loads models exactly the way :class:`EmbeddingScorer` does: the default
    from its pinned snapshot, any other catalog model through fastembed's own
    resolution, cache first, with the network fallback refused under
    ``HEADROOM_OFFLINE``. ORT runs are capped at :data:`EMBED_BATCH_SIZE`
    texts and shrink their arena where that is measured to help.

    The model is loaded on the first :meth:`embed`, not here, so building the
    backend (at startup, to validate a configuration) costs no download.

    Args:
        model: fastembed model id. Defaults to ``BAAI/bge-small-en-v1.5``.
        dimension: Expected output width. Known (384) for the default model,
            where any other value is an error; otherwise learnt from the first
            call, or checked against it when given.

    Raises:
        RuntimeError: fastembed is not installed.
        ValueError: ``dimension`` is not a positive integer, or is not the
            default model's width.
    """

    name = "fastembed"

    def __init__(self, model: str | None = None, *, dimension: int | None = None) -> None:
        if not EmbeddingScorer.is_available():
            raise RuntimeError(
                "The fastembed embedding backend requires fastembed. "
                'Install with: pip install "headroom-ai[relevance]"'
            )
        self.model_name = (model or "").strip() or DEFAULT_MODEL_NAME
        declared = _positive_int_or_none(dimension, "dimension")
        if self.model_name == DEFAULT_MODEL_NAME:
            if declared not in (None, DEFAULT_MODEL_DIMENSION):
                raise ValueError(
                    f"{DEFAULT_MODEL_NAME} produces {DEFAULT_MODEL_DIMENSION}-dimensional "
                    f"vectors, not {declared}"
                )
            declared = DEFAULT_MODEL_DIMENSION
        self.dimension: int | None = declared
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> TextEmbedding:
        model = self._model
        if model is not None:
            return model
        with self._lock:
            if self._model is None:
                try:
                    loaded = _load_fastembed_model(self.model_name)
                except OfflineEgressBlocked as blocked:
                    # A BackendError, unlike the scorer's plain RuntimeError, so
                    # embed_texts passes the seeding hint through instead of
                    # reducing it to an exception class.
                    raise BackendLoadError(
                        _model_unavailable_message(self.model_name, blocked)
                    ) from None
                _shrink_arena_after_each_run(loaded)
                self._model = loaded
            return self._model

    def embed(self, texts: list[str]) -> Any:
        """Embed ``texts`` and return a float32 matrix, one row per text."""
        np = _get_numpy()
        if not texts:
            return np.zeros((0, self.dimension or 0), dtype=np.float32)
        model = self._get_model()
        rows = np.asarray(
            list(model.embed(list(texts), batch_size=EMBED_BATCH_SIZE)), dtype=np.float32
        )
        if rows.ndim == 2:
            width = int(rows.shape[1])
            if self.dimension is None:
                self.dimension = width
            elif width != self.dimension:
                raise EmbeddingBackendResponseError(
                    f"fastembed model {self.model_name!r} produced {width}-dimensional "
                    f"vectors, not {self.dimension}"
                )
        return rows

    def __repr__(self) -> str:
        return f"FastEmbedBackend(model={self.model_name!r}, dimension={self.dimension!r})"
