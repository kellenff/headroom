"""Memory's async ``Embedder`` over a pluggable, synchronous embedding backend.

Backends from :mod:`headroom.models.backends` are synchronous: ``embed()``
blocks while a model runs or an HTTP request is in flight. Memory's
:class:`~headroom.memory.ports.Embedder` protocol is async, so
:class:`BackendEmbedder` runs every call on a worker thread, through
:func:`~headroom.models.backends.embed_texts`, which keeps empty texts away from
the backend, chunks by its ``max_batch_size`` and validates every row before a
vector reaches an index.

The worker pool is dedicated, not the event loop's default executor: a slow
remote backend must not occupy the threads that ``asyncio.to_thread`` callers
elsewhere in the proxy rely on. It is sized like the local embedders' pool
(``HEADROOM_EMBED_CONCURRENCY``) so one memory embedder never claims more
parallelism than the built-in one would.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from headroom.log_safety import describe_class, describe_exception
from headroom.memory.adapters.embedders import _resolve_embed_concurrency
from headroom.models.backends import embed_texts
from headroom.models.backends.embedding import _backend_label

if TYPE_CHECKING:
    import numpy as np

    from headroom.models.backends import EmbeddingBackend

logger = logging.getLogger(__name__)


class BackendEmbedder:
    """Adapt a synchronous :class:`~headroom.models.backends.EmbeddingBackend` to memory.

    Args:
        backend: The backend to call. Shared freely: it must be thread-safe
            (part of the backend contract).
        dimension: The vector index's dimension. Every row is checked against
            it, so a backend that answers with another width fails the write
            instead of corrupting the index.

    Rows are L2-normalised, like every other memory embedder's, so cosine
    scores mean the same whichever embedder wrote them.

    ``close()`` is idempotent and revivable: it releases the worker pool and
    closes the backend, and a later ``embed`` starts a fresh pool. Headroom
    shares one embedder between many memory backends, so a close by one of
    them must not break the others.
    """

    DEFAULT_MAX_TOKENS = 8192

    def __init__(self, backend: EmbeddingBackend, *, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
        self._backend = backend
        self._dimension = dimension
        self._executor: ThreadPoolExecutor | None = None
        # Guards the pool so a submit never races close() onto a pool that is
        # already shutting down.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ Embedder

    async def embed(self, text: str) -> np.ndarray:
        """Embed one text. An empty or whitespace-only text gives a zero vector."""
        matrix = await self._embed_texts([text])
        row: np.ndarray = matrix[0]
        return row

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed ``texts`` in order; empty texts give zero vectors."""
        if not texts:
            return []
        matrix = await self._embed_texts(list(texts))
        return list(matrix)

    @property
    def dimension(self) -> int:
        """The configured vector dimension (what every row is checked against)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """The model the backend serves."""
        return str(self._backend.model_name)

    @property
    def max_tokens(self) -> int:
        """The backend's input window, or 8192 when it does not say.

        Read on every access: some backends learn it when their model loads.
        """
        value = getattr(self._backend, "max_tokens", None)
        return value if isinstance(value, int) and value > 0 else self.DEFAULT_MAX_TOKENS

    @property
    def backend(self) -> EmbeddingBackend:
        """The wrapped backend."""
        return self._backend

    async def close(self) -> None:
        """Release the worker pool and close the backend. Safe to call repeatedly.

        Queued (not yet started) work is cancelled and resubmitted by its
        caller to a fresh pool; a call already running finishes on its thread.
        """
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        close = getattr(self._backend, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            # Closing is best-effort: a backend that fails to close must not
            # stop the memory system from closing its stores and indexes.
            logger.debug("Embedding backend close failed: %s", describe_exception(exc))
            logger.warning(
                "Memory: closing embedding backend %r failed (%s)",
                _backend_label(self._backend),
                describe_class(type(exc)),
            )

    # ------------------------------------------------------------------ internals

    def _submit(self, call: Callable[[], np.ndarray]) -> Future[np.ndarray]:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=_resolve_embed_concurrency(),
                    thread_name_prefix="backend-embed",
                )
            return self._executor.submit(call)

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        call = functools.partial(
            embed_texts,
            self._backend,
            texts,
            expected_dimension=self._dimension,
            normalize=True,
        )
        while True:
            future = self._submit(call)
            try:
                return await asyncio.wrap_future(future)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if future.cancelled() and task is not None and not task.cancelling():
                    # close() dropped this queued call; the caller was not
                    # cancelled, so run it on the revived pool instead.
                    continue
                raise

    def __repr__(self) -> str:
        # Not the backend's own repr, nor a name that is not a registry name: a
        # third-party one may carry its options (an endpoint, a key).
        return (
            f"BackendEmbedder(backend={_backend_label(self._backend)!r}, model={self.model_name!r}, "
            f"dimension={self._dimension})"
        )
