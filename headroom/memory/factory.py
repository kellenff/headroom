"""Factory module for creating memory system components.

Provides a unified factory function that creates all memory system components
from a single configuration object, ensuring consistent initialization
and proper wiring between components.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from importlib.metadata import entry_points
from inspect import iscoroutinefunction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headroom.log_safety import describe_exception
from headroom.memory.config import (
    EmbedderBackend,
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
    external_embedder_selection,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from headroom.memory.ports import (
        Embedder,
        MemoryCache,
        MemoryStore,
        TextFilter,
        TextIndex,
        TextSearchResult,
        VectorIndex,
    )


logger = logging.getLogger(__name__)

# Extension groups for memory backends registered via setuptools entry points.
_MEMORY_STORE_GROUP = "headroom.memory_store"
_MEMORY_VECTOR_GROUP = "headroom.memory_vector"
_MEMORY_TEXT_GROUP = "headroom.memory_text"

# Process-wide embedder cache keyed by model and connection settings. Embedders are
# stateless with respect to the memory store, so a single instance can
# safely serve every per-project ``LocalBackend`` created by the
# BackendRouter. Without this cache, opening N project DBs would load
# the sentence-transformers / ONNX model N times. A cached embedder is shared,
# so no single memory system owns it (see ``embedder_is_shared``).
_EMBEDDER_CACHE: dict[tuple[str, ...], Embedder] = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()


def _load_external_backend(
    group: str,
    name: str | None,
    field_name: str,
    config: MemoryConfig,
) -> Any:
    """Load a memory backend registered via setuptools entry points.

    Mirrors the pattern used by
    `headroom.cache.compression_store._create_default_ccr_backend`.
    """
    if not name:
        raise ValueError(
            f"{field_name} is required when backend is EXTERNAL; "
            f"set it to the entry-point name registered under '{group}'."
        )
    ep = next((e for e in entry_points(group=group) if e.name == name), None)
    if ep is None:
        raise ValueError(
            f"No entry point registered under '{group}' with name '{name}'. "
            f"Install the package that provides it."
        )
    return ep.load()(config)


async def create_memory_system(
    config: MemoryConfig | None = None,
) -> tuple[MemoryStore, VectorIndex, TextIndex, Embedder, MemoryCache | None]:
    """Create a complete memory system from configuration.

    This factory function creates and initializes all memory system components
    based on the provided configuration. Components are created in dependency
    order to ensure proper initialization.

    Args:
        config: Memory system configuration. If None, uses default configuration.

    Returns:
        A tuple of (store, vector_index, text_index, embedder, cache) where:
        - store: The memory persistence backend
        - vector_index: The vector similarity search index
        - text_index: The full-text search index
        - embedder: The text embedding generator
        - cache: The memory cache (or None if caching is disabled)

    Raises:
        ValueError: If an unknown backend type is specified in the config.

    Example:
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.LOCAL,
            cache_max_size=2000,
        )
        store, vector, text, embedder, cache = await create_memory_system(config)
    """
    config = config or MemoryConfig()

    # Create the embedder first: it is where configuration mistakes surface
    # (unknown backend, missing model or key, wrong dimension), and failing
    # here leaves no database file behind for the operator to clean up.
    embedder = _create_embedder(config)

    # Create store
    store = _create_store(config)

    # Create vector index
    vector_index = _create_vector_index(config)

    # Create text index
    text_index = _create_text_index(config)

    # Create cache (optional)
    cache = _create_cache(config) if config.cache_enabled else None

    return store, vector_index, text_index, embedder, cache


def _create_store(config: MemoryConfig) -> MemoryStore:
    """Create a memory store backend.

    Args:
        config: Memory system configuration.

    Returns:
        A MemoryStore implementation based on config.store_backend.

    Raises:
        ValueError: If the store backend is not supported.
    """
    if config.store_backend == StoreBackend.SQLITE:
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        return SQLiteMemoryStore(config.db_path)

    if config.store_backend == StoreBackend.EXTERNAL:
        return _load_external_backend(  # type: ignore[no-any-return]
            _MEMORY_STORE_GROUP,
            config.store_backend_name,
            "store_backend_name",
            config,
        )

    raise ValueError(f"Unknown store backend: {config.store_backend}")


def _create_embedder(config: MemoryConfig) -> Embedder:
    """Create or return a cached embedder backend.

    The embedder is shared across every ``LocalBackend`` instance that
    requests the same model, endpoint, credentials, and vector dimension. This
    matters for the per-project storage router, which can open many
    backends in the same process and must not pay the
    sentence-transformers / ONNX model-load cost more than once.

    ``config.embedder``, when set, wins over ``embedder_backend`` and is never
    cached: the caller built it and owns its lifecycle.

    For ``EmbedderBackend.EXTERNAL`` the backend comes from the
    :mod:`headroom.models.backends` registry. Its model comes only from
    ``embedder_backend_name`` (``"name:model"``) or ``embedder_options["model"]``;
    ``embedder_model`` is ignored, because it always holds a MiniLM default that
    would otherwise be sent to whichever backend was chosen.

    Only EXTERNAL and injected embedders are checked against
    ``vector_dimension`` up front. The built-in embedders keep their historical
    behaviour (Ollama, for one, reports 768 while the default index is 384) and
    are validated row by row instead.

    Args:
        config: Memory system configuration.

    Returns:
        An Embedder implementation based on config.embedder_backend.

    Raises:
        ValueError: If the embedder backend is not supported, an EXTERNAL or
            injected embedder's dimension differs from ``vector_dimension``,
            or the EXTERNAL model is given twice with different values.
        TypeError: ``config.embedder`` is not an embedder, backend or callable.
        headroom.models.backends.BackendError: An EXTERNAL backend is unknown
            or failed to construct.
    """
    if config.embedder is not None:
        return _injected_embedder(config)

    if config.embedder_backend == EmbedderBackend.EXTERNAL:
        return _external_embedder(config)

    # Reject missing credentials even when another caller has a cached client.
    if config.embedder_backend == EmbedderBackend.OPENAI and not config.openai_api_key:
        raise ValueError("openai_api_key is required for OpenAI embedder")

    key = (
        config.embedder_backend.value
        if hasattr(config.embedder_backend, "value")
        else str(config.embedder_backend),
        config.embedder_model or "",
        # The Ollama backend is built with ``base_url=config.ollama_base_url``,
        # so two configs that share a backend and model but point at different
        # Ollama servers must NOT share a cached embedder — otherwise the second
        # caller silently gets an embedder bound to the first server.
        config.ollama_base_url or "",
        config.openai_base_url or os.environ.get("OPENAI_BASE_URL", ""),
        config.openai_api_key or "",
        str(config.vector_dimension),
    )

    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is not None:
            return cached

        if config.embedder_backend == EmbedderBackend.LOCAL:
            from headroom.memory.adapters.embedders import LocalEmbedder

            embedder: Embedder = LocalEmbedder(model_name=config.embedder_model)

        elif config.embedder_backend == EmbedderBackend.ONNX:
            from headroom.memory.adapters.embedders import OnnxLocalEmbedder

            embedder = OnnxLocalEmbedder()

        elif config.embedder_backend == EmbedderBackend.OPENAI:
            from headroom.memory.adapters.embedders import OpenAIEmbedder

            embedder = OpenAIEmbedder(
                api_key=config.openai_api_key,
                model_name=config.embedder_model,
                base_url=config.openai_base_url,
                dimension=config.vector_dimension,
            )

        elif config.embedder_backend == EmbedderBackend.OLLAMA:
            from headroom.memory.adapters.embedders import OllamaEmbedder

            embedder = OllamaEmbedder(
                base_url=config.ollama_base_url,
                model_name=config.embedder_model,
            )
        else:
            raise ValueError(f"Unknown embedder backend: {config.embedder_backend}")

        _EMBEDDER_CACHE[key] = embedder
        return embedder


def embedder_is_shared(config: MemoryConfig, embedder: object) -> bool:
    """Whether ``embedder`` belongs to someone other than one memory system.

    True for the caller's own ``config.embedder`` and for any instance in the
    process-wide cache. A memory system built around a shared embedder must not
    close it: the proxy closes and rebuilds memory backends (after a failed or
    timed-out init, on shutdown of one project), and every other backend in the
    process still holds the same embedder.
    """
    if config.embedder is not None:
        return True
    with _EMBEDDER_CACHE_LOCK:
        return any(cached is embedder for cached in _EMBEDDER_CACHE.values())


def _is_async_embedder(obj: object) -> bool:
    from headroom.memory.ports import Embedder

    return (
        isinstance(obj, Embedder)
        and iscoroutinefunction(getattr(obj, "embed", None))
        and iscoroutinefunction(getattr(obj, "embed_batch", None))
    )


def _injected_embedder(config: MemoryConfig) -> Embedder:
    """Adapt ``config.embedder`` without caching it (the caller owns it)."""
    supplied = config.embedder
    if _is_async_embedder(supplied):
        embedder: Embedder = supplied
        declared = getattr(supplied, "dimension", None)
    else:
        if iscoroutinefunction(getattr(supplied, "embed", None)):
            raise TypeError(
                "An async embedder must implement the memory Embedder protocol: async "
                "embed() and embed_batch(), plus dimension, model_name and max_tokens"
            )
        from headroom.memory.adapters.backend_embedder import BackendEmbedder
        from headroom.models.backends import as_embedding_backend

        backend = as_embedding_backend(supplied)
        declared = backend.dimension
        embedder = BackendEmbedder(backend, dimension=config.vector_dimension)
    _check_declared_dimension(declared, config.vector_dimension, "The supplied embedder")
    return embedder


def _external_embedder(config: MemoryConfig) -> Embedder:
    """Build (or reuse) the registry backend selected by ``embedder_backend_name``."""
    name, model = external_embedder_selection(config)
    options = {
        key: value
        for key, value in config.embedder_options.items()
        if key not in ("model", "dimension")
    }
    key = (
        EmbedderBackend.EXTERNAL.value,
        name,
        model or "",
        str(config.vector_dimension),
        # Options carry endpoints and credentials: two configs that differ in
        # any of them must not share a client. Hashed, so the key never holds
        # a raw credential longer than the options themselves do.
        hashlib.sha256(
            json.dumps(options, sort_keys=True, default=repr).encode("utf-8")
        ).hexdigest(),
    )

    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is not None:
            return cached

        from headroom.memory.adapters.backend_embedder import BackendEmbedder
        from headroom.models.backends import create_embedding_backend

        backend = create_embedding_backend(
            name, **options, model=model, dimension=config.vector_dimension
        )
        try:
            _check_declared_dimension(
                backend.dimension, config.vector_dimension, f"Embedding backend {name!r}"
            )
        except ValueError:
            _close_quietly(backend)
            raise
        embedder: Embedder = BackendEmbedder(backend, dimension=config.vector_dimension)
        _EMBEDDER_CACHE[key] = embedder
        return embedder


def _close_quietly(backend: object) -> None:
    """Release a backend that is being discarded; its own close error is secondary."""
    close = getattr(backend, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        logger.debug("Closing a discarded embedding backend failed: %s", describe_exception(exc))


def _check_declared_dimension(declared: object, expected: int, what: str) -> None:
    """Refuse an embedder whose stated width cannot fill the vector index."""
    if isinstance(declared, int) and not isinstance(declared, bool) and declared != expected:
        raise ValueError(
            f"{what} produces {declared}-dimensional vectors, but the memory vector index "
            f"has vector_dimension={expected}. Set vector_dimension (proxy: "
            f"HEADROOM_EMBEDDER_DIMENSION) to the model's output width."
        )


def _reset_embedder_cache_for_tests() -> None:
    """Clear the process-wide embedder cache. Test-only seam."""

    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def _create_vector_index(config: MemoryConfig) -> VectorIndex:
    """Create a vector index backend.

    Args:
        config: Memory system configuration.

    Returns:
        A VectorIndex implementation based on config.vector_backend.

    Raises:
        ValueError: If the vector backend is not supported or unavailable.
    """
    backend = config.vector_backend

    if backend == VectorBackend.EXTERNAL:
        return _load_external_backend(  # type: ignore[no-any-return]
            _MEMORY_VECTOR_GROUP,
            config.vector_backend_name,
            "vector_backend_name",
            config,
        )

    # AUTO: prefer SQLITE_VEC → HNSW → fail with helpful message
    if backend == VectorBackend.AUTO:
        from headroom.memory.adapters import HNSW_AVAILABLE, SQLITE_VEC_AVAILABLE

        if SQLITE_VEC_AVAILABLE:
            backend = VectorBackend.SQLITE_VEC
        elif HNSW_AVAILABLE:
            backend = VectorBackend.HNSW
        else:
            raise ValueError(
                "No vector index backend available for memory. Install one:\n"
                "  pip install sqlite-vec   (recommended, lightweight)\n"
                "  pip install hnswlib      (alternative)\n"
                "Or install the full proxy bundle: pip install headroom-ai[proxy]"
            )

    if backend == VectorBackend.SQLITE_VEC:
        from headroom.memory.adapters import SQLITE_VEC_AVAILABLE

        if not SQLITE_VEC_AVAILABLE:
            raise ValueError(
                "sqlite-vec is not available. Install with: pip install sqlite-vec\n"
                "Or use vector_backend=VectorBackend.HNSW"
            )

        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        # Derive vector db path from main db path if not specified
        if config.vector_db_path:
            vector_db_path = config.vector_db_path
        else:
            # "memory.db" -> "memory_vectors.db"
            vector_db_path = config.db_path.parent / f"{config.db_path.stem}_vectors.db"

        return SQLiteVectorIndex(
            dimension=config.vector_dimension,
            db_path=vector_db_path,
            page_cache_size_kb=config.vector_cache_size_kb,
        )

    if backend == VectorBackend.HNSW:
        from headroom.memory.adapters import HNSW_AVAILABLE

        if not HNSW_AVAILABLE:
            raise ValueError(
                "hnswlib is not available. Install with: pip install hnswlib\n"
                "Or use vector_backend=VectorBackend.SQLITE_VEC"
            )

        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        # Derive persistent save path from the main DB path so the HNSW
        # index survives across process restarts (critical for cross-agent
        # interop: memories saved by Codex MCP must be searchable by Claude).
        hnsw_save_path: str | Path | None = None
        if config.db_path:
            hnsw_save_path = config.db_path.parent / f"{config.db_path.stem}_hnsw"

        return HNSWVectorIndex(
            dimension=config.vector_dimension,
            ef_construction=config.hnsw_ef_construction,
            m=config.hnsw_m,
            ef_search=config.hnsw_ef_search,
            max_entries=config.hnsw_max_entries,
            save_path=hnsw_save_path,
            auto_save=True,
        )

    raise ValueError(f"Unknown vector backend: {config.vector_backend}")


def _create_text_index(config: MemoryConfig) -> TextIndex:
    """Create a text index backend.

    Args:
        config: Memory system configuration.

    Returns:
        A TextIndex implementation based on config.text_backend.

    Raises:
        ValueError: If the text backend is not supported.
    """
    if config.text_backend == TextBackend.FTS5:
        from headroom.memory.adapters.fts5 import FTS5TextIndex

        # FTS5TextIndex has a compatible interface but different method signatures
        return FTS5TextIndex(db_path=config.db_path)  # type: ignore[return-value]

    if config.text_backend == TextBackend.EXTERNAL:
        return _load_external_backend(  # type: ignore[no-any-return]
            _MEMORY_TEXT_GROUP,
            config.text_backend_name,
            "text_backend_name",
            config,
        )

    raise ValueError(f"Unknown text backend: {config.text_backend}")


def _adapt_text_search(
    text_index: TextIndex,
) -> Callable[[TextFilter], Awaitable[list[TextSearchResult]]]:
    """Adapt built-in legacy search without changing the supplied index.

    FTS5 keeps its synchronous string API; external backends use the TextIndex
    search contract even when they have an unrelated ``search_memories`` helper.
    An FTS5 subclass with an async search override also uses that contract.
    Look up the method when called so construction does not require search for
    callers that only use indexing, and method replacements remain visible.
    """
    from headroom.memory.adapters.fts5 import FTS5TextIndex

    def search(filter: TextFilter) -> Awaitable[list[TextSearchResult]]:
        if isinstance(text_index, FTS5TextIndex) and not iscoroutinefunction(text_index.search):
            return text_index.search_memories(filter)
        return text_index.search(filter)

    return search


def _create_cache(config: MemoryConfig) -> MemoryCache:
    """Create a memory cache.

    Args:
        config: Memory system configuration.

    Returns:
        A MemoryCache implementation.
    """
    from headroom.memory.adapters.cache import LRUMemoryCache

    # LRUMemoryCache implements MemoryCache protocol
    return LRUMemoryCache(max_size=config.cache_max_size)  # type: ignore[return-value]
