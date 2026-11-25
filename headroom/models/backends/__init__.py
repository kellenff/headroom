"""Pluggable model backends.

Headroom runs several models of its own: text embeddings for memory, relevance
scoring and the semantic cache, the Kompress compression model, and the image
technique router. This package lets each be replaced by a backend you choose: a
built-in (for example an OpenAI-compatible embedding server), a function or
object you pass in, or a third-party package registered through an entry point.

Each kind of backend has its own contract, registry and entry-point group:

=================  ===============================  ====================================
Kind               Entry-point group                Contract
=================  ===============================  ====================================
text embeddings    ``headroom.embedding_backend``   :class:`EmbeddingBackend`
=================  ===============================  ====================================

Backends are selected by explicit name only; installing a package never changes
behaviour. See ``docs/content/docs/model-backends.mdx``.
"""

from __future__ import annotations

from .embedding import (
    EMBEDDING_BACKENDS,
    EMBEDDING_ENTRY_POINT_GROUP,
    CallableEmbeddingBackend,
    EmbeddingBackend,
    EmbeddingBackendResponseError,
    as_embedding_backend,
    available_embedding_backends,
    create_embedding_backend,
    embed_texts,
    register_embedding_backend,
    unregister_embedding_backend,
)
from .env import (
    MEMORY_EMBEDDER_ENV_PREFIX,
    RELEVANCE_EMBEDDING_ENV_PREFIX,
    EmbeddingBackendSettings,
    embedding_backend_settings_from_env,
)
from .registry import (
    BackendError,
    BackendLoadError,
    BackendRegistry,
    UnknownBackendError,
    parse_backend_spec,
)

__all__ = [
    # generic
    "BackendError",
    "BackendLoadError",
    "BackendRegistry",
    "UnknownBackendError",
    "parse_backend_spec",
    # embeddings
    "EMBEDDING_BACKENDS",
    "EMBEDDING_ENTRY_POINT_GROUP",
    "CallableEmbeddingBackend",
    "EmbeddingBackend",
    "EmbeddingBackendResponseError",
    "as_embedding_backend",
    "available_embedding_backends",
    "create_embedding_backend",
    "embed_texts",
    "register_embedding_backend",
    "unregister_embedding_backend",
    # env
    "MEMORY_EMBEDDER_ENV_PREFIX",
    "RELEVANCE_EMBEDDING_ENV_PREFIX",
    "EmbeddingBackendSettings",
    "embedding_backend_settings_from_env",
]
