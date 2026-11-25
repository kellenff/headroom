"""The memory embedder selected by ``HEADROOM_EMBEDDER_*``, as LocalBackend settings.

Every process that opens a memory database builds its own
:class:`~headroom.memory.backends.local.LocalBackendConfig`: the proxy, the
memory MCP server, the sync subprocess, ``headroom memory`` and
:class:`headroom.memory.Memory`. They must agree on the embedder, or the
database's embedder identity check refuses the odd one out. Reading the
variables in one place keeps them agreeing.

``HEADROOM_EMBEDDER_PROVIDER`` (or the older ``HEADROOM_EMBEDDER_RUNTIME=openai``
spelling) names a backend from :mod:`headroom.models.backends`; see
:func:`headroom.models.backends.memory_embedder_settings_from_env` for the
companion variables. Unset, every caller keeps its built-in default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from headroom.models.backends import (
    MEMORY_EMBEDDER_ENV_PREFIX,
    EmbeddingBackendSettings,
    memory_embedder_settings_from_env,
)

#: The variable that sizes the memory vector index for a configured provider.
MEMORY_EMBEDDER_DIMENSION_ENV = f"{MEMORY_EMBEDDER_ENV_PREFIX}_DIMENSION"


def validate_memory_embedder_env(
    env: Mapping[str, str] | None = None,
) -> EmbeddingBackendSettings | None:
    """Parse and check ``HEADROOM_EMBEDDER_*`` without constructing the backend.

    Cheap enough for startup: it resolves the backend name (importing its
    factory) but opens no connection and loads no model.

    Returns:
        The settings, or ``None`` when no provider is configured.

    Raises:
        ValueError: A variable is malformed, the variables contradict each
            other, or ``HEADROOM_EMBEDDER_DIMENSION`` is missing. Memory
            requires the dimension: it sizes the vector index, and guessing it
            would only defer the failure to the first write.
        headroom.models.backends.BackendError: The provider names no known
            backend, or its factory cannot be imported.
    """
    settings = memory_embedder_settings_from_env(env)
    if settings is None:
        return None
    if settings.dimension is None:
        raise ValueError(
            f"{MEMORY_EMBEDDER_DIMENSION_ENV} is required when {settings.source} selects the "
            "memory embedder: set it to the model's output width (for example 1536 for "
            "OpenAI text-embedding-3-small), which sizes the memory vector index"
        )
    settings.validate()
    return settings


def local_backend_embedder_kwargs(settings: EmbeddingBackendSettings | None) -> dict[str, Any]:
    """``LocalBackendConfig`` keyword arguments for already-validated ``settings``.

    Returns ``{}`` for ``None``, so callers keep their own default embedder.
    The dimension becomes ``vector_dimension`` (the factory passes it back to
    the backend); everything else, the model included, goes in
    ``embedder_options``.
    """
    if settings is None:
        return {}
    options = settings.backend_options()
    dimension = options.pop("dimension", None)
    if dimension is None:
        raise ValueError(f"{MEMORY_EMBEDDER_DIMENSION_ENV} is required for the memory embedder")
    return {
        "embedder_backend": "external",
        "embedder_backend_name": settings.name,
        "embedder_options": options,
        "vector_dimension": dimension,
    }


def local_backend_embedder_kwargs_from_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """``LocalBackendConfig`` keyword arguments for the ``HEADROOM_EMBEDDER_*`` embedder.

    ``{}`` when no provider is configured. Otherwise::

        {
            "embedder_backend": "external",
            "embedder_backend_name": "<provider>",
            "embedder_options": {...},  # model, base_url, api_key, ...; no dimension
            "vector_dimension": <HEADROOM_EMBEDDER_DIMENSION>,
        }

    Raises:
        ValueError: As :func:`validate_memory_embedder_env`.
        headroom.models.backends.BackendError: As :func:`validate_memory_embedder_env`.
    """
    return local_backend_embedder_kwargs(validate_memory_embedder_env(env))
