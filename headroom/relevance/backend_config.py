"""Turn ``RelevanceScorerConfig.embedding_backend`` into a backend.

The config holds a string, never a backend instance: the router hashes its
config into the disk cache's namespace, which must read the same in every
process. Two kinds of value are accepted:

* A registry spec, ``"name"`` or ``"name:model"``, constructed with no further
  options. For backends that need some (a URL, a key), register a factory with
  them bound, or pass an instance to ``create_scorer(embedding_backend=...)``.
* ``"env"``: the backend the ``HEADROOM_RELEVANCE_EMBEDDER_*`` variables select
  (see :func:`~headroom.models.backends.relevance_embedder_settings_from_env`),
  read when the scorer is built. Nothing selected means the default fastembed
  scorer. This is how the proxy passes an operator's choice through config;
  the library never reads these variables unless its config says ``"env"``.

The built-in ``openai`` backend is built here with request-path defaults, a
5 s timeout and no retries, unless the settings set either. Relevance runs
inside compression on the request thread, where the backend's own defaults (a
30 s timeout, two retries) could stall one request for well over a minute.
Spec strings given to ``create_scorer(embedding_backend=...)`` and
``EmbeddingScorer(backend=...)`` are built by
:func:`create_relevance_embedding_backend` too, so a spec gets the same bounds
whichever entry point it came through.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from headroom.models.backends import (
    EMBEDDING_BACKENDS,
    EmbeddingBackend,
    create_embedding_backend,
    parse_backend_spec,
    relevance_embedder_settings_from_env,
)

__all__ = [
    "EMBEDDING_BACKEND_FROM_ENV",
    "RELEVANCE_BACKEND_MAX_RECORDS",
    "RELEVANCE_OPENAI_MAX_RETRIES",
    "RELEVANCE_OPENAI_TIMEOUT_SECONDS",
    "create_relevance_embedding_backend",
    "describe_relevance_embedding_backend",
    "resolve_relevance_embedding_backend",
    "validate_relevance_embedding_backend_value",
]

#: ``RelevanceScorerConfig.embedding_backend`` value that defers to the
#: ``HEADROOM_RELEVANCE_EMBEDDER_*`` variables.
EMBEDDING_BACKEND_FROM_ENV = "env"

#: Request-path defaults for the built-in ``openai`` backend.
RELEVANCE_OPENAI_TIMEOUT_SECONDS = 5.0
RELEVANCE_OPENAI_MAX_RETRIES = 0

#: Record cap for the router's relevance split while a configured backend
#: scores it and ``ContentRouterConfig.relevance_max_records`` is 0 (no cap).
#: An output segmenting into more records is not split, so a remote backend
#: is never asked to embed an unbounded tool output per block: at the
#: ``openai`` backend's default batch size (256) one split is at most three
#: requests. Set ``relevance_max_records`` to choose another bound.
RELEVANCE_BACKEND_MAX_RECORDS = 512


def _is_env(value: str) -> bool:
    return value.strip().lower() == EMBEDDING_BACKEND_FROM_ENV


def validate_relevance_embedding_backend_value(value: object) -> None:
    """Check a ``RelevanceScorerConfig.embedding_backend`` value without constructing it.

    Only the shape is checked (a string; ``"env"`` or a well-formed spec):
    whether the name resolves is known only once plugins are importable,
    which is when :func:`resolve_relevance_embedding_backend` runs.

    Raises:
        TypeError: ``value`` is not a string or ``None``.
        ValueError: ``value`` is not ``"env"`` or a valid ``"name[:model]"`` spec.
    """
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(
            "embedding_backend must be a backend spec string ('name' or 'name:model') or "
            f"'{EMBEDDING_BACKEND_FROM_ENV}', got {type(value).__name__}. Pass a backend "
            "instance to headroom.relevance.create_scorer(embedding_backend=...) instead."
        )
    if _is_env(value):
        return
    name, model = parse_backend_spec(value)
    if name == EMBEDDING_BACKEND_FROM_ENV and model is not None:
        raise ValueError(
            f"embedding_backend '{EMBEDDING_BACKEND_FROM_ENV}' takes no model; set "
            "HEADROOM_RELEVANCE_EMBEDDER_MODEL instead"
        )


def describe_relevance_embedding_backend(value: str | None) -> str:
    """A label for ``value`` that is safe in any log line; never raises, reads no env.

    The backend name for a spec (its model part is left out), the variable
    family for ``"env"``, ``"fastembed (default)"`` for ``None``.
    """
    if value is None:
        return "fastembed (default)"
    if not isinstance(value, str):
        return "<invalid>"
    if _is_env(value):
        return "HEADROOM_RELEVANCE_EMBEDDER_*"
    try:
        return parse_backend_spec(value)[0]
    except ValueError:
        return "<invalid>"


def _request_path_options(name: str, options: dict[str, Any]) -> dict[str, Any]:
    """Bound the built-in ``openai`` backend's latency unless told otherwise.

    Only the built-in: a factory registered under the same name with
    ``replace=True`` may not take these options.
    """
    if name == "openai" and EMBEDDING_BACKENDS.is_builtin("openai"):
        options.setdefault("timeout", RELEVANCE_OPENAI_TIMEOUT_SECONDS)
        options.setdefault("max_retries", RELEVANCE_OPENAI_MAX_RETRIES)
    return options


def create_relevance_embedding_backend(spec: str) -> EmbeddingBackend:
    """Construct the backend a ``"name"`` or ``"name:model"`` spec names, for relevance.

    :func:`~headroom.models.backends.create_embedding_backend` with the
    request-path defaults applied: the built-in ``openai`` backend gets a 5 s
    timeout and no retries. Every relevance entry point that takes a spec
    string builds it here. To choose other bounds, construct the backend
    yourself and pass the instance.

    Raises:
        ValueError: ``spec`` is malformed (``UnknownBackendError`` is one: no
            backend has that name).
        BackendLoadError: The backend failed to import or construct.
        TypeError: The factory returned something that is not an embedding backend.
        OfflineEgressBlocked: ``HEADROOM_OFFLINE`` is set and the backend dials
            out (a ``BaseException``; catch it explicitly).
    """
    name, _model = parse_backend_spec(spec)
    return create_embedding_backend(spec, **_request_path_options(name, {}))


def resolve_relevance_embedding_backend(
    value: str | None, env: Mapping[str, str] | None = None
) -> EmbeddingBackend | None:
    """Construct the relevance embedding backend ``value`` selects.

    Args:
        value: ``RelevanceScorerConfig.embedding_backend``: ``None``, ``"env"``
            or a registry spec.
        env: The environment ``"env"`` reads; ``os.environ`` when ``None``.
            Ignored for any other value.

    Returns:
        The backend, or ``None`` when nothing is selected (``value`` is
        ``None``, or ``"env"`` with ``HEADROOM_RELEVANCE_EMBEDDER_PROVIDER``
        unset): the caller keeps the default fastembed scorer.

    Raises:
        ValueError: The value or a variable is malformed (``UnknownBackendError``
            is one: no backend has that name; it names the variable for ``"env"``).
        BackendLoadError: The backend failed to import or construct.
        OfflineEgressBlocked: ``HEADROOM_OFFLINE`` is set and the backend dials
            out (a ``BaseException``; catch it explicitly).
    """
    validate_relevance_embedding_backend_value(value)
    if value is None:
        return None
    if _is_env(value):
        settings = relevance_embedder_settings_from_env(env)
        if settings is None:
            return None
        settings.validate()
        options = _request_path_options(settings.name, settings.backend_options())
        return create_embedding_backend(settings.name, **options)
    return create_relevance_embedding_backend(value)
