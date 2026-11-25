"""Environment-variable selection of embedding backends, for proxy operators.

Proxy operators have no Python surface, so each consumer reads one family of
variables under a prefix. Memory uses ``HEADROOM_EMBEDDER``; relevance scoring
uses ``HEADROOM_RELEVANCE_EMBEDDER``. For prefix ``P``:

=================  ==============================================================
``P_PROVIDER``     Registry name of the backend (``openai``, ``fastembed``, a
                   plugin's entry-point name). Unset or blank: the consumer's
                   built-in default. Must be a plain name; set the model with
                   ``P_MODEL``.
``P_MODEL``        Model id passed as the ``model`` option.
``P_BASE_URL``     ``base_url`` option (remote backends).
``P_API_KEY``      ``api_key`` option. Required for the built-in ``openai``
                   backend on api.openai.com (``P_BASE_URL`` unset): a selection
                   made here never falls back to ``OPENAI_API_KEY``, which in
                   the proxy is the chat credential.
``P_DIMENSION``    ``dimension`` option, a positive integer. Memory requires it.
``P_HEADERS``      ``headers`` option, as ``Name=value`` pairs separated by commas.
                   Values cannot contain commas; use ``P_OPTIONS`` for those.
``P_OPTIONS``      JSON object of further options. The dedicated variables above
                   win over the same keys here; ``model`` is not accepted here.
=================  ==============================================================

Two shortcuts keep the common cases to one variable:

* ``HEADROOM_EMBEDDER_RUNTIME=openai`` (the spelling from the first custom-API
  release) is an alias for ``HEADROOM_EMBEDDER_PROVIDER=openai``.
* ``HEADROOM_RELEVANCE_EMBEDDER_PROVIDER=memory`` reuses the memory embedder's
  ``HEADROOM_EMBEDDER_*`` settings, so one server and one key serve both.

Malformed values raise ``ValueError`` naming the variable, never its value.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .embedding import EMBEDDING_BACKENDS, EmbeddingBackend, create_embedding_backend
from .registry import NAME_RE, UnknownBackendError

#: Prefix for the memory embedder variables (``HEADROOM_EMBEDDER_PROVIDER`` ...).
MEMORY_EMBEDDER_ENV_PREFIX = "HEADROOM_EMBEDDER"
#: Prefix for the relevance-scoring variables (``HEADROOM_RELEVANCE_EMBEDDER_PROVIDER`` ...).
RELEVANCE_EMBEDDER_ENV_PREFIX = "HEADROOM_RELEVANCE_EMBEDDER"
#: ``HEADROOM_RELEVANCE_EMBEDDER_PROVIDER`` value that reuses the memory settings.
REUSE_MEMORY_EMBEDDER = "memory"
#: The memory runtime variable. ``openai`` is an alias for the ``openai`` provider;
#: its other values (``cpu``, ``pytorch_mps``) pick a device for the built-in model.
MEMORY_EMBEDDER_RUNTIME_ENV = "HEADROOM_EMBEDDER_RUNTIME"

_COMPANIONS = ("MODEL", "BASE_URL", "API_KEY", "DIMENSION", "HEADERS", "OPTIONS")
#: The built-in ``openai`` backend's default host, where it would otherwise take
#: ``OPENAI_API_KEY`` from the environment.
_OPENAI_API_HOST = "api.openai.com"


def parse_header_pairs(raw: str, var: str) -> dict[str, str]:
    """Parse ``Name=value,Name2=value2`` strictly.

    Raises:
        ValueError: A pair is malformed. The message names ``var`` only.
    """
    pairs: dict[str, str] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value or any(ch in key + value for ch in "\r\n\x00"):
            raise ValueError(f"{var} must hold Name=value pairs separated by commas")
        pairs[key] = value
    if not pairs:
        raise ValueError(f"{var} must hold Name=value pairs separated by commas")
    return pairs


@dataclass(frozen=True)
class EmbeddingBackendSettings:
    """An embedding backend selection read from the environment.

    Attributes:
        name: Registry name of the backend.
        model: Model id, or ``None`` for the backend's default.
        options: Further constructor options (``base_url``, ``api_key``, ...).
            Excluded from ``repr`` because they can hold credentials.
        source: The variable that selected the backend, for messages.
    """

    name: str
    model: str | None
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    source: str = ""

    @property
    def dimension(self) -> int | None:
        value = self.options.get("dimension")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def backend_options(self) -> dict[str, Any]:
        """Options to construct the backend with, ``model`` included when set."""
        options = dict(self.options)
        if self.model is not None:
            options["model"] = self.model
        return options

    def identity(self) -> str:
        """A stable, credential-free label for the embedding space: ``name:model``."""
        return f"{self.name}:{self.model or ''}"

    def validate(self) -> None:
        """Check the backend name resolves, without constructing it.

        Raises:
            UnknownBackendError: No backend has this name (the message names
                :attr:`source`, not the value).
            BackendLoadError: The backend's factory could not be imported.
        """
        try:
            EMBEDDING_BACKENDS.get_factory(self.name)
        except UnknownBackendError:
            raise UnknownBackendError(
                f"{self.source or 'The configured provider'} does not name a known embedding "
                f"backend. Available: {', '.join(EMBEDDING_BACKENDS.names()) or '(none)'}. "
                f"A plugin must be installed and registered under "
                f"'{EMBEDDING_BACKENDS.entry_point_group}'."
            ) from None

    def create(self) -> EmbeddingBackend:
        """Validate the name, then construct the backend."""
        self.validate()
        return create_embedding_backend(self.name, **self.backend_options())


def _positive_int(raw: str, var: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{var} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{var} must be a positive integer")
    return value


def _get(env: Mapping[str, str], var: str) -> str:
    return (env.get(var) or "").strip()


def _read_family(
    prefix: str, name: str, source: str, env: Mapping[str, str]
) -> EmbeddingBackendSettings:
    options: dict[str, Any] = {}
    options_var = f"{prefix}_OPTIONS"
    raw_options = _get(env, options_var)
    if raw_options:
        try:
            parsed = json.loads(raw_options)
        except ValueError:
            raise ValueError(f"{options_var} must be a JSON object") from None
        if not isinstance(parsed, dict):
            raise ValueError(f"{options_var} must be a JSON object")
        if "model" in parsed:
            raise ValueError(f"{options_var} must not set 'model'; use {prefix}_MODEL")
        options.update(parsed)

    model = _get(env, f"{prefix}_MODEL") or None
    if base_url := _get(env, f"{prefix}_BASE_URL"):
        options["base_url"] = base_url
    if api_key := _get(env, f"{prefix}_API_KEY"):
        options["api_key"] = api_key
    if dimension := _get(env, f"{prefix}_DIMENSION"):
        options["dimension"] = _positive_int(dimension, f"{prefix}_DIMENSION")
    if headers := _get(env, f"{prefix}_HEADERS"):
        options["headers"] = parse_header_pairs(headers, f"{prefix}_HEADERS")
    return EmbeddingBackendSettings(name=name, model=model, options=options, source=source)


def _require_own_openai_key(
    prefix: str, settings: EmbeddingBackendSettings | None
) -> EmbeddingBackendSettings | None:
    """Refuse an api.openai.com selection that has no key of its own.

    The built-in ``openai`` backend falls back to ``OPENAI_API_KEY`` for
    api.openai.com. In the proxy that variable is the chat credential, so an
    operator selection would silently embed and bill under it. Returns
    ``settings`` unchanged when they are fine.

    Raises:
        ValueError: Names ``{prefix}_API_KEY``, never a value.
    """
    if settings is None or settings.name != "openai" or not EMBEDDING_BACKENDS.is_builtin("openai"):
        return settings
    options = settings.options
    api_key = options.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or api_key.strip()):
        return settings  # set; a non-string is the backend's to reject
    base_url = options.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        try:
            host = urlsplit(base_url.strip()).hostname
        except ValueError:
            return settings  # malformed; the backend rejects it, naming base_url
        if host != _OPENAI_API_HOST:
            return settings
    elif base_url is not None and not isinstance(base_url, str):
        return settings  # the backend rejects it
    raise ValueError(
        f"{prefix}_API_KEY is required for the openai embedding backend on api.openai.com "
        f"(set {prefix}_BASE_URL for another server); OPENAI_API_KEY is not used for embeddings"
    )


def _provider_name(raw: str, var: str, prefix: str) -> str:
    name = raw.strip().lower()
    if ":" in name:
        raise ValueError(f"{var} must be a backend name only; set the model with {prefix}_MODEL")
    if not NAME_RE.match(name):
        raise ValueError(
            f"{var} must be a backend name: lower-case letters, digits, '.', '_' or '-'"
        )
    return name


def embedding_backend_settings_from_env(
    prefix: str,
    env: Mapping[str, str] | None = None,
) -> EmbeddingBackendSettings | None:
    """Read ``{prefix}_PROVIDER`` and its companion variables.

    Returns ``None`` when ``{prefix}_PROVIDER`` is unset or blank.

    Raises:
        ValueError: A variable is malformed, or the built-in ``openai`` backend
            is selected for api.openai.com without ``{prefix}_API_KEY``.
    """
    return _require_own_openai_key(prefix, _provider_settings(prefix, env))


def _provider_settings(
    prefix: str, env: Mapping[str, str] | None
) -> EmbeddingBackendSettings | None:
    env = os.environ if env is None else env
    var = f"{prefix}_PROVIDER"
    raw = _get(env, var)
    if not raw:
        return None
    return _read_family(prefix, _provider_name(raw, var, prefix), var, env)


def memory_embedder_settings_from_env(
    env: Mapping[str, str] | None = None,
) -> EmbeddingBackendSettings | None:
    """The memory embedder selected by ``HEADROOM_EMBEDDER_*``, or ``None`` for the default.

    ``HEADROOM_EMBEDDER_RUNTIME=openai`` selects the ``openai`` provider. Any
    other runtime value (``cpu``, ``pytorch_mps``) configures the built-in local
    embedder and cannot be combined with ``HEADROOM_EMBEDDER_PROVIDER``.

    Raises:
        ValueError: A variable is malformed, the variables contradict each
            other, or ``openai`` is selected for api.openai.com without
            ``HEADROOM_EMBEDDER_API_KEY``.
    """
    env = os.environ if env is None else env
    prefix = MEMORY_EMBEDDER_ENV_PREFIX
    runtime = _get(env, MEMORY_EMBEDDER_RUNTIME_ENV).lower()
    settings = _provider_settings(prefix, env)
    if runtime == "openai":
        if settings is not None and settings.name != "openai":
            raise ValueError(
                f"{MEMORY_EMBEDDER_RUNTIME_ENV}=openai contradicts {prefix}_PROVIDER; "
                f"set only {prefix}_PROVIDER"
            )
        if settings is None:
            settings = _read_family(prefix, "openai", f"{MEMORY_EMBEDDER_RUNTIME_ENV}=openai", env)
    elif settings is not None and runtime not in ("", "cpu"):
        raise ValueError(
            f"{MEMORY_EMBEDDER_RUNTIME_ENV} picks the device for the built-in embedder and "
            f"cannot be combined with {prefix}_PROVIDER; unset one of them"
        )
    return _require_own_openai_key(prefix, settings)


def relevance_embedder_settings_from_env(
    env: Mapping[str, str] | None = None,
) -> EmbeddingBackendSettings | None:
    """The relevance-scoring backend selected by ``HEADROOM_RELEVANCE_EMBEDDER_*``.

    ``HEADROOM_RELEVANCE_EMBEDDER_PROVIDER=memory`` returns the memory embedder's
    settings; its companion variables must then be unset.

    Raises:
        ValueError: A variable is malformed, or ``memory`` is selected while no
            memory provider is configured.
    """
    env = os.environ if env is None else env
    prefix = RELEVANCE_EMBEDDER_ENV_PREFIX
    var = f"{prefix}_PROVIDER"
    raw = _get(env, var)
    if raw.lower() != REUSE_MEMORY_EMBEDDER:
        return embedding_backend_settings_from_env(prefix, env)
    stray = [f"{prefix}_{name}" for name in _COMPANIONS if _get(env, f"{prefix}_{name}")]
    if stray:
        raise ValueError(
            f"{var}=memory reuses the HEADROOM_EMBEDDER_* settings; unset {', '.join(stray)}"
        )
    settings = memory_embedder_settings_from_env(env)
    if settings is None:
        raise ValueError(
            f"{var}=memory needs a memory embedder provider (HEADROOM_EMBEDDER_PROVIDER)"
        )
    return settings
