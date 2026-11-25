"""Environment-variable configuration for model backends.

Proxy operators have no Python surface, so each consumer reads one family of
variables under a prefix. For prefix ``HEADROOM_RELEVANCE_EMBEDDING``:

==========================================  ==========================================
``HEADROOM_RELEVANCE_EMBEDDING_BACKEND``    Backend spec, ``name`` or ``name:model``.
                                            Unset or blank: the consumer's default.
``HEADROOM_RELEVANCE_EMBEDDING_MODEL``      Model id (overrides a model in the spec).
``HEADROOM_RELEVANCE_EMBEDDING_BASE_URL``   ``base_url`` option (remote backends).
``HEADROOM_RELEVANCE_EMBEDDING_API_KEY``    ``api_key`` option.
``HEADROOM_RELEVANCE_EMBEDDING_DIMENSION``  ``dimension`` option, a positive integer.
``HEADROOM_RELEVANCE_EMBEDDING_HEADERS``    ``headers`` option, ``k=v,k2=v2``.
``HEADROOM_RELEVANCE_EMBEDDING_OPTIONS``    JSON object of any further options. The
                                            dedicated variables above win over keys
                                            repeated here.
==========================================  ==========================================

Values that fail to parse raise ``ValueError`` naming the variable; the values
of ``_API_KEY``, ``_HEADERS`` and ``_OPTIONS`` are never echoed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .embedding import EmbeddingBackend, create_embedding_backend
from .registry import parse_backend_spec

#: Prefix for the memory embedder variables (``HEADROOM_EMBEDDER_BACKEND`` ...).
MEMORY_EMBEDDER_ENV_PREFIX = "HEADROOM_EMBEDDER"
#: Prefix for the relevance-scoring variables (``HEADROOM_RELEVANCE_EMBEDDING_BACKEND`` ...).
RELEVANCE_EMBEDDING_ENV_PREFIX = "HEADROOM_RELEVANCE_EMBEDDING"


def parse_header_pairs(raw: str | None) -> dict[str, str]:
    """Parse ``k=v,k2=v2`` into a dict.

    Same format as ``HEADROOM_KOMPRESS_ENDPOINT_HEADERS``. Malformed pairs are skipped.
    """
    pairs: dict[str, str] = {}
    for item in (raw or "").split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            pairs[key] = value
    return pairs


@dataclass(frozen=True)
class EmbeddingBackendSettings:
    """An embedding backend selection read from the environment.

    Attributes:
        name: Registry name of the backend.
        model: Model id, or ``None`` for the backend's default.
        options: Further constructor options (``base_url``, ``api_key``, ...).
            Excluded from ``repr`` because they can hold credentials.
        source: The ``*_BACKEND`` variable this came from, for messages.
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

    def create(self) -> EmbeddingBackend:
        """Construct the backend."""
        return create_embedding_backend(self.name, **self.backend_options())


def _positive_int(raw: str, var: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{var} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{var} must be a positive integer, got {raw!r}")
    return value


def embedding_backend_settings_from_env(
    prefix: str,
    env: Mapping[str, str] | None = None,
) -> EmbeddingBackendSettings | None:
    """Read ``{prefix}_BACKEND`` and its companion variables.

    Returns ``None`` when ``{prefix}_BACKEND`` is unset or blank.

    Raises:
        ValueError: A variable is malformed.
    """
    env = os.environ if env is None else env
    spec_var = f"{prefix}_BACKEND"
    spec = (env.get(spec_var) or "").strip()
    if not spec:
        return None
    try:
        name, model = parse_backend_spec(spec)
    except ValueError as exc:
        raise ValueError(f"{spec_var}: {exc}") from None

    options: dict[str, Any] = {}
    options_var = f"{prefix}_OPTIONS"
    raw_options = (env.get(options_var) or "").strip()
    if raw_options:
        try:
            parsed = json.loads(raw_options)
        except ValueError:
            raise ValueError(f"{options_var} must be a JSON object") from None
        if not isinstance(parsed, dict):
            raise ValueError(f"{options_var} must be a JSON object")
        options.update(parsed)
    if "model" in options:
        option_model = options.pop("model")
        if model is None and isinstance(option_model, str) and option_model.strip():
            model = option_model.strip()

    model_override = (env.get(f"{prefix}_MODEL") or "").strip()
    if model_override:
        model = model_override

    base_url = (env.get(f"{prefix}_BASE_URL") or "").strip()
    if base_url:
        options["base_url"] = base_url
    api_key = (env.get(f"{prefix}_API_KEY") or "").strip()
    if api_key:
        options["api_key"] = api_key
    dimension = (env.get(f"{prefix}_DIMENSION") or "").strip()
    if dimension:
        options["dimension"] = _positive_int(dimension, f"{prefix}_DIMENSION")
    headers_var = f"{prefix}_HEADERS"
    if (env.get(headers_var) or "").strip():
        headers = parse_header_pairs(env.get(headers_var))
        if not headers:
            raise ValueError(f"{headers_var} must hold k=v pairs separated by commas")
        options["headers"] = headers

    return EmbeddingBackendSettings(name=name, model=model, options=options, source=spec_var)
