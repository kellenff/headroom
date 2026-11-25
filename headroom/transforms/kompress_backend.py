"""Pluggable Kompress model backends: the contract, the registry and env selection.

Kompress is Headroom's ML text compressor. By default its model runs in-process
(:class:`~headroom.transforms.kompress_compressor.KompressCompressor`) or behind
a hosted ``/compress`` endpoint
(:class:`~headroom.transforms.kompress_remote.RemoteKompressCompressor`). A
Kompress *backend* replaces only the model: some other text compressor (a
hosted service, an in-house model, a different library) that maps text to
shorter text.

A backend supplies text in and text out, nothing else. Everything Headroom
promises around the model stays Headroom's, because the backend runs inside
:class:`~headroom.transforms.kompress_remote.ProviderKompressCompressor`, which
shares its post-processing with remote Kompress:

* blocks under the word floor (``KompressConfig.min_input_words``) are never sent;
* the request's ML budget is enforced: a call is declined when too little is
  left, capped to what is left otherwise, and passed through when it runs over;
* any failure passes the block through unchanged;
* the CCR store stays in this process, the retrieval marker is appended here,
  and a result that cannot pay for its marker passes through;
* ``ccr_original`` (the pre-tag-protection text) is what CCR stores.

Contract (:class:`KompressModelBackend`)
----------------------------------------
A backend is any object with:

* ``name: str``: the backend id. Used in messages only.
* ``compress_text(content, *, target_ratio, question)``: returns the compressed
  text as a ``str``, or a mapping with the fields of the remote endpoint's
  response: ``compressed`` (required, a non-empty ``str``) and optionally
  ``original_tokens`` / ``compressed_tokens`` (non-negative ``int``),
  ``compression_ratio`` (finite, non-negative number) and ``model_used``
  (``str``). A ``None`` optional field counts as absent.

  ``target_ratio`` is the fraction of the text to keep when the caller forces
  one (``None``: the backend decides). ``question`` is the user's question when
  the caller has one, for question-aware compression; most backends ignore it.

Optional: ``close() -> None``, called when the compressor holding the backend
is closed. It must be idempotent.

Guarantees a backend can rely on:

* ``content`` is a ``str`` with at least ``KompressConfig.min_input_words``
  (64) words.

Obligations a backend must meet:

* Thread-safety. Calls arrive concurrently from several threads.
* Determinism. The same input must give the same output: compressed blocks sit
  in the prompt prefix the provider caches, and Headroom caches results by
  content.
* Bounded latency. A call that outlives the request's ML budget is abandoned,
  not interrupted (Python cannot interrupt another thread). It keeps one of the
  compressor's 16 worker slots until it returns, and while every slot is held
  each new call waits for one until its own budget runs out, then passes
  through. Put a timeout on any I/O.
* Air-gap compliance. A backend that opens network connections must call
  :func:`headroom.offline.guard_egress` in its constructor and before each
  request. Headroom cannot enforce it for third-party code.
* Safe errors. Headroom logs only the class of an exception a backend or its
  factory raises, never its message (a :class:`BackendError` subclass
  included), so a message may not be the only place a failure is explained.

Headroom may construct a backend more than once (one per compression pipeline).
A backend that loads a large model in-process should share it between
instances. Construction should not depend on a remote service being up: a
factory that raises disables ML compression until the selection changes or the
process restarts.

Registering a backend
---------------------
Programmatically::

    from headroom.transforms.kompress_backend import register_kompress_backend

    register_kompress_backend("my-kompress", MyCompressor)  # MyCompressor(**options)

Or from a package, with no Headroom code changes::

    [project.entry-points."headroom.kompress_backend"]
    my-kompress = "my_pkg.kompress:MyCompressor"

Installing a package never changes behaviour; a backend runs only when
selected by name. There are no built-ins.

Selecting a backend (proxy operators)
-------------------------------------
=========================================  ===================================================
``HEADROOM_KOMPRESS_PROVIDER``             Registry name of the backend. Unset or blank:
                                           Kompress as before.
``HEADROOM_KOMPRESS_PROVIDER_OPTIONS``     JSON object of keyword options for the backend's
                                           factory.
=========================================  ===================================================

Not ``HEADROOM_KOMPRESS_BACKEND``: that one already picks the inference runtime
(ONNX or PyTorch) of the built-in model.

Precedence in :class:`~headroom.transforms.content_router.ContentRouter`:
Kompress disabled (config or the per-request ``kompress_model="disabled"``)
wins, then the provider, then ``HEADROOM_KOMPRESS_ENDPOINT``, then the local
model. With both the provider and the endpoint set the provider is used and a
WARNING says so. A provider that cannot be resolved or constructed disables
ML compression for that router, with one ERROR: falling back to the local
model would silently run a different compressor than the one configured.

Malformed values raise ``ValueError`` naming the variable, never its value,
because the options routinely carry credentials.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from ..log_safety import describe_class, describe_exception
from ..models.backends.registry import (
    NAME_RE,
    BackendError,
    BackendLoadError,
    BackendRegistry,
    UnknownBackendError,
)

logger = logging.getLogger(__name__)

KOMPRESS_ENTRY_POINT_GROUP = "headroom.kompress_backend"
#: Selects the backend by registry name.
KOMPRESS_PROVIDER_ENV = "HEADROOM_KOMPRESS_PROVIDER"
#: JSON object of keyword options for the selected backend's factory.
KOMPRESS_PROVIDER_OPTIONS_ENV = "HEADROOM_KOMPRESS_PROVIDER_OPTIONS"

# The optional response fields and the check each must pass. Integers exclude
# bool on purpose: ``True`` is an int to Python but never a token count.
_INT_FIELDS = ("original_tokens", "compressed_tokens")


class KompressBackendCallError(BackendError):
    """A backend's ``compress_text`` raised.

    The message names only the backend and the exception class; the original
    is chained. Every exception is wrapped, :class:`BackendError` subclasses
    included: a Kompress backend is always third-party code, so its message
    is never Headroom's to repeat.
    """


class KompressBackendResponseError(BackendError, ValueError):
    """A backend returned something that is not a valid compression result."""


@runtime_checkable
class KompressModelBackend(Protocol):
    """Text-in, text-out Kompress model. See the module docstring for the contract."""

    name: str

    def compress_text(
        self,
        content: str,
        *,
        target_ratio: float | None,
        question: str | None,
    ) -> str | Mapping[str, Any]:
        """Return the compressed text, or a mapping holding it under ``compressed``."""
        ...


#: The Kompress-backend registry. It has no built-ins.
KOMPRESS_BACKENDS: BackendRegistry[KompressModelBackend] = BackendRegistry(
    "kompress", KOMPRESS_ENTRY_POINT_GROUP
)


def register_kompress_backend(
    name: str,
    factory: Callable[..., KompressModelBackend] | str,
    *,
    replace: bool = False,
) -> None:
    """Register a Kompress backend factory under ``name``.

    ``factory`` is called with the options from
    ``HEADROOM_KOMPRESS_PROVIDER_OPTIONS`` as keywords and must return a
    :class:`KompressModelBackend`. A lazy ``"package.module:attr"`` string works
    too.
    """
    KOMPRESS_BACKENDS.register(name, factory, replace=replace)


def unregister_kompress_backend(name: str) -> bool:
    """Remove a registration made with :func:`register_kompress_backend`."""
    return KOMPRESS_BACKENDS.unregister(name)


def available_kompress_backends() -> list[str]:
    """Names of every selectable Kompress backend (entry points are not loaded)."""
    return KOMPRESS_BACKENDS.names()


def create_kompress_backend(name: str, /, **options: Any) -> KompressModelBackend:
    """Construct the Kompress backend registered under ``name`` with ``options``.

    Every exception the factory raises becomes a :class:`BackendLoadError`
    that names only its class, :class:`BackendError` subclasses included:
    there are no built-ins, so a factory is always third-party code.

    Raises:
        UnknownBackendError: No backend has that name.
        BackendLoadError: The factory failed to import or construct.
        TypeError: The factory returned something that is not a Kompress backend.
    """
    factory = KOMPRESS_BACKENDS.get_factory(name)
    try:
        backend = factory(**options)
    except Exception as exc:
        logger.debug("Kompress backend %r failed to initialise: %s", name, describe_exception(exc))
        raise BackendLoadError(
            f"kompress backend {name!r} failed to initialise ({describe_class(type(exc))})"
        ) from exc
    if not isinstance(backend, KompressModelBackend):
        raise TypeError(
            f"Kompress backend {name!r} returned {describe_class(type(backend))}, which does "
            "not implement KompressModelBackend (needs name and compress_text())"
        )
    return backend


# --------------------------------------------------------------------------- env


def kompress_provider_settings_from_env(
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Read ``HEADROOM_KOMPRESS_PROVIDER`` and ``HEADROOM_KOMPRESS_PROVIDER_OPTIONS``.

    Returns ``(name, options)``, or ``None`` when no provider is selected. The
    options variable is ignored when the provider variable is unset, like the
    companion variables of the embedding families.

    Raises:
        ValueError: A variable is malformed. The message names the variable,
            never its value.
    """
    env = os.environ if env is None else env
    raw = (env.get(KOMPRESS_PROVIDER_ENV) or "").strip()
    if not raw:
        return None
    name = raw.lower()
    if not NAME_RE.match(name):
        raise ValueError(
            f"{KOMPRESS_PROVIDER_ENV} must be a backend name: 1-64 lower-case letters, "
            "digits, '.', '_' or '-'; put backend settings in "
            f"{KOMPRESS_PROVIDER_OPTIONS_ENV}"
        )
    options: dict[str, Any] = {}
    raw_options = (env.get(KOMPRESS_PROVIDER_OPTIONS_ENV) or "").strip()
    if raw_options:
        try:
            parsed = json.loads(raw_options)
        except ValueError:
            raise ValueError(f"{KOMPRESS_PROVIDER_OPTIONS_ENV} must be a JSON object") from None
        if not isinstance(parsed, dict):
            raise ValueError(f"{KOMPRESS_PROVIDER_OPTIONS_ENV} must be a JSON object")
        options.update(parsed)
    return name, options


def validate_kompress_provider_env(env: Mapping[str, str] | None = None) -> str | None:
    """Check the env-selected provider resolves, without constructing it.

    For startup validation: an operator hears about a typo or a missing plugin
    when the proxy starts rather than as "compression stopped".

    Returns:
        The selected backend name, or ``None`` when no provider is selected.

    Raises:
        ValueError: A variable is malformed.
        UnknownBackendError: No backend has the selected name. The message names
            the variable, not its value, which may be a pasted credential.
        BackendLoadError: The backend's factory could not be imported.
    """
    settings = kompress_provider_settings_from_env(env)
    if settings is None:
        return None
    name, _options = settings
    resolve_kompress_provider(name)
    return name


def resolve_kompress_provider(name: str) -> None:
    """Resolve the factory for an env-selected ``name``, wording errors for the variable.

    Raises:
        UnknownBackendError: Nothing is registered under ``name``.
        BackendLoadError: The factory could not be imported.
    """
    try:
        KOMPRESS_BACKENDS.get_factory(name)
    except UnknownBackendError:
        raise UnknownBackendError(
            f"{KOMPRESS_PROVIDER_ENV} does not name a known Kompress backend. Available: "
            f"{', '.join(KOMPRESS_BACKENDS.names()) or '(none)'}. A plugin must be installed "
            f"and registered under '{KOMPRESS_BACKENDS.entry_point_group}'."
        ) from None


# --------------------------------------------------------------------------- calls


def call_kompress_backend(
    backend: KompressModelBackend,
    content: str,
    *,
    target_ratio: float | None,
    question: str | None,
    label: str,
) -> dict[str, Any]:
    """Call ``backend`` and return its result as the remote endpoint's response fields.

    The mapping always has ``compressed`` (a non-empty ``str``) and holds only
    the optional fields the backend set, each checked, so the caller can read
    it exactly like an endpoint's JSON.

    Args:
        label: Safe name for messages (the registry name it was selected by).

    Raises:
        KompressBackendCallError: ``compress_text`` raised.
        KompressBackendResponseError: The result is not a valid compression.
    """
    try:
        raw = backend.compress_text(content, target_ratio=target_ratio, question=question)
    except Exception as exc:  # BackendError too: its message is the plugin's, not ours
        logger.debug("Kompress backend %r failed: %s", label, describe_exception(exc))
        raise KompressBackendCallError(
            f"Kompress backend {label!r} failed ({describe_class(type(exc))})"
        ) from exc
    return _response_fields(raw, label)


def _response_fields(raw: object, label: str) -> dict[str, Any]:
    fields: dict[str, Any]
    if isinstance(raw, str):
        fields = {"compressed": raw}
    elif isinstance(raw, Mapping):
        fields = {key: value for key, value in raw.items() if value is not None}
    else:
        raise KompressBackendResponseError(
            f"Kompress backend {label!r} returned {describe_class(type(raw))}; "
            "compress_text must return a str or a mapping"
        )
    compressed = fields.get("compressed")
    if not isinstance(compressed, str):
        raise KompressBackendResponseError(
            f"Kompress backend {label!r} returned no string 'compressed' field"
        )
    # Every input has at least the word floor's words, so an empty result is a
    # broken backend, never a compression: accepting it would delete the block.
    if not compressed.strip():
        raise KompressBackendResponseError(f"Kompress backend {label!r} returned empty text")
    out: dict[str, Any] = {"compressed": compressed}
    for key in _INT_FIELDS:
        if key in fields:
            value = fields[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise KompressBackendResponseError(
                    f"Kompress backend {label!r} returned a {key!r} that is not a "
                    "non-negative integer"
                )
            out[key] = value
    if "compression_ratio" in fields:
        ratio = fields["compression_ratio"]
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio < 0
        ):
            raise KompressBackendResponseError(
                f"Kompress backend {label!r} returned a 'compression_ratio' that is not a "
                "finite, non-negative number"
            )
        out["compression_ratio"] = float(ratio)
    if "model_used" in fields:
        model_used = fields["model_used"]
        if not isinstance(model_used, str):
            raise KompressBackendResponseError(
                f"Kompress backend {label!r} returned a 'model_used' that is not a string"
            )
        out["model_used"] = model_used
    return out


__all__ = [
    "KOMPRESS_BACKENDS",
    "KOMPRESS_ENTRY_POINT_GROUP",
    "KOMPRESS_PROVIDER_ENV",
    "KOMPRESS_PROVIDER_OPTIONS_ENV",
    "KompressBackendCallError",
    "KompressBackendResponseError",
    "KompressModelBackend",
    "available_kompress_backends",
    "call_kompress_backend",
    "create_kompress_backend",
    "kompress_provider_settings_from_env",
    "register_kompress_backend",
    "resolve_kompress_provider",
    "unregister_kompress_backend",
    "validate_kompress_provider_env",
]
