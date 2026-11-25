"""Generic, name-addressable registry for pluggable model backends.

One :class:`BackendRegistry` exists per backend *kind* (text embeddings, the
Kompress compression model, the image technique router, ...). Each kind has its
own contract and its own setuptools entry-point group, following the repo's
one-group-per-contract convention (``headroom.memory_store``,
``headroom.ccr_backend``, ``headroom.toin_backend``, ...).

A backend is produced by a *factory*: any callable that accepts keyword options
and returns an object implementing the kind's contract. A class works, since
calling it with ``**options`` constructs an instance. Factories come from three
places, resolved in this order:

1. Built-ins, registered by Headroom with :meth:`BackendRegistry.register_builtin`.
   Usually given as a lazy ``"package.module:attr"`` string, so a built-in's
   heavy dependencies are imported only when it is selected. An entry point can
   never shadow a built-in name.
2. Programmatic registrations, from :meth:`BackendRegistry.register`.
   ``replace=True`` may override a built-in on purpose.
3. Entry points in :attr:`BackendRegistry.entry_point_group`. Only the entry
   point with the requested name is loaded, and only when it is requested.
   Installing a package therefore never changes behaviour; a backend is used
   only when something selects it by name.

Selection failures are loud. An unknown name raises
:class:`UnknownBackendError`, a ``ValueError``. A factory that fails to import or
construct raises :class:`BackendLoadError`. Neither falls back to another
backend: for embeddings a silent fallback would mix vector spaces in persisted
indexes. :class:`~headroom.offline.OfflineEgressBlocked` is never wrapped. It is
a ``BaseException`` by design and passes through unchanged.

Error messages name the kind and the backend, never the options, because
options routinely carry API keys.
"""

from __future__ import annotations

import importlib
import logging
import re
import threading
from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Generic, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Valid backend names: lower-case, starting with a letter or digit, at most 64 chars.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

_LAZY_TARGET_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class BackendError(RuntimeError):
    """Base class for model-backend selection and construction failures."""


class UnknownBackendError(BackendError, ValueError):
    """No built-in, registration or entry point exists under the requested name."""


class BackendLoadError(BackendError):
    """A backend factory could not be imported or failed while constructing."""


def validate_backend_name(name: str) -> str:
    """Return ``name`` unchanged if it is a valid backend name, else raise ``ValueError``."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(
            f"Invalid backend name {name!r}: use 1-64 lower-case letters, digits, "
            "'.', '_' or '-', starting with a letter or digit."
        )
    return name


def parse_backend_spec(spec: str) -> tuple[str, str | None]:
    """Split ``"name"`` or ``"name:model"`` into ``(name, model)``.

    Only the first colon separates, so a model id may contain colons:
    ``"ollama:nomic-embed-text:latest"`` gives model ``"nomic-embed-text:latest"``.
    The name is stripped and lower-cased. The model is stripped, and an empty
    model becomes ``None``.

    Raises:
        ValueError: ``spec`` is empty, or its name part is not a valid name.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Backend spec must be a non-empty string such as 'openai' or 'name:model'")
    name, sep, model = spec.strip().partition(":")
    name = name.strip().lower()
    validate_backend_name(name)
    model_part = model.strip() if sep else ""
    return name, (model_part or None)


def _import_target(target: str) -> Any:
    module_name, _, attr_path = target.partition(":")
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


class BackendRegistry(Generic[T]):
    """Registry of factories for one kind of model backend.

    Args:
        kind: Human-readable kind, used in messages (``"embedding"``).
        entry_point_group: The setuptools entry-point group third-party
            packages register under (``"headroom.embedding_backend"``).

    Thread-safe: every mutation and lazy load happens under one lock.
    """

    def __init__(self, kind: str, entry_point_group: str) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._lock = threading.RLock()
        self._builtins: dict[str, Callable[..., T] | str] = {}
        self._registered: dict[str, Callable[..., T] | str] = {}
        self._loaded: dict[str, Callable[..., T]] = {}

    # ------------------------------------------------------------------ registration

    def register_builtin(self, name: str, factory: Callable[..., T] | str) -> None:
        """Register a Headroom built-in. Entry points can never shadow it."""
        validate_backend_name(name)
        self._check_factory(factory)
        with self._lock:
            self._builtins[name] = factory
            self._loaded.pop(name, None)

    def register(
        self,
        name: str,
        factory: Callable[..., T] | str,
        *,
        replace: bool = False,
    ) -> None:
        """Register a backend factory under ``name``.

        Args:
            name: The name callers select the backend by.
            factory: A callable accepting keyword options and returning a
                backend, or a lazy ``"package.module:attr"`` import string.
            replace: Allow replacing an existing registration or built-in.

        Raises:
            ValueError: The name is invalid, or already taken and ``replace``
                is false.
        """
        validate_backend_name(name)
        self._check_factory(factory)
        with self._lock:
            taken = name in self._registered or name in self._builtins
            if taken and not replace:
                raise ValueError(
                    f"A {self.kind} backend named {name!r} is already registered; "
                    "pass replace=True to override it."
                )
            self._registered[name] = factory
            self._loaded.pop(name, None)

    def unregister(self, name: str) -> bool:
        """Remove a programmatic registration. Returns whether one existed.

        Built-ins cannot be removed; unregistering a name that only overrode a
        built-in restores the built-in.
        """
        with self._lock:
            removed = self._registered.pop(name, None) is not None
            if removed:
                self._loaded.pop(name, None)
            return removed

    # ------------------------------------------------------------------ lookup

    def _entry_points(self) -> list[EntryPoint]:
        try:
            return list(entry_points(group=self.entry_point_group))
        except Exception as exc:  # noqa: BLE001 - broken metadata must not break lookup
            logger.warning(
                "Could not enumerate %s entry points: %s",
                self.entry_point_group,
                type(exc).__name__,
            )
            return []

    def names(self) -> list[str]:
        """Every selectable name: built-ins, registrations and entry points.

        Entry points are listed, not loaded.
        """
        with self._lock:
            names = set(self._builtins) | set(self._registered)
        for ep in self._entry_points():
            if NAME_RE.match(ep.name):
                names.add(ep.name)
        return sorted(names)

    def is_registered(self, name: str) -> bool:
        """Whether ``name`` is selectable, without loading anything."""
        return name in self.names()

    def get_factory(self, name: str) -> Callable[..., T]:
        """Return the factory for ``name``, importing it lazily if needed.

        Raises:
            UnknownBackendError: Nothing is selectable under ``name``.
            BackendLoadError: The factory could not be imported.
        """
        with self._lock:
            cached = self._loaded.get(name)
            if cached is not None:
                return cached
            source: Callable[..., T] | str | None = self._registered.get(name)
            if source is None:
                source = self._builtins.get(name)
            if source is not None:
                factory = self._resolve(name, source)
                self._loaded[name] = factory
                return factory

        ep = self._find_entry_point(name)
        if ep is None:
            raise UnknownBackendError(
                f"Unknown {self.kind} backend {name!r}. Available: "
                f"{', '.join(self.names()) or '(none)'}. A third-party backend must be "
                f"installed and registered under the '{self.entry_point_group}' entry-point group."
            )
        try:
            loaded = ep.load()
        except Exception as exc:
            raise BackendLoadError(
                f"Could not load {self.kind} backend {name!r} from entry point "
                f"{ep.value!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if not callable(loaded):
            raise BackendLoadError(
                f"Entry point {name!r} in '{self.entry_point_group}' is not callable "
                f"(got {type(loaded).__name__}); it must be a backend class or factory."
            )
        factory = cast("Callable[..., T]", loaded)
        with self._lock:
            # A registration may have raced us while the entry point was loading.
            return self._loaded.setdefault(name, factory)

    def create(self, name: str, /, **options: Any) -> T:
        """Construct the backend registered under ``name`` with ``options``.

        Raises:
            UnknownBackendError: Nothing is selectable under ``name``.
            BackendLoadError: The factory raised. The original exception is
                chained; the message never includes ``options``.
        """
        factory = self.get_factory(name)
        try:
            return factory(**options)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendLoadError(
                f"{self.kind} backend {name!r} failed to initialise: {type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ internals

    def _find_entry_point(self, name: str) -> EntryPoint | None:
        matches = [ep for ep in self._entry_points() if ep.name == name]
        if not matches:
            return None
        if len(matches) > 1:
            logger.warning(
                "%d packages register a %s backend named %r under %s; using %s.",
                len(matches),
                self.kind,
                name,
                self.entry_point_group,
                matches[0].value,
            )
        return matches[0]

    def _resolve(self, name: str, source: Callable[..., T] | str) -> Callable[..., T]:
        if not isinstance(source, str):
            return source
        try:
            target = _import_target(source)
        except Exception as exc:
            raise BackendLoadError(
                f"Could not import {self.kind} backend {name!r} ({source}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(target):
            raise BackendLoadError(f"{self.kind} backend {name!r} ({source}) is not callable")
        return cast("Callable[..., T]", target)

    @staticmethod
    def _check_factory(factory: object) -> None:
        if isinstance(factory, str):
            if not _LAZY_TARGET_RE.match(factory):
                raise ValueError(
                    f"Lazy backend factory {factory!r} must look like 'package.module:attr'."
                )
            return
        if not callable(factory):
            raise TypeError(
                f"Backend factory must be callable or a 'module:attr' string, got {type(factory).__name__}"
            )

    def _reset_for_tests(self) -> None:
        """Drop programmatic registrations and loaded factories; keep built-ins."""
        with self._lock:
            self._registered.clear()
            self._loaded.clear()
