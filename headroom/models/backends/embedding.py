"""Text-embedding backends: the contract, the registry and the shared helpers.

Every Python consumer of text embeddings (memory, relevance scoring, the
semantic cache) can take a backend from here instead of its hard-wired model.

Contract (:class:`EmbeddingBackend`)
------------------------------------
A backend is any object with:

* ``name: str``: the backend id (``"openai"``, ``"fastembed"``, ``"my-plugin"``).
* ``model_name: str``: the model it serves.
* ``dimension: int | None``: the output width if known up front, else ``None``.
* ``embed(texts: list[str]) -> array-like``: one row of floats per input text,
  in input order, all the same width. A ``numpy.ndarray`` of shape
  ``(len(texts), d)`` or a list of lists both work. Rows need not be normalised.

Optional attributes, read with ``getattr``: ``max_batch_size: int | None``
(``embed_texts`` chunks larger requests), ``max_tokens: int | None`` (the model's
input window) and ``close() -> None``.

Guarantees a backend can rely on, because every consumer calls it through
:func:`embed_texts`:

* ``texts`` is non-empty and every text has non-whitespace content. Empty texts
  never reach the backend; their rows come back as zeros.
* At most ``max_batch_size`` texts per call when that attribute is set.

Obligations a backend must meet:

* Thread-safety. Consumers call ``embed`` from executor and worker threads,
  sometimes concurrently.
* Determinism for a given model. Memory persists the vectors and relevance
  thresholds assume a stable score distribution.
* Air-gap compliance. A backend that opens network connections must call
  :func:`headroom.offline.guard_egress` in its constructor, so an air-gapped
  process refuses at startup, and again before each request. The built-in
  ``openai`` backend does; Headroom cannot enforce it for third-party code.
* Recoverable ``close()``. Headroom shares one backend between many users and
  may call ``close()`` while others still hold it. ``close()`` must be
  idempotent and a later ``embed`` must work (re-open lazily).
* Safe errors. Headroom logs only the class of an exception raised by a
  third-party backend, not its message, unless ``HEADROOM_DEBUG_DUMP=full``.

Factories
---------
A backend is built by a factory: a class or function called with keyword
options. Every factory must accept the keywords ``model`` (``str | None``) and
``dimension`` (``int | None``), even if it ignores them: Headroom passes the
model the operator chose and, for memory, the vector index's dimension. Other
options are whatever the factory documents (``base_url``, ``api_key``, ...).

Registering a backend
---------------------
Programmatically::

    from headroom.models.backends import register_embedding_backend

    register_embedding_backend("my-embedder", MyEmbedder)  # MyEmbedder(**options)

Or from a package, with no Headroom code changes::

    [project.entry-points."headroom.embedding_backend"]
    my-embedder = "my_pkg.embed:MyEmbedder"

A plain function works too: pass ``fn(list[str]) -> rows`` wherever a backend is
accepted and it is wrapped in :class:`CallableEmbeddingBackend`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from headroom.log_safety import describe_class, describe_exception

from .registry import NAME_RE as _SAFE_LABEL_RE
from .registry import BackendError, BackendRegistry, parse_backend_spec

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_ENTRY_POINT_GROUP = "headroom.embedding_backend"


class EmbeddingBackendResponseError(BackendError, ValueError):
    """A backend returned rows that do not match the request (count, width, finiteness)."""


class EmbeddingBackendCallError(BackendError):
    """A backend's ``embed`` raised.

    For third-party backends the message names only the exception class; the
    original is chained. Errors that are already :class:`BackendError`
    subclasses (raised by Headroom's own built-ins, with messages written to be
    safe to log) pass through unchanged.
    """


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Synchronous text-embedding backend. See the module docstring for the contract."""

    name: str
    model_name: str
    dimension: int | None

    def embed(self, texts: list[str]) -> Any:
        """Return one embedding row per text, in order."""
        ...


#: The embedding-backend registry. Built-ins are registered at import, lazily.
EMBEDDING_BACKENDS: BackendRegistry[EmbeddingBackend] = BackendRegistry(
    "embedding", EMBEDDING_ENTRY_POINT_GROUP
)
EMBEDDING_BACKENDS.register_builtin(
    "openai", "headroom.models.backends.openai_compat:OpenAICompatibleEmbeddingBackend"
)
EMBEDDING_BACKENDS.register_builtin("fastembed", "headroom.relevance.embedding:FastEmbedBackend")
EMBEDDING_BACKENDS.register_builtin(
    "sentence-transformers",
    "headroom.models.backends.sentence_transformer_backend:SentenceTransformerBackend",
)


def register_embedding_backend(
    name: str,
    factory: Callable[..., EmbeddingBackend] | str,
    *,
    replace: bool = False,
) -> None:
    """Register an embedding backend factory under ``name``.

    ``factory`` is called with keyword options (``model``, ``dimension``, plus
    whatever the caller configured) and must return an :class:`EmbeddingBackend`.
    """
    EMBEDDING_BACKENDS.register(name, factory, replace=replace)


def unregister_embedding_backend(name: str) -> bool:
    """Remove a programmatic registration made with :func:`register_embedding_backend`."""
    return EMBEDDING_BACKENDS.unregister(name)


def available_embedding_backends() -> list[str]:
    """Names of every selectable embedding backend (entry points are not loaded)."""
    return EMBEDDING_BACKENDS.names()


def create_embedding_backend(spec: str, /, **options: Any) -> EmbeddingBackend:
    """Construct an embedding backend from ``"name"`` or ``"name:model"``.

    A model in the spec is passed as the ``model`` option unless ``options``
    already has one.

    Raises:
        UnknownBackendError: No backend has that name.
        BackendLoadError: The backend failed to construct.
        TypeError: The factory returned something that is not an embedding backend.
    """
    name, model = parse_backend_spec(spec)
    if model is not None and options.get("model") is None:
        options["model"] = model
    backend = EMBEDDING_BACKENDS.create(name, **options)
    _check_conforms(backend, f"embedding backend {name!r}")
    return backend


def as_embedding_backend(obj: object, /, **options: Any) -> EmbeddingBackend:
    """Coerce a backend, a spec string or a plain function into a backend.

    * An :class:`EmbeddingBackend` is returned as is. ``options`` must then be
      empty.
    * A string is a spec for :func:`create_embedding_backend`.
    * Any other callable is wrapped as ``CallableEmbeddingBackend(obj, **options)``.

    Raises:
        TypeError: ``obj`` is none of these, or options were given for an instance.
    """
    if isinstance(obj, str):
        return create_embedding_backend(obj, **options)
    if isinstance(obj, EmbeddingBackend):
        if options:
            raise TypeError("options cannot be applied to an already-constructed backend")
        return obj
    if callable(obj):
        return CallableEmbeddingBackend(obj, **options)
    raise TypeError(
        "Expected an EmbeddingBackend, a backend spec string or a callable "
        f"fn(list[str]) -> rows, got {type(obj).__name__}"
    )


def _check_conforms(backend: object, what: str) -> None:
    if not isinstance(backend, EmbeddingBackend):
        raise TypeError(
            f"{what} returned {type(backend).__name__}, which does not implement "
            "EmbeddingBackend (needs name, model_name, dimension and embed())"
        )


def _positive_int_or_none(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return int(value)


def _backend_label(backend: object) -> str:
    name = getattr(backend, "name", None)
    if isinstance(name, str) and _SAFE_LABEL_RE.match(name):
        return name
    return describe_class(type(backend))


def embed_texts(
    backend: EmbeddingBackend,
    texts: Sequence[str],
    *,
    expected_dimension: int | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Embed ``texts`` with ``backend`` and return a validated float32 matrix.

    Empty and whitespace-only texts are not sent; their rows are zeros. Requests
    are chunked by the backend's ``max_batch_size``. Every response is checked
    for row count, a consistent width that matches ``backend.dimension`` and
    ``expected_dimension`` when they are set, and finite values.

    Args:
        backend: The backend to call.
        texts: Texts to embed.
        expected_dimension: Required width, e.g. a vector index's dimension.
        normalize: L2-normalise every non-zero row.

    Returns:
        ``numpy.ndarray`` of shape ``(len(texts), d)`` and dtype float32. When
        every text is empty, ``d`` is ``expected_dimension``, else
        ``backend.dimension``, else 0.

    Raises:
        EmbeddingBackendResponseError: The backend's output does not fit the request.
    """
    import numpy as np

    label = _backend_label(backend)
    declared = getattr(backend, "dimension", None)
    if declared is not None and expected_dimension is not None and declared != expected_dimension:
        raise EmbeddingBackendResponseError(
            f"Embedding backend {label!r} produces {declared}-dimensional vectors, "
            f"but {expected_dimension} are required"
        )
    width: int | None = expected_dimension if expected_dimension is not None else declared

    for text in texts:
        if not isinstance(text, str):
            raise TypeError(f"embed_texts expects strings, got {type(text).__name__}")
    indices = [i for i, text in enumerate(texts) if text.strip()]

    if not indices:
        return np.zeros((len(texts), width or 0), dtype=np.float32)

    batch = getattr(backend, "max_batch_size", None)
    batch = batch if isinstance(batch, int) and batch > 0 else len(indices)
    chunks: list[np.ndarray] = []
    for start in range(0, len(indices), batch):
        sent = [texts[i] for i in indices[start : start + batch]]
        rows = _as_matrix(_call_embed(backend, sent, label), label)
        if rows.shape[0] != len(sent):
            raise EmbeddingBackendResponseError(
                f"Embedding backend {label!r} returned {rows.shape[0]} vectors for {len(sent)} texts"
            )
        if width is None:
            width = int(rows.shape[1])
        if rows.shape[1] != width or width == 0:
            raise EmbeddingBackendResponseError(
                f"Embedding backend {label!r} returned {rows.shape[1]}-dimensional vectors; "
                f"expected {width}"
            )
        if not np.isfinite(rows).all():
            raise EmbeddingBackendResponseError(
                f"Embedding backend {label!r} returned non-finite values"
            )
        chunks.append(rows)

    assert width is not None
    out = np.zeros((len(texts), width), dtype=np.float32)
    out[indices] = np.concatenate(chunks, axis=0)
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = (out / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    return out


def _call_embed(backend: EmbeddingBackend, texts: list[str], label: str) -> Any:
    try:
        return backend.embed(texts)
    except BackendError:
        raise
    except Exception as exc:
        logger.debug("Embedding backend %r failed: %s", label, describe_exception(exc))
        raise EmbeddingBackendCallError(
            f"Embedding backend {label!r} failed ({describe_class(type(exc))})"
        ) from exc


def _as_matrix(rows: Any, label: str) -> np.ndarray:
    import numpy as np

    try:
        if isinstance(rows, np.ndarray):
            matrix = rows.astype(np.float32, copy=False)
        else:
            matrix = np.asarray([np.asarray(row, dtype=np.float32) for row in rows])
    except (TypeError, ValueError) as exc:
        raise EmbeddingBackendResponseError(
            f"Embedding backend {label!r} returned rows that are not numeric vectors of one width"
        ) from exc
    if matrix.ndim != 2:
        raise EmbeddingBackendResponseError(
            f"Embedding backend {label!r} must return a 2-D array of shape (texts, dimension); "
            f"got {matrix.ndim}-D"
        )
    return matrix


class CallableEmbeddingBackend:
    """Adapt a plain function ``fn(list[str]) -> rows`` into an :class:`EmbeddingBackend`.

    Args:
        fn: Called with a list of non-empty texts; returns one row per text.
        dimension: Output width, if known. Validated on every call when set.
        model_name: Reported model name.
        name: Reported backend name.
        max_batch_size: Largest list ``fn`` is called with.
        max_tokens: The model's input window, if known.
    """

    def __init__(
        self,
        fn: Callable[[list[str]], Any],
        *,
        dimension: int | None = None,
        model_name: str = "callable",
        name: str = "callable",
        max_batch_size: int | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn
        self.name = name
        self.model_name = model or model_name
        self.dimension = _positive_int_or_none(dimension, "dimension")
        self.max_batch_size = _positive_int_or_none(max_batch_size, "max_batch_size")
        self.max_tokens = _positive_int_or_none(max_tokens, "max_tokens")

    def embed(self, texts: list[str]) -> Any:
        return self._fn(texts)

    def __repr__(self) -> str:
        return (
            f"CallableEmbeddingBackend(fn={getattr(self._fn, '__qualname__', self._fn)!r}, "
            f"model_name={self.model_name!r}, dimension={self.dimension!r})"
        )


def cosine_rows(matrix: np.ndarray, query: np.ndarray) -> list[float]:
    """Cosine similarity of every row of ``matrix`` with ``query``, clamped to [0, 1].

    Zero rows (empty texts) score 0.0.
    """
    import numpy as np

    q_norm = float(np.linalg.norm(query))
    if q_norm == 0 or matrix.shape[0] == 0:
        return [0.0] * int(matrix.shape[0])
    norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query
    out: list[float] = []
    for dot, norm in zip(dots.tolist(), norms.tolist()):
        if norm == 0 or not math.isfinite(dot):
            out.append(0.0)
        else:
            out.append(max(0.0, min(1.0, float(dot) / (norm * q_norm))))
    return out
