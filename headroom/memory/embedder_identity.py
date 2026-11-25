"""Which embedding space a memory database was written in, and refusing another.

Vectors from different embedding models are not comparable: a query embedded by
one model and searched against rows from another returns confident nonsense,
and no error ever says why. Several processes open the same memory database
(the proxy, the memory MCP server, ``headroom memory`` and the sync subprocess),
each configuring its embedder independently, so a mismatch is one forgotten
environment variable away.

So every :class:`~headroom.memory.backends.local.LocalBackend` stamps its
database with an *embedder identity* on first open and refuses to open it
later with a different one. The identity is ``"family:model:dimension"``:

======================  ==================================================
embedder                identity
======================  ==================================================
LOCAL                   ``sentence-transformers:<model>:<dimension>``
                        (any ``sentence-transformers/`` prefix dropped)
ONNX                    ``sentence-transformers:all-MiniLM-L6-v2:384``
                        (it always loads MiniLM, so a proxy on ONNX and a
                        CLI on sentence-transformers interoperate)
OPENAI                  ``openai:<model>:<dimension>``
OLLAMA                  ``ollama:<model>:<dimension>``
EXTERNAL                ``<backend name>:<model it serves>:<dimension>``
                        (the model the backend resolved, its default
                        included; the ``sentence-transformers`` backend drops
                        the prefix like LOCAL, so both share identities)
injected ``embedder``   ``custom:<model_name>:<dimension>``
======================  ==================================================

A database with no recorded identity (written before this check existed)
adopts the identity of the first process that opens it, but only when the
vectors it already stores have that identity's width. One holding vectors of
another width is refused and left unrecorded, so the embedder that wrote it can
still claim it. The record is a single-row table in the memory store's own
SQLite file; :func:`clear_embedder_identity` removes a wrong one. Disable the
check with ``LocalBackendConfig(enforce_embedder_identity=False)`` or
``HEADROOM_EMBEDDER_IDENTITY_CHECK=0``; a mismatch is then logged, not raised,
and nothing is recorded.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from headroom.fileperms import connect_private_sqlite
from headroom.log_safety import describe_exception
from headroom.memory.config import EmbedderBackend, MemoryConfig, external_embedder_selection

logger = logging.getLogger(__name__)

#: Set to ``0`` (or ``false``/``no``/``off``) to skip the identity check.
IDENTITY_CHECK_ENV = "HEADROOM_EMBEDDER_IDENTITY_CHECK"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

_TABLE = "headroom_embedder_identity"
_ONNX_IDENTITY = "sentence-transformers:all-MiniLM-L6-v2:384"
_ST_PREFIX = "sentence-transformers/"
_ST_BACKEND = "sentence-transformers"
#: The family of the recorded side when stored vectors, not a record, disagree.
_UNRECORDED = "unrecorded"


class EmbedderMismatchError(ValueError):
    """A memory database was written by a different embedder than this process uses.

    Attributes:
        db_path: The database that was refused.
        recorded: The identity stored in the database, or ``"unrecorded::N"``
            for a database with no record whose stored vectors are ``N`` wide.
        current: The identity this process is configured with.
        stored_dimension: ``N`` when the stored vectors, not a record,
            disagreed; otherwise ``None``.
    """

    def __init__(
        self,
        db_path: Path,
        recorded: str,
        current: str,
        *,
        stored_dimension: int | None = None,
    ) -> None:
        self.db_path = db_path
        self.recorded = recorded
        self.current = current
        self.stored_dimension = stored_dimension
        if stored_dimension is None:
            found = f"was written with embedder {recorded!r}"
            recovery = (
                "A wrong record is removed by "
                "headroom.memory.embedder_identity.clear_embedder_identity()"
            )
        else:
            found = f"holds {stored_dimension}-wide vectors from an unrecorded embedder"
            recovery = "Nothing was recorded, so the embedder that wrote it can still open it"
        super().__init__(
            f"Memory database {db_path} {found}, but this process uses {current!r}; "
            "vectors from different models cannot be compared, so it was not opened. Use the "
            "same HEADROOM_EMBEDDER_* settings wherever it is opened (proxy, memory MCP server, "
            f"`headroom memory`, memory sync) or a fresh memory directory. {recovery}; "
            f"{IDENTITY_CHECK_ENV}=0 skips the check."
        )


def identity_check_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether ``HEADROOM_EMBEDDER_IDENTITY_CHECK`` leaves the check on (the default)."""
    env = os.environ if env is None else env
    return (env.get(IDENTITY_CHECK_ENV) or "").strip().lower() not in _DISABLED_VALUES


def embedder_identity(config: MemoryConfig, embedder: object | None = None) -> str:
    """The identity of the embedding space ``config`` produces.

    Args:
        config: The memory configuration.
        embedder: The embedder built from ``config``. Consulted for an injected
            ``config.embedder`` and for EXTERNAL, whose model is the one the
            backend resolved (its default when none is configured); without
            it, EXTERNAL falls back to the configured model.
    """
    dimension = config.vector_dimension
    if config.embedder is not None:
        source = embedder if embedder is not None else config.embedder
        model = getattr(source, "model_name", None)
        return f"custom:{model if isinstance(model, str) and model else 'callable'}:{dimension}"

    backend = config.embedder_backend
    if backend == EmbedderBackend.ONNX:
        return _ONNX_IDENTITY
    if backend == EmbedderBackend.LOCAL:
        model = config.embedder_model
        if model.startswith(_ST_PREFIX):
            model = model[len(_ST_PREFIX) :]
        return f"sentence-transformers:{model}:{dimension}"
    if backend == EmbedderBackend.OPENAI:
        return f"openai:{config.embedder_model}:{dimension}"
    if backend == EmbedderBackend.OLLAMA:
        return f"ollama:{config.embedder_model}:{dimension}"
    if backend == EmbedderBackend.EXTERNAL:
        name, configured = external_embedder_selection(config)
        served = getattr(embedder, "model_name", None) if embedder is not None else None
        model = served.strip() if isinstance(served, str) and served.strip() else configured or ""
        if name == _ST_BACKEND and model.startswith(_ST_PREFIX):
            # LOCAL's and ONNX's family string: one model, one identity.
            model = model[len(_ST_PREFIX) :]
        return f"{name}:{model}:{dimension}"
    raise ValueError(f"Unknown embedder backend: {backend}")


def is_file_database(db_path: str | os.PathLike[str]) -> bool:
    """Whether ``db_path`` names an on-disk SQLite file (not ``:memory:``)."""
    return os.fspath(db_path) not in ("", ":memory:")


def read_embedder_identity(db_path: str | os.PathLike[str]) -> str | None:
    """The identity recorded in ``db_path``, or ``None`` if there is none (or no file)."""
    path = Path(db_path)
    if not is_file_database(db_path) or not path.exists():
        return None
    conn = connect_private_sqlite(path, what="memory store")
    try:
        return _read(conn)
    finally:
        conn.close()


def verify_embedder_identity(
    db_path: str | os.PathLike[str],
    identity: str,
    *,
    enforce: bool = True,
) -> None:
    """Record ``identity`` in ``db_path`` on first open; refuse a different one later.

    With ``enforce=False`` nothing is written and a mismatch is only logged.
    Databases that are not files (``:memory:``) are skipped.

    Raises:
        EmbedderMismatchError: ``enforce`` is set and the database records a
            different identity.
    """
    if not is_file_database(db_path):
        return
    path = Path(db_path)
    if not enforce:
        recorded = read_embedder_identity(path)
        if recorded is None and path.exists():
            conn = connect_private_sqlite(path, what="memory store")
            try:
                width = _stored_width_mismatch(conn, identity)
            finally:
                conn.close()
            recorded = None if width is None else f"{_UNRECORDED}::{width}"
        if recorded is not None and recorded != identity:
            logger.warning(
                "Memory: %s was written with embedder %r but is being opened with %r; "
                "%s=0 disables the check, so search results will mix embedding spaces.",
                path,
                recorded,
                identity,
                IDENTITY_CHECK_ENV,
            )
        return

    # Autocommit mode with an explicit BEGIN IMMEDIATE: the read and the insert
    # hold the write lock together, so two processes opening a fresh database
    # at once cannot both record their own identity. Closing without COMMIT
    # (on any error) rolls the transaction back.
    conn = connect_private_sqlite(path, what="memory store", isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "identity TEXT NOT NULL, "
            "recorded_at TEXT NOT NULL)"
        )
        recorded = _read(conn)
        stored_dimension: int | None = None
        if recorded is None:
            # No record yet. Adopt this identity only if the vectors already
            # stored (a database from before identities) have its width.
            stored_dimension = _stored_width_mismatch(conn, identity)
            if stored_dimension is not None:
                recorded = f"{_UNRECORDED}::{stored_dimension}"
            else:
                conn.execute(
                    f"INSERT INTO {_TABLE} (id, identity, recorded_at) VALUES (1, ?, ?)",
                    (identity, datetime.now(timezone.utc).isoformat()),
                )
                recorded = identity
                logger.info("Memory: recorded embedder %r for %s", identity, path)
        conn.execute("COMMIT")
    finally:
        conn.close()

    if recorded != identity:
        raise EmbedderMismatchError(path, recorded, identity, stored_dimension=stored_dimension)


def clear_embedder_identity(db_path: str | os.PathLike[str]) -> str | None:
    """Remove the identity recorded in ``db_path``; return it, or ``None`` if there was none.

    The recovery for a wrong record, such as one stamped by a process that
    opened the database first with other ``HEADROOM_EMBEDDER_*`` settings. The
    next open records its own embedder again, after checking it against the
    stored vectors' width. Memories and vectors are left untouched.
    """
    path = Path(db_path)
    if not is_file_database(db_path) or not path.exists():
        return None
    conn = connect_private_sqlite(path, what="memory store", isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        recorded = _read(conn)
        if recorded is not None:
            conn.execute(f"DELETE FROM {_TABLE}")
            logger.info("Memory: cleared the recorded embedder for %s", path)
        conn.execute("COMMIT")
    finally:
        conn.close()
    return recorded


def _stored_width_mismatch(conn: sqlite3.Connection, identity: str) -> int | None:
    """The stored vectors' width when ``identity`` names another; otherwise ``None``.

    Reads the most recently written ``memories.embedding`` blob (float32, four
    bytes a value). ``None`` also when nothing tells: no memories table, no
    stored vector, or an identity without a trailing width.
    """
    _, _, tail = identity.rpartition(":")
    if not tail.isdigit():
        return None
    has_memories = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memories'"
    ).fetchone()
    if has_memories is None:
        return None
    try:
        row = conn.execute(
            "SELECT length(embedding) FROM memories WHERE typeof(embedding) = 'blob' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    except sqlite3.DatabaseError as exc:
        # A memories table that is not the memory store's (no embedding column).
        logger.debug("Memory: could not read a stored vector: %s", describe_exception(exc))
        return None
    if row is None or not isinstance(row[0], int) or row[0] <= 0 or row[0] % 4:
        return None
    width = row[0] // 4
    return None if width == int(tail) else width


def _read(conn: sqlite3.Connection) -> str | None:
    # No table: a database written before identities were recorded.
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_TABLE,)
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute(f"SELECT identity FROM {_TABLE} WHERE id = 1").fetchone()
    return str(row[0]) if row is not None else None
