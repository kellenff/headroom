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
EXTERNAL                ``<backend name>:<model or ''>:<dimension>``
injected ``embedder``   ``custom:<model_name>:<dimension>``
======================  ==================================================

A database with no recorded identity (written before this check existed)
adopts the identity of the first process that opens it. The record is a
single-row table in the memory store's own SQLite file. Disable the check with
``LocalBackendConfig(enforce_embedder_identity=False)`` or
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
from headroom.memory.config import EmbedderBackend, MemoryConfig, external_embedder_selection

logger = logging.getLogger(__name__)

#: Set to ``0`` (or ``false``/``no``/``off``) to skip the identity check.
IDENTITY_CHECK_ENV = "HEADROOM_EMBEDDER_IDENTITY_CHECK"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

_TABLE = "headroom_embedder_identity"
_ONNX_IDENTITY = "sentence-transformers:all-MiniLM-L6-v2:384"
_ST_PREFIX = "sentence-transformers/"


class EmbedderMismatchError(ValueError):
    """A memory database was written by a different embedder than this process uses.

    Attributes:
        db_path: The database that was refused.
        recorded: The identity stored in the database.
        current: The identity this process is configured with.
    """

    def __init__(self, db_path: Path, recorded: str, current: str) -> None:
        self.db_path = db_path
        self.recorded = recorded
        self.current = current
        super().__init__(
            f"Memory database {db_path} was written with embedder {recorded!r}, but this "
            f"process is configured for {current!r}. Vectors from different embedding models "
            "cannot be compared, so the database was not opened. Use the same "
            "HEADROOM_EMBEDDER_* settings in every process that opens this database (the "
            "proxy, the memory MCP server, `headroom memory` and memory sync), or point this "
            f"process at a fresh memory directory. {IDENTITY_CHECK_ENV}=0 skips this check."
        )


def identity_check_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether ``HEADROOM_EMBEDDER_IDENTITY_CHECK`` leaves the check on (the default)."""
    env = os.environ if env is None else env
    return (env.get(IDENTITY_CHECK_ENV) or "").strip().lower() not in _DISABLED_VALUES


def embedder_identity(config: MemoryConfig, embedder: object | None = None) -> str:
    """The identity of the embedding space ``config`` produces.

    Args:
        config: The memory configuration.
        embedder: The embedder built from ``config``. Only consulted for an
            injected ``config.embedder``, whose model name is known only to the
            object itself.
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
        name, model_or_none = external_embedder_selection(config)
        return f"{name}:{model_or_none or ''}:{dimension}"
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
        if recorded is None:
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
        raise EmbedderMismatchError(path, recorded, identity)


def _read(conn: sqlite3.Connection) -> str | None:
    # No table: a database written before identities were recorded.
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_TABLE,)
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute(f"SELECT identity FROM {_TABLE} WHERE id = 1").fetchone()
    return str(row[0]) if row is not None else None
