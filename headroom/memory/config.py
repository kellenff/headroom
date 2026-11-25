"""Configuration dataclasses for Headroom's hierarchical memory system.

Provides configuration options for all pluggable components:
- Storage backends (SQLite, future: PostgreSQL, DynamoDB)
- Vector index backends (SQLITE_VEC recommended, HNSW fallback)
- Text index backends (FTS5, future: Elasticsearch)
- Embedder backends (local sentence-transformers, OpenAI, Ollama, or any
  backend from :mod:`headroom.models.backends`)
- Caching options
- Bubbling behavior defaults
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from headroom.models.backends.registry import parse_backend_spec
from headroom.models.config import ML_MODEL_DEFAULTS


class StoreBackend(Enum):
    """Supported memory store backends."""

    SQLITE = "sqlite"
    EXTERNAL = "external"  # Loaded from entry_points(group="headroom.memory_store")


class VectorBackend(Enum):
    """Supported vector index backends."""

    AUTO = "auto"  # Auto-select: SQLITE_VEC if available, else HNSW
    SQLITE_VEC = "sqlite_vec"  # SQLite-based, bounded memory, recommended
    HNSW = "hnsw"  # hnswlib-based, unbounded unless max_entries set
    EXTERNAL = "external"  # Loaded from entry_points(group="headroom.memory_vector")


class TextBackend(Enum):
    """Supported text index backends."""

    FTS5 = "fts5"
    EXTERNAL = "external"  # Loaded from entry_points(group="headroom.memory_text")


class EmbedderBackend(Enum):
    """Supported embedder backends."""

    LOCAL = "local"  # sentence-transformers (requires torch ~2GB)
    ONNX = "onnx"  # ONNX Runtime (no torch, ~86MB, recommended)
    OPENAI = "openai"
    OLLAMA = "ollama"
    # A text-embedding backend from headroom.models.backends, selected by its
    # registry name (MemoryConfig.embedder_backend_name).
    EXTERNAL = "external"


@dataclass
class MemoryConfig:
    """Complete configuration for the memory system.

    This dataclass holds all configuration options needed to initialize
    the memory system components. Each component can be configured
    independently, allowing for flexible deployment scenarios.

    Attributes:
        store_backend: Which storage backend to use for memory persistence.
        db_path: Path to the database file (for file-based backends like SQLite).

        vector_backend: Which vector index backend to use (AUTO, SQLITE_VEC, HNSW).
            AUTO (default) selects SQLITE_VEC if available, else HNSW.
        vector_dimension: Dimension of embedding vectors.
        vector_db_path: Path to vector index database (for SQLITE_VEC). Derived from
            db_path if None.
        vector_cache_size_kb: SQLite page cache size for vector index (8MB default).
        hnsw_ef_construction: HNSW index build-time accuracy parameter.
        hnsw_m: HNSW maximum number of connections per node.
        hnsw_ef_search: HNSW search-time accuracy parameter.
        hnsw_max_entries: Maximum entries for HNSW (None = unbounded).

        text_backend: Which text index backend to use for full-text search.

        embedder_backend: Which embedder to use for generating embeddings.
        embedder_model: Model name/identifier for the LOCAL, OPENAI and OLLAMA
            embedders. Ignored for EXTERNAL: it always holds a MiniLM default,
            and silently sending that to another backend would pick a model
            nobody chose. Name an EXTERNAL model in ``embedder_backend_name``
            or ``embedder_options["model"]`` instead.
        openai_api_key: API key for OpenAI embeddings (if using OpenAI backend).
        openai_base_url: Optional OpenAI-compatible embedding API base URL.
        ollama_base_url: Base URL for Ollama server (if using Ollama backend).
        embedder_backend_name: For EXTERNAL, the embedding backend's registry
            spec: ``"name"`` or ``"name:model"`` (see
            :func:`headroom.models.backends.create_embedding_backend`).
        embedder_options: For EXTERNAL, constructor options for that backend
            (``base_url``, ``api_key``, ...). ``model`` may be given here
            instead of in the spec. ``dimension`` is always
            ``vector_dimension``. Excluded from ``repr``: options routinely
            carry credentials.
        embedder: An embedder you construct yourself; when set it wins over
            ``embedder_backend``. Either an async memory
            :class:`~headroom.memory.ports.Embedder` (used as is), a synchronous
            :class:`~headroom.models.backends.EmbeddingBackend`, or a plain
            function ``fn(list[str]) -> rows``. Its vectors must have
            ``vector_dimension`` dimensions. Never closed by Headroom: the
            caller owns its lifecycle.

        cache_enabled: Whether to enable the memory cache layer.
        cache_max_size: Maximum number of entries in the cache.

        auto_bubble: Whether to automatically bubble memories up the hierarchy.
        bubble_threshold: Minimum importance score for bubbling (0.0 - 1.0).

    Example:
        config = MemoryConfig(
            db_path=Path("./my_memory.db"),
            embedder_backend=EmbedderBackend.OPENAI,
            openai_api_key="sk-...",
            cache_max_size=2000,
        )

        # Any OpenAI-compatible server through the backend registry:
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.EXTERNAL,
            embedder_backend_name="openai:nomic-embed-text",
            embedder_options={"base_url": "http://localhost:8080/v1"},
            vector_dimension=768,
        )
    """

    # Storage
    store_backend: StoreBackend = StoreBackend.SQLITE
    store_backend_name: str | None = None  # Required when store_backend == EXTERNAL
    db_path: Path = field(default_factory=lambda: Path("headroom_memory.db"))

    # Vector index
    vector_backend: VectorBackend = VectorBackend.AUTO  # Auto-select best available
    vector_backend_name: str | None = None  # Required when vector_backend == EXTERNAL
    vector_dimension: int = 384
    vector_db_path: Path | None = (
        None  # For SQLite-based vector index (derived from db_path if None)
    )
    vector_cache_size_kb: int = 8192  # SQLite page cache size (8MB default)
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16
    hnsw_ef_search: int = 50
    hnsw_max_entries: int | None = None  # Max entries for HNSW (None = unbounded)

    # Text index
    text_backend: TextBackend = TextBackend.FTS5
    text_backend_name: str | None = None  # Required when text_backend == EXTERNAL

    # Embedder
    embedder_backend: EmbedderBackend = EmbedderBackend.LOCAL
    embedder_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer)
    openai_api_key: str | None = field(default=None, repr=False)
    ollama_base_url: str = "http://localhost:11434"

    # Cache
    cache_enabled: bool = True
    cache_max_size: int = 1000

    # Bubbling defaults
    auto_bubble: bool = True
    bubble_threshold: float = 0.7  # Minimum importance for bubbling

    # OpenAI-compatible embedding service; None uses the SDK's default endpoint.
    openai_base_url: str | None = None

    # Pluggable embedding backend (headroom.models.backends) for EXTERNAL.
    embedder_backend_name: str | None = None  # "name" or "name:model"
    embedder_options: dict[str, Any] = field(default_factory=dict, repr=False)
    # Caller-supplied embedder; wins over embedder_backend. Not compared: two
    # configs that differ only in this object are still the same settings.
    embedder: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.vector_dimension < 1:
            raise ValueError(f"vector_dimension must be positive, got {self.vector_dimension}")

        if self.hnsw_ef_construction < 1:
            raise ValueError(
                f"hnsw_ef_construction must be positive, got {self.hnsw_ef_construction}"
            )

        if self.hnsw_m < 1:
            raise ValueError(f"hnsw_m must be positive, got {self.hnsw_m}")

        if self.hnsw_ef_search < 1:
            raise ValueError(f"hnsw_ef_search must be positive, got {self.hnsw_ef_search}")

        if self.cache_max_size < 1:
            raise ValueError(f"cache_max_size must be positive, got {self.cache_max_size}")

        if self.embedder_backend == EmbedderBackend.OPENAI and not self.openai_api_key:
            raise ValueError("openai_api_key is required when using OpenAI embedder backend")

        if not isinstance(self.embedder_options, dict):
            raise TypeError(
                f"embedder_options must be a dict, got {type(self.embedder_options).__name__}"
            )
        if self.embedder_backend_name is not None:
            parse_backend_spec(self.embedder_backend_name)
        if self.embedder_backend == EmbedderBackend.EXTERNAL and self.embedder is None:
            if not self.embedder_backend_name:
                raise ValueError(
                    "embedder_backend_name is required when embedder_backend is EXTERNAL; "
                    "set it to an embedding backend name such as 'openai' or 'name:model'"
                )
            # Settle model and dimension contradictions here, not on first use.
            external_embedder_selection(self)

        # Ensure db_path is a Path object
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)


def external_embedder_selection(config: MemoryConfig) -> tuple[str, str | None]:
    """The ``(backend name, model)`` an EXTERNAL embedder is built with.

    The model comes only from the spec suffix (``"name:model"``) or
    ``embedder_options["model"]``, never from ``embedder_model``, whose MiniLM
    default would otherwise be sent to a backend that serves something else.
    ``None`` lets the backend use its own default model.

    Raises:
        ValueError: ``embedder_backend_name`` is missing or malformed, the two
            model sources disagree, or ``embedder_options["dimension"]``
            contradicts ``vector_dimension``.
    """
    if not config.embedder_backend_name:
        raise ValueError("embedder_backend_name is required for an EXTERNAL embedder")
    name, spec_model = parse_backend_spec(config.embedder_backend_name)

    option_model = config.embedder_options.get("model")
    if option_model is not None and not isinstance(option_model, str):
        raise ValueError("embedder_options['model'] must be a string")
    option_model = option_model.strip() if option_model else None
    if spec_model and option_model and spec_model != option_model:
        raise ValueError(
            f"embedder_backend_name names model {spec_model!r} but embedder_options['model'] "
            f"is {option_model!r}; set the model in one place"
        )

    dimension = config.embedder_options.get("dimension")
    if dimension is not None and dimension != config.vector_dimension:
        raise ValueError(
            f"embedder_options['dimension'] ({dimension!r}) contradicts vector_dimension "
            f"({config.vector_dimension}); set only vector_dimension"
        )
    return name, spec_model or option_model
