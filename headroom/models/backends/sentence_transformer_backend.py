"""Built-in ``sentence-transformers`` embedding backend.

Loads the model through :class:`headroom.models.ml_models.MLModelRegistry`, so
it shares the process-wide instance other components use, is cache-first, and
refuses legibly under ``HEADROOM_OFFLINE`` when the weights are not cached.
Needs the ``sentence-transformers`` package (and torch).
"""

from __future__ import annotations

import threading
from typing import Any

from .embedding import _positive_int_or_none


class SentenceTransformerBackend:
    """Embeddings from a local sentence-transformers model.

    Args:
        model: Model id or local path. Defaults to ``HEADROOM_SENTENCE_TRANSFORMER``,
            then ``all-MiniLM-L6-v2``.
        dimension: Expected output width. Learnt from the model when omitted.
        device: ``cpu``, ``cuda`` or ``mps``. Auto-detected when omitted.
        batch_size: Batch size passed to ``encode``.
    """

    name = "sentence-transformers"

    def __init__(
        self,
        model: str | None = None,
        *,
        dimension: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from headroom.models.config import ML_MODEL_DEFAULTS

        self.model_name = (model or "").strip() or ML_MODEL_DEFAULTS.sentence_transformer
        self.dimension = _positive_int_or_none(dimension, "dimension")
        self.device = device
        self.max_batch_size = _positive_int_or_none(batch_size, "batch_size")
        self.max_tokens: int | None = None
        self._model: Any = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from headroom.models.ml_models import MLModelRegistry

                    model = MLModelRegistry.get_sentence_transformer(self.model_name, self.device)
                    native = getattr(model, "get_sentence_embedding_dimension", None)
                    width = native() if callable(native) else None
                    if self.dimension is not None and width and width != self.dimension:
                        raise ValueError(
                            f"sentence-transformers model {self.model_name!r} produces "
                            f"{width}-dimensional vectors, not {self.dimension}"
                        )
                    if self.dimension is None and isinstance(width, int) and width > 0:
                        self.dimension = width
                    seq_len = getattr(model, "max_seq_length", None)
                    if isinstance(seq_len, int) and seq_len > 0:
                        self.max_tokens = seq_len
                    self._model = model
        return self._model

    def embed(self, texts: list[str]) -> Any:
        model = self._get_model()
        return model.encode(
            list(texts),
            batch_size=self.max_batch_size or 32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

    def __repr__(self) -> str:
        return (
            f"SentenceTransformerBackend(model={self.model_name!r}, dimension={self.dimension!r})"
        )
