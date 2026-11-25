"""Relevance scoring module for Headroom SDK.

This module provides a unified interface for computing item relevance against
query contexts. All scorers implement the RelevanceScorer protocol:

    relevance(item, context) -> RelevanceScore

Available scorers:

1. HybridScorer (DEFAULT - recommended)
   - Combines BM25 + embeddings for best accuracy
   - Adaptive alpha: more BM25 for UUIDs, more semantic for natural language
   - Falls back gracefully to BM25 if sentence-transformers not installed
   - Install for full support: pip install "headroom-ai[relevance]"

2. BM25Scorer (zero dependencies)
   - Fast keyword matching
   - Good for exact UUIDs, IDs, specific terms
   - May miss semantic matches ("errors" won't match "failed")

3. EmbeddingScorer (requires sentence-transformers)
   - Pure semantic similarity
   - Best for natural language queries
   - Install: pip install "headroom-ai[relevance]"

Embedding backends: the embedding half can come from any
:class:`~headroom.models.backends.EmbeddingBackend` instead of fastembed,
e.g. ``create_scorer(embedding_backend="openai:text-embedding-3-small")``. A
failing backend never fails a hybrid score: the call falls back to BM25 and a
circuit breaker stops calling the backend for a cooldown.

WHY HYBRID IS DEFAULT:
- Missing important items during compression is catastrophic
- BM25 alone gives low scores for single-term matches (e.g., "Alice" = 0.07)
- Semantic matching catches "errors" -> "failed", "issues", etc.
- 5-10ms latency is acceptable vs. losing critical data

Example usage:
    from headroom.relevance import HybridScorer, create_scorer

    # Default: Hybrid (recommended)
    scorer = create_scorer()  # or HybridScorer()

    # Zero-dependency fallback
    scorer = create_scorer("bm25")

    # Score items
    items = [
        '{"id": "123", "name": "Alice"}',
        '{"id": "456", "name": "Bob"}',
    ]
    scores = scorer.score_batch(items, "find user 123")
    # scores[0].score > scores[1].score
"""

from collections.abc import Callable
from typing import Any

from headroom.models.backends import EmbeddingBackend

from .base import RelevanceScore, RelevanceScorer
from .bm25 import BM25Scorer
from .embedding import (
    EmbeddingBackendCircuitOpenError,
    EmbeddingScorer,
    FastEmbedBackend,
    embedding_available,
)
from .hybrid import HybridScorer

__all__ = [
    # Base types
    "RelevanceScore",
    "RelevanceScorer",
    # Scorers
    "BM25Scorer",
    "EmbeddingScorer",
    "HybridScorer",
    # Embedding backends
    "EmbeddingBackendCircuitOpenError",
    "FastEmbedBackend",
    # Utilities
    "embedding_available",
    # Factory function
    "create_scorer",
]


def create_scorer(
    tier: str = "hybrid",
    *,
    embedding_backend: EmbeddingBackend | str | Callable[[list[str]], Any] | None = None,
    **kwargs: Any,
) -> RelevanceScorer:
    """Factory function to create a relevance scorer.

    Args:
        tier: Scorer tier to create:
            - "hybrid": Hybrid BM25 + embedding (DEFAULT, recommended)
            - "bm25": BM25 keyword scorer (zero deps, fast)
            - "embedding": Embedding scorer (requires sentence-transformers)
        embedding_backend: Embed with this instead of fastembed: an
            ``EmbeddingBackend``, a registry spec (``"openai:<model>"``) or a
            function ``fn(list[str]) -> rows``. Needs tier "embedding" or
            "hybrid", and no fastembed. A spec naming the built-in ``openai``
            backend gets the request-path bounds (a 5 s timeout, no retries),
            as ``RelevanceScorerConfig.embedding_backend`` does; pass an
            instance to choose others.
        **kwargs: Additional arguments passed to scorer constructor.

    Returns:
        RelevanceScorer instance.

    Raises:
        ValueError: If tier is unknown, or ``embedding_backend`` is combined
            with tier "bm25" or with an ``embedding_scorer`` argument.
        RuntimeError: If tier requires unavailable dependencies.

    Note:
        HybridScorer gracefully falls back to BM25 if sentence-transformers
        is not installed, so it's safe to use as the default.

    Example:
        # Create default hybrid scorer (recommended)
        scorer = create_scorer()

        # Create BM25 scorer for zero-dependency environments
        scorer = create_scorer("bm25")

        # Create hybrid scorer with custom alpha
        scorer = create_scorer("hybrid", alpha=0.6, adaptive=True)

        # Hybrid scorer embedding through an OpenAI-compatible server
        scorer = create_scorer(
            embedding_backend=create_embedding_backend(
                "openai:nomic-embed-text", base_url="http://localhost:11434/v1"
            )
        )
    """
    tier = tier.lower()

    if embedding_backend is not None:
        return _create_backend_scorer(tier, embedding_backend, kwargs)

    if tier == "bm25":
        return BM25Scorer(**kwargs)

    elif tier == "embedding":
        if not EmbeddingScorer.is_available():
            raise RuntimeError(
                "EmbeddingScorer requires sentence-transformers. "
                'Install with: pip install "headroom-ai[relevance]"'
            )
        return EmbeddingScorer(**kwargs)

    elif tier == "hybrid":
        return HybridScorer(**kwargs)

    else:
        valid_tiers = ["bm25", "embedding", "hybrid"]
        raise ValueError(f"Unknown scorer tier: {tier}. Valid tiers: {valid_tiers}")


def _create_backend_scorer(
    tier: str,
    embedding_backend: EmbeddingBackend | str | Callable[[list[str]], Any],
    kwargs: dict[str, Any],
) -> RelevanceScorer:
    """``create_scorer`` for an explicit embedding backend.

    Contradictions fail rather than resolve silently: tier "bm25" would never
    call the backend, and a second embedding scorer would leave it unclear
    which one embeds.
    """
    if tier == "bm25":
        raise ValueError(
            "embedding_backend needs tier 'embedding' or 'hybrid'; tier 'bm25' never embeds"
        )
    if "embedding_scorer" in kwargs:
        raise ValueError("Pass embedding_backend or embedding_scorer, not both")
    if tier == "embedding":
        # No fastembed availability check: the backend does the embedding.
        return EmbeddingScorer(backend=embedding_backend, **kwargs)
    if tier == "hybrid":
        return HybridScorer(embedding_scorer=EmbeddingScorer(backend=embedding_backend), **kwargs)
    valid_tiers = ["bm25", "embedding", "hybrid"]
    raise ValueError(f"Unknown scorer tier: {tier}. Valid tiers: {valid_tiers}")
