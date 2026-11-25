"""Hybrid relevance scorer combining BM25 and embeddings.

This module provides a hybrid scorer that combines BM25 (keyword matching)
with embedding-based semantic similarity. Uses adaptive alpha tuning to
automatically adjust the balance based on query characteristics.

Key features:
- Best of both worlds: exact ID matching + semantic understanding
- Adaptive alpha: increases BM25 weight for UUID/ID-heavy queries
- Graceful degradation: falls back to BM25 if embeddings are unavailable,
  or for any call in which the embedding half fails
- Research-backed: Dynamic Alpha Tuning gives +2-7.5% gains (Hsu et al., 2025)

Recommended for production use where accuracy matters.
"""

from __future__ import annotations

import logging
import re
import threading

from headroom.log_safety import describe_class, describe_exception

from .base import RelevanceScore, RelevanceScorer
from .bm25 import BM25Scorer
from .embedding import EmbeddingScorer

logger = logging.getLogger(__name__)


class HybridScorer(RelevanceScorer):
    """Hybrid BM25 + Embedding scorer with adaptive fusion.

    Combines keyword matching (BM25) with semantic similarity (embeddings)
    using score fusion. The fusion weight (alpha) can be:

    1. Fixed: Use a constant alpha for all queries
    2. Adaptive: Automatically adjust alpha based on query characteristics

    Adaptive alpha increases BM25 weight when the query contains:
    - UUIDs (exact match critical)
    - Numeric IDs (4+ digits)
    - Specific hostnames or email addresses

    Example:
        # Create hybrid scorer with adaptive alpha
        scorer = HybridScorer(adaptive=True)

        # UUID query: alpha ~0.8 (favor BM25)
        score1 = scorer.score(item, "find 550e8400-e29b-41d4-a716-446655440000")

        # Semantic query: alpha ~0.5 (balanced)
        score2 = scorer.score(item, "show me the failed requests")

    If fastembed is not installed, falls back to pure BM25. If the embedding
    scorer raises during a call (a remote backend timing out, a circuit
    breaker refusing), that call is scored with BM25 alone, exactly as it would
    be without embeddings; the first such failure is logged at WARNING.
    """

    # Patterns that indicate exact match is important
    _UUID_PATTERN = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    _NUMERIC_ID_PATTERN = re.compile(r"\b\d{4,}\b")
    _HOSTNAME_PATTERN = re.compile(
        r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b"
    )
    _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    def __init__(
        self,
        alpha: float = 0.5,
        adaptive: bool = True,
        bm25_scorer: BM25Scorer | None = None,
        embedding_scorer: RelevanceScorer | None = None,
    ):
        """Initialize hybrid scorer.

        Args:
            alpha: Base fusion weight for BM25 (default 0.5).
                Combined score = alpha * BM25 + (1 - alpha) * Embedding.
            adaptive: If True, adjust alpha per query based on patterns.
            bm25_scorer: Custom BM25 scorer instance (uses default if None).
            embedding_scorer: Custom embedding scorer (uses default if None),
                e.g. ``EmbeddingScorer(backend=...)`` or any scorer whose
                scores are semantic similarities in [0, 1].
        """
        self.base_alpha = alpha
        self.adaptive = adaptive

        # Initialize scorers
        self.bm25 = bm25_scorer or BM25Scorer()

        self._failure_warned = False
        self._failure_lock = threading.Lock()

        # Embedding scorer with graceful fallback
        self.embedding: RelevanceScorer | None = None
        if embedding_scorer is not None:
            self.embedding = embedding_scorer
            self._embedding_available = True
        elif EmbeddingScorer.is_available():
            self.embedding = EmbeddingScorer()
            self._embedding_available = True
        else:
            self._embedding_available = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if hybrid scoring is available.

        Note: HybridScorer is always available (falls back to BM25).
        Use has_embedding_support() to check if embeddings are available.

        Returns:
            Always True.
        """
        return True

    def has_embedding_support(self) -> bool:
        """Check if embedding scoring is available.

        Returns:
            True if an embedding scorer was given or fastembed is installed.
            A True result does not promise every call succeeds; see
            :meth:`score_batch`.
        """
        return self._embedding_available

    def _compute_alpha(self, context: str) -> float:
        """Compute adaptive alpha based on query characteristics.

        Higher alpha = more BM25 weight (exact matching).
        Lower alpha = more embedding weight (semantic matching).

        Args:
            context: Query context.

        Returns:
            Alpha value in [0.3, 0.9].
        """
        if not self.adaptive:
            return self.base_alpha

        context_lower = context.lower()

        # Count patterns that need exact matching
        uuid_count = len(self._UUID_PATTERN.findall(context))
        id_count = len(self._NUMERIC_ID_PATTERN.findall(context))
        hostname_count = len(self._HOSTNAME_PATTERN.findall(context_lower))
        email_count = len(self._EMAIL_PATTERN.findall(context_lower))

        # Adjust alpha based on pattern counts
        alpha = self.base_alpha

        if uuid_count > 0:
            alpha = max(alpha, 0.85)  # UUIDs need exact match
        elif id_count >= 2:
            alpha = max(alpha, 0.75)  # Multiple IDs suggest lookup
        elif id_count == 1:
            alpha = max(alpha, 0.65)
        elif hostname_count > 0 or email_count > 0:
            alpha = max(alpha, 0.6)

        # Clamp to valid range
        return max(0.3, min(0.9, alpha))

    @staticmethod
    def _boosted(result: RelevanceScore) -> RelevanceScore:
        """The BM25-only score: what every call returns when embeddings are not used.

        BM25 scores are inherently low on single-term matches, so a match is
        lifted to at least 0.3 (plus 0.2 for two or more matched terms) to
        stay above typical thresholds without semantic understanding.
        """
        boosted_score = result.score
        if result.matched_terms:
            # Ensure matched items get at least 0.3 score
            boosted_score = max(boosted_score, 0.3)
            # Additional boost for multiple matches
            if len(result.matched_terms) >= 2:
                boosted_score = min(1.0, boosted_score + 0.2)
        return RelevanceScore(
            score=boosted_score,
            reason=f"Hybrid (BM25 only, boosted): {result.reason}",
            matched_terms=result.matched_terms,
        )

    def _embedding_failed(self, exc: Exception) -> None:
        """Report a failed embedding call: WARNING the first time, DEBUG after.

        Only the exception's class is named at WARNING: an embedding backend's
        message can quote its endpoint, key or the scored content.
        """
        with self._failure_lock:
            first, self._failure_warned = not self._failure_warned, True
        if first:
            logger.warning(
                "Relevance embedding scoring failed (%s); calls it fails in are scored with "
                "BM25 only. Logged once per scorer; later failures are logged at DEBUG.",
                describe_class(type(exc)),
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Embedding half of hybrid scoring failed; using BM25 only: %s",
                describe_exception(exc),
            )

    def score(self, item: str, context: str) -> RelevanceScore:
        """Score item using hybrid BM25 + embedding fusion.

        Args:
            item: Item text.
            context: Query context.

        Returns:
            RelevanceScore with combined score, or the BM25-only score when
            embeddings are unavailable or fail for this call.
        """
        # Get BM25 score
        bm25_result = self.bm25.score(item, context)

        if not self._embedding_available or self.embedding is None:
            return self._boosted(bm25_result)

        # Get embedding score. Exception only: OfflineEgressBlocked is a
        # BaseException on purpose and must reach the caller, not degrade here.
        try:
            emb_result = self.embedding.score(item, context)
        except Exception as exc:
            self._embedding_failed(exc)
            return self._boosted(bm25_result)

        # Compute adaptive alpha
        alpha = self._compute_alpha(context)

        # Combine scores
        combined_score = alpha * bm25_result.score + (1 - alpha) * emb_result.score

        return RelevanceScore(
            score=combined_score,
            reason=(
                f"Hybrid (α={alpha:.2f}): "
                f"BM25={bm25_result.score:.2f}, "
                f"Semantic={emb_result.score:.2f}"
            ),
            matched_terms=bm25_result.matched_terms,
        )

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        """Score multiple items using hybrid fusion.

        Efficiently batches BM25 and embedding scoring.

        Args:
            items: List of items to score.
            context: Query context.

        Returns:
            List of RelevanceScore objects; BM25-only scores when embeddings
            are unavailable or fail for this call.
        """
        if not items:
            return []

        # Get BM25 scores
        bm25_results = self.bm25.score_batch(items, context)

        if not self._embedding_available or self.embedding is None:
            return [self._boosted(r) for r in bm25_results]

        # Get embedding scores (Exception only, as in score()).
        try:
            emb_results = self.embedding.score_batch(items, context)
        except Exception as exc:
            self._embedding_failed(exc)
            return [self._boosted(r) for r in bm25_results]

        # Compute adaptive alpha (same for all items in batch)
        alpha = self._compute_alpha(context)

        # Combine scores
        results = []
        for bm25_r, emb_r in zip(bm25_results, emb_results):
            combined = alpha * bm25_r.score + (1 - alpha) * emb_r.score
            results.append(
                RelevanceScore(
                    score=combined,
                    reason=f"Hybrid (α={alpha:.2f}): BM25={bm25_r.score:.2f}, Emb={emb_r.score:.2f}",
                    matched_terms=bm25_r.matched_terms,
                )
            )

        return results
