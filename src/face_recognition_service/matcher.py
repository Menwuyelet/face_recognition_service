"""
face_recognition_service.matcher
================================

Cosine-similarity matching for 512-d face embeddings.

This module is pure numpy — no framework dependencies.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Expected dimensionality of InsightFace embeddings
_EMBEDDING_DIM = 512


class EmbeddingMatcher:
    """Compare face embeddings using cosine similarity.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity to consider two embeddings a match.
        Default ``0.6`` (suitable for InsightFace buffalo_l).
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float]:
        """Compare two embeddings and return (is_match, similarity_score).

        Parameters
        ----------
        embedding_a, embedding_b : array-like
            512-d face embedding vectors.
        threshold : float or None
            Override the instance-level threshold for this call.

        Returns
        -------
        tuple[bool, float]
            ``(is_match, cosine_similarity)`` where ``is_match`` is True
            when the similarity meets or exceeds the threshold.

        Raises
        ------
        ValueError
            If either embedding cannot be normalized (e.g. wrong shape,
            contains NaN, or zero-norm).
        """
        thr = threshold if threshold is not None else self.threshold

        norm_a = self._normalize(embedding_a)
        norm_b = self._normalize(embedding_b)

        similarity = float(np.dot(norm_a, norm_b))

        logger.info(
            "MATCHER_SIMILARITY | similarity=%.4f threshold=%.2f",
            similarity,
            thr,
        )

        return (similarity >= thr, similarity)

    def find_best_match(
        self,
        probe: np.ndarray,
        gallery: dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        """Find the best-matching identity from a gallery.

        Parameters
        ----------
        probe : array-like
            The query embedding (512-d).
        gallery : dict[str, array-like]
            Mapping of ``identity_id → embedding``.
        threshold : float or None
            Override the instance-level threshold for this call.

        Returns
        -------
        tuple[str | None, float]
            ``(identity_id, similarity)`` for the best match, or
            ``(None, best_similarity)`` if no match exceeds the threshold.
        """
        thr = threshold if threshold is not None else self.threshold

        norm_probe = self._normalize(probe)

        best_id: Optional[str] = None
        best_sim: float = -1.0

        for identity_id, stored_emb in gallery.items():
            try:
                norm_stored = self._normalize(stored_emb)
            except ValueError:
                logger.warning(
                    "MATCHER_SKIP | identity=%s reason=invalid_embedding",
                    identity_id,
                )
                continue

            sim = float(np.dot(norm_probe, norm_stored))
            if sim > best_sim:
                best_sim = sim
                best_id = identity_id

        if best_id is not None and best_sim >= thr:
            logger.info(
                "MATCHER_BEST_MATCH | identity=%s similarity=%.4f",
                best_id,
                best_sim,
            )
            return (best_id, best_sim)

        logger.info(
            "MATCHER_NO_MATCH | best_similarity=%.4f threshold=%.2f",
            best_sim,
            thr,
        )
        return (None, best_sim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize an embedding to unit length.

        Parameters
        ----------
        vector : array-like
            Raw embedding vector.

        Returns
        -------
        numpy.ndarray
            Unit-length embedding of shape ``(512,)``.

        Raises
        ------
        ValueError
            On dimension mismatch, NaN values, or zero-norm vectors.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(-1).copy()

        if vec.shape[0] != _EMBEDDING_DIM:
            raise ValueError(
                f"Expected {_EMBEDDING_DIM}-d embedding, got {vec.shape[0]}-d"
            )

        if np.isnan(vec).any():
            raise ValueError("Embedding contains NaN values")

        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Embedding has zero norm")

        return vec / norm
