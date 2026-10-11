"""
face_recognition_service.service
================================

High-level façade that wires together :class:`FaceEngine`,
:class:`EmbeddingMatcher`, and :class:`EmbeddingCrypto` into a single
convenience object.

For fine-grained control, use the individual classes directly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from face_recognition_service.crypto import EmbeddingCrypto
from face_recognition_service.face_engine import FaceEngine
from face_recognition_service.matcher import EmbeddingMatcher

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """All-in-one face recognition service.

    Combines embedding extraction, matching, and encryption into a
    single callable object.

    Parameters
    ----------
    encryption_key : str or bytes or None
        Fernet key for :class:`EmbeddingCrypto`.  Pass ``None`` to
        auto-generate a key (retrieve it from ``service.crypto.key``).
    model_name : str
        InsightFace model pack.  Default ``"buffalo_l"``.
    ctx_id : int
        ONNX execution context.  ``0`` = GPU, ``-1`` = CPU.
    det_size : tuple[int, int]
        Face-detection input size.
    match_threshold : float
        Default cosine-similarity threshold for matching.
    """

    def __init__(
        self,
        encryption_key: Optional[str | bytes] = None,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (320, 320),
        match_threshold: float = 0.6,
    ) -> None:
        self.engine = FaceEngine(
            model_name=model_name,
            ctx_id=ctx_id,
            det_size=det_size,
        )
        self.matcher = EmbeddingMatcher(threshold=match_threshold)
        self.crypto = EmbeddingCrypto(key=encryption_key)

        logger.info("FACE_RECOGNITION_SERVICE_READY")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract the face embedding from raw image bytes.

        Delegates to :meth:`FaceEngine.extract`.

        Returns
        -------
        numpy.ndarray or None
            512-d embedding or ``None`` if no face is detected.
        """
        return self.engine.extract(image_bytes)

    def extract_all_embeddings(self, image_bytes: bytes) -> list[np.ndarray]:
        """Extract embeddings for every face in the image.

        Returns
        -------
        list[numpy.ndarray]
            One 512-d embedding per detected face.
        """
        return self.engine.extract_all(image_bytes)

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Encrypt an embedding for secure storage / transport.

        Returns
        -------
        bytes
            Fernet ciphertext.
        """
        return self.crypto.encrypt(embedding)

    def decrypt_embedding(self, ciphertext: bytes) -> np.ndarray:
        """Decrypt ciphertext back to a numpy embedding.

        Returns
        -------
        numpy.ndarray
            The restored 512-d embedding.
        """
        return self.crypto.decrypt(ciphertext)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float]:
        """Compare two embeddings.

        Returns
        -------
        tuple[bool, float]
            ``(is_match, cosine_similarity)``.
        """
        return self.matcher.match(embedding_a, embedding_b, threshold=threshold)

    def find_best_match(
        self,
        probe: np.ndarray,
        gallery: dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        """Find the closest match from a gallery of known embeddings.

        Returns
        -------
        tuple[str | None, float]
            ``(identity_id, similarity)`` or ``(None, best_sim)``.
        """
        return self.matcher.find_best_match(probe, gallery, threshold=threshold)

    # ------------------------------------------------------------------
    # Convenience: extract → encrypt pipeline
    # ------------------------------------------------------------------

    def extract_and_encrypt(self, image_bytes: bytes) -> Optional[bytes]:
        """Extract a face embedding and return it encrypted.

        A common pipeline: take a photo, get back encrypted bytes
        suitable for database storage.

        Returns
        -------
        bytes or None
            Encrypted embedding, or ``None`` if no face was detected.
        """
        embedding = self.extract_embedding(image_bytes)
        if embedding is None:
            return None
        return self.encrypt_embedding(embedding)

    def decrypt_and_match(
        self,
        ciphertext_a: bytes,
        ciphertext_b: bytes,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float]:
        """Decrypt two ciphertexts and compare the embeddings.

        Returns
        -------
        tuple[bool, float]
            ``(is_match, cosine_similarity)``.
        """
        emb_a = self.decrypt_embedding(ciphertext_a)
        emb_b = self.decrypt_embedding(ciphertext_b)
        return self.match(emb_a, emb_b, threshold=threshold)
