"""
face_recognition_service
========================

Standalone face recognition library providing:

- **Embedding extraction** — convert image bytes to 512-d face embeddings
  using InsightFace (buffalo_l model).
- **Matching** — cosine-similarity comparison of two embeddings with
  configurable threshold.
- **Encryption** — Fernet-based symmetric encryption/decryption of raw
  embedding bytes for secure storage and transport.

Quick start
-----------
::

    from face_recognition_service import FaceRecognitionService

    svc = FaceRecognitionService(encryption_key="your-fernet-key")

    # Extract
    embedding = svc.extract_embedding(image_bytes)

    # Encrypt / Decrypt
    encrypted = svc.encrypt_embedding(embedding)
    restored  = svc.decrypt_embedding(encrypted)

    # Match
    is_match, score = svc.match(embedding_a, embedding_b, threshold=0.6)
"""

from face_recognition_service.service import FaceRecognitionService
from face_recognition_service.face_engine import FaceEngine
from face_recognition_service.matcher import EmbeddingMatcher
from face_recognition_service.crypto import EmbeddingCrypto

__all__ = [
    "FaceRecognitionService",
    "FaceEngine",
    "EmbeddingMatcher",
    "EmbeddingCrypto",
]

__version__ = "1.0.0"
