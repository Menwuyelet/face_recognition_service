"""
face_recognition_service.face_engine
=====================================

Wrapper around InsightFace's FaceAnalysis model for extracting 512-d
face embeddings from raw image bytes.

This module is framework-agnostic — it depends only on ``insightface``,
``opencv-python-headless``, and ``numpy``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

# Silence noisy ONNX / InsightFace logs at import time
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
os.environ.setdefault("GLOG_minloglevel", "3")

from insightface.app import FaceAnalysis  # noqa: E402

logger = logging.getLogger(__name__)


class FaceEngine:
    """Extract face embeddings from images using InsightFace (buffalo_l).

    The underlying ``FaceAnalysis`` model is initialized **lazily** on the
    first call to :meth:`extract` (or eagerly via :meth:`initialize`), so
    importing this module is cheap.

    Parameters
    ----------
    model_name : str
        InsightFace model pack name.  Default ``"buffalo_l"``.
    ctx_id : int
        ONNX execution-provider context.  ``0`` = first GPU if available,
        ``-1`` = CPU only.
    det_size : tuple[int, int]
        Detection input resolution.  Smaller = faster but less accurate.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (320, 320),
    ) -> None:
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._app: Optional[FaceAnalysis] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Eagerly load the model.  Called automatically on first use."""
        if self._app is not None:
            return
        logger.info(
            "FACE_ENGINE_INIT_STARTED | model=%s ctx_id=%s det_size=%s",
            self._model_name,
            self._ctx_id,
            self._det_size,
        )
        self._app = FaceAnalysis(name=self._model_name)
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        logger.info("FACE_ENGINE_INIT_SUCCESS")

    @property
    def app(self) -> FaceAnalysis:
        """Return the underlying ``FaceAnalysis`` instance (lazy-init)."""
        if self._app is None:
            self.initialize()
        return self._app

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert raw image bytes to a 512-d face embedding.

        Parameters
        ----------
        image_bytes : bytes
            JPEG / PNG encoded image data.

        Returns
        -------
        numpy.ndarray or None
            The embedding vector (shape ``(512,)``, dtype ``float32``) of
            the first detected face, or ``None`` when no face is found.
        """
        if not image_bytes:
            logger.warning("FACE_ENGINE_EXTRACT | empty image_bytes")
            return None

        logger.info(
            "FACE_ENGINE_EXTRACT_STARTED | image_size=%d",
            len(image_bytes),
        )

        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("FACE_ENGINE_EXTRACT | failed to decode image")
            return None

        faces = self.app.get(img)

        if not faces:
            logger.info("FACE_ENGINE_EXTRACT | no face detected")
            return None

        embedding = faces[0].embedding

        logger.info(
            "FACE_ENGINE_EXTRACT_SUCCESS | embedding_dim=%d",
            embedding.shape[0],
        )

        return embedding

    def extract_all(self, image_bytes: bytes) -> list[np.ndarray]:
        """Extract embeddings for **all** faces detected in the image.

        Parameters
        ----------
        image_bytes : bytes
            JPEG / PNG encoded image data.

        Returns
        -------
        list[numpy.ndarray]
            A list of embedding vectors, one per detected face.
            Empty list when no faces are found.
        """
        if not image_bytes:
            return []

        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return []

        faces = self.app.get(img)
        return [f.embedding for f in faces]
