"""Tests for FaceEngine (mocked InsightFace — no model download needed)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFaceEngine:
    @patch("face_recognition_service.face_engine.FaceAnalysis")
    def test_extract_returns_embedding(self, mock_fa_cls):
        """Engine should return the first face's embedding."""
        # Arrange
        expected = np.random.randn(512).astype(np.float32)
        mock_face = MagicMock()
        mock_face.embedding = expected

        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        from face_recognition_service.face_engine import FaceEngine

        engine = FaceEngine()

        # Create a minimal valid JPEG (1x1 white pixel)
        import cv2

        _, buf = cv2.imencode(".jpg", np.ones((1, 1, 3), dtype=np.uint8) * 255)
        image_bytes = buf.tobytes()

        # Act
        result = engine.extract(image_bytes)

        # Assert
        np.testing.assert_array_equal(result, expected)

    @patch("face_recognition_service.face_engine.FaceAnalysis")
    def test_extract_no_face_returns_none(self, mock_fa_cls):
        """Engine should return None when no faces detected."""
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_fa_cls.return_value = mock_app

        from face_recognition_service.face_engine import FaceEngine

        engine = FaceEngine()

        import cv2

        _, buf = cv2.imencode(".jpg", np.ones((1, 1, 3), dtype=np.uint8) * 255)
        result = engine.extract(buf.tobytes())

        assert result is None

    @patch("face_recognition_service.face_engine.FaceAnalysis")
    def test_extract_empty_bytes_returns_none(self, mock_fa_cls):
        from face_recognition_service.face_engine import FaceEngine

        engine = FaceEngine()
        assert engine.extract(b"") is None

    @patch("face_recognition_service.face_engine.FaceAnalysis")
    def test_extract_all_multiple_faces(self, mock_fa_cls):
        """extract_all should return one embedding per detected face."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        face1, face2 = MagicMock(), MagicMock()
        face1.embedding = emb1
        face2.embedding = emb2

        mock_app = MagicMock()
        mock_app.get.return_value = [face1, face2]
        mock_fa_cls.return_value = mock_app

        from face_recognition_service.face_engine import FaceEngine

        engine = FaceEngine()

        import cv2

        _, buf = cv2.imencode(".jpg", np.ones((1, 1, 3), dtype=np.uint8) * 255)
        results = engine.extract_all(buf.tobytes())

        assert len(results) == 2
        np.testing.assert_array_equal(results[0], emb1)
        np.testing.assert_array_equal(results[1], emb2)
