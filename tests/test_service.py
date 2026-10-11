"""Tests for FaceRecognitionService (integration-level, mocked engine)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_recognition_service.service import FaceRecognitionService


def _random_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(512).astype(np.float32)


class TestServicePipelines:
    @patch("face_recognition_service.face_engine.FaceAnalysis")
    def test_extract_and_encrypt_roundtrip(self, mock_fa_cls):
        """extract_and_encrypt → decrypt_embedding should recover the vector."""
        expected = _random_embedding()
        mock_face = MagicMock()
        mock_face.embedding = expected

        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        svc = FaceRecognitionService()

        import cv2

        _, buf = cv2.imencode(".jpg", np.ones((1, 1, 3), dtype=np.uint8) * 255)
        encrypted = svc.extract_and_encrypt(buf.tobytes())

        assert encrypted is not None

        restored = svc.decrypt_embedding(encrypted)
        np.testing.assert_array_almost_equal(expected, restored)

    @patch("face_recognition_service.face_engine.FaceAnalysis")
    def test_extract_and_encrypt_no_face(self, mock_fa_cls):
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_fa_cls.return_value = mock_app

        svc = FaceRecognitionService()

        import cv2

        _, buf = cv2.imencode(".jpg", np.ones((1, 1, 3), dtype=np.uint8) * 255)
        result = svc.extract_and_encrypt(buf.tobytes())

        assert result is None

    def test_decrypt_and_match_same_embedding(self):
        svc = FaceRecognitionService.__new__(FaceRecognitionService)
        from face_recognition_service.crypto import EmbeddingCrypto
        from face_recognition_service.matcher import EmbeddingMatcher

        svc.crypto = EmbeddingCrypto()
        svc.matcher = EmbeddingMatcher()

        emb = _random_embedding()
        ct_a = svc.crypto.encrypt(emb)
        ct_b = svc.crypto.encrypt(emb)

        is_match, sim = svc.decrypt_and_match(ct_a, ct_b)
        assert is_match is True
        assert sim == pytest.approx(1.0, abs=1e-5)
