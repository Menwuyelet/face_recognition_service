"""Tests for EmbeddingCrypto (no model required)."""

import numpy as np
import pytest

from face_recognition_service.crypto import EmbeddingCrypto


@pytest.fixture
def crypto():
    return EmbeddingCrypto()  # auto-generate key


def _random_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(512).astype(np.float32)


class TestEncryptDecrypt:
    def test_round_trip(self, crypto):
        original = _random_embedding()
        encrypted = crypto.encrypt(original)
        restored = crypto.decrypt(encrypted)
        np.testing.assert_array_almost_equal(original, restored)

    def test_ciphertext_is_not_plaintext(self, crypto):
        original = _random_embedding()
        encrypted = crypto.encrypt(original)
        raw = original.tobytes()
        assert encrypted != raw

    def test_wrong_key_fails(self, crypto):
        original = _random_embedding()
        encrypted = crypto.encrypt(original)

        other_crypto = EmbeddingCrypto()  # different key
        with pytest.raises(Exception):
            other_crypto.decrypt(encrypted)


class TestKeyGeneration:
    def test_generated_key_is_string(self):
        key = EmbeddingCrypto.generate_key()
        assert isinstance(key, str)
        assert len(key) == 44  # Fernet key is 44 chars base64

    def test_auto_generated_key_accessible(self):
        crypto = EmbeddingCrypto()
        assert isinstance(crypto.key, str)
        assert len(crypto.key) == 44

    def test_custom_key_accepted(self):
        key = EmbeddingCrypto.generate_key()
        crypto = EmbeddingCrypto(key=key)
        assert crypto.key == key
