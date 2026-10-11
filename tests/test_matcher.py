"""Tests for EmbeddingMatcher (pure numpy — no model required)."""

import numpy as np
import pytest

from face_recognition_service.matcher import EmbeddingMatcher


@pytest.fixture
def matcher():
    return EmbeddingMatcher(threshold=0.6)


def _random_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(512).astype(np.float32)


class TestNormalize:
    def test_valid_vector_is_unit_length(self, matcher):
        vec = _random_embedding()
        normed = matcher._normalize(vec)
        assert normed.shape == (512,)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-5

    def test_wrong_dimension_raises(self, matcher):
        with pytest.raises(ValueError, match="256-d"):
            matcher._normalize(np.zeros(256))

    def test_nan_raises(self, matcher):
        vec = _random_embedding()
        vec[0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            matcher._normalize(vec)

    def test_zero_norm_raises(self, matcher):
        with pytest.raises(ValueError, match="zero norm"):
            matcher._normalize(np.zeros(512))


class TestMatch:
    def test_identical_embeddings_match(self, matcher):
        emb = _random_embedding()
        is_match, sim = matcher.match(emb, emb)
        assert is_match is True
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_do_not_match(self, matcher):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        is_match, sim = matcher.match(a, b)
        assert is_match is False
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_threshold_override(self, matcher):
        a = _random_embedding(1)
        b = _random_embedding(2)
        # With threshold -1.0 any pair should match (cosine sim >= -1)
        is_match, _ = matcher.match(a, b, threshold=-1.0)
        assert is_match is True


class TestFindBestMatch:
    def test_finds_correct_identity(self, matcher):
        target = _random_embedding(10)
        gallery = {
            "alice": _random_embedding(20),
            "bob": target.copy(),
            "carol": _random_embedding(30),
        }
        identity, sim = matcher.find_best_match(target, gallery)
        assert identity == "bob"
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_returns_none_when_below_threshold(self, matcher):
        probe = _random_embedding(1)
        gallery = {"stranger": _random_embedding(99)}
        identity, _ = matcher.find_best_match(probe, gallery, threshold=0.99)
        assert identity is None
