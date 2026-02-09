import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from api.semantic_analyzer import SemanticAnalyzer, EmbeddingCache


def test_analyzer_raises_when_not_loaded():
    analyzer = SemanticAnalyzer()
    analyzer.model_loaded = False
    with pytest.raises(RuntimeError, match="not loaded"):
        analyzer.classify_text("save money", ["savings_claim"])
    with pytest.raises(RuntimeError, match="not loaded"):
        analyzer.batch_classify(["text"], ["savings_claim"])


def test_batch_classify_empty():
    analyzer = SemanticAnalyzer()
    analyzer.model_loaded = True
    assert analyzer.batch_classify([], ["savings_claim"]) == []


def test_cosine_similarity():
    analyzer = SemanticAnalyzer()
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(analyzer._cosine_similarity(v1, v2) - 1.0) < 1e-5
    v3 = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(analyzer._cosine_similarity(v1, v3)) < 1e-5


def test_embedding_cache():
    cache = EmbeddingCache(max_size=2)
    arr = np.array([1.0, 2.0], dtype=np.float32)
    assert cache.get("a") is None
    cache.set("a", arr)
    assert cache.get("a") is not None
    np.testing.assert_array_almost_equal(cache.get("a"), arr)
