import os
import pytest
from api.config import (
    MAX_FILE_SIZE,
    DEFAULT_GUIDELINES_ID,
    SEMANTIC_MODEL_VERSION,
    EMBEDDING_CACHE_MAX_SIZE,
    RATE_LIMIT_HEADER,
)


def test_max_file_size_positive():
    assert MAX_FILE_SIZE > 0


def test_default_guidelines_id():
    assert DEFAULT_GUIDELINES_ID == "default"


def test_semantic_model_version():
    assert isinstance(SEMANTIC_MODEL_VERSION, str)
    assert len(SEMANTIC_MODEL_VERSION) >= 1


def test_embedding_cache_max_size_positive():
    assert EMBEDDING_CACHE_MAX_SIZE > 0


def test_rate_limit_header():
    assert isinstance(RATE_LIMIT_HEADER, str)
