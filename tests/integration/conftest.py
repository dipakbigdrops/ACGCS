import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _mock_semantic_load(self):
    self.model_loaded = True
    self.model = MagicMock()
    self.model.encode = lambda texts, **kw: np.zeros((len(texts) if isinstance(texts, list) else 1, 384), dtype=np.float32)
    self.category_embeddings = {
        "savings_claim": np.zeros(384, dtype=np.float32),
        "urgency": np.zeros(384, dtype=np.float32),
        "government_association": np.zeros(384, dtype=np.float32),
        "guaranteed_outcome": np.zeros(384, dtype=np.float32),
        "misleading_language": np.zeros(384, dtype=np.float32),
    }


@pytest.fixture
def app():
    with patch("api.semantic_analyzer.SemanticAnalyzer.load_model", _mock_semantic_load):
        from api.main import app
        yield app


@pytest.fixture
async def client(app):
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        async with app.router.lifespan_context(app):
            await ac.get("/health")
            yield ac
