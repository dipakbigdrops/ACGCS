import io
import json
import os
import tempfile
import pytest
from api.guidelines_store import GuidelinesStore
from api.models import Rule
from api.pdf_processor import PDFProcessor


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_rules():
    return [
        Rule(guideline_text="Do not use savings claims", rule_type="prohibited_semantic_claim", params={"banned_categories": ["savings_claim"]}),
    ]


def test_store_set_and_get(sample_rules):
    store = GuidelinesStore()
    store.set("test-id", sample_rules, "test.pdf", ttl_hours=24)
    data = store.get("test-id")
    assert data is not None
    assert data["filename"] == "test.pdf"
    assert len(data["rules"]) == 1
    assert data["rules"][0].rule_type == "prohibited_semantic_claim"


def test_store_get_nonexistent():
    store = GuidelinesStore()
    assert store.get("nonexistent-id") is None


def test_store_delete(sample_rules):
    store = GuidelinesStore()
    store.set("del-id", sample_rules, "x.pdf")
    assert store.get("del-id") is not None
    store.delete("del-id")
    assert store.get("del-id") is None


def test_store_list_ids(sample_rules):
    store = GuidelinesStore()
    store.set("id1", sample_rules, "a.pdf")
    store.set("id2", sample_rules, "b.pdf")
    ids, total = store.list_ids(page=1, page_size=10)
    assert total >= 2
    assert "id1" in ids or "id2" in ids or len(ids) >= 1


def test_store_persistence(temp_dir, sample_rules):
    store = GuidelinesStore(persistence_dir=temp_dir)
    store.set("persist-id", sample_rules, "p.pdf", pdf_bytes=b"%PDF-1.4")
    meta_path = os.path.join(temp_dir, "persist-id.meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["filename"] == "p.pdf"
    assert "rules_data" in meta
