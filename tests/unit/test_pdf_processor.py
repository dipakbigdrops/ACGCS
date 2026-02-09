import io
import pytest
from api.pdf_processor import PDFProcessor
from api.models import Rule


@pytest.fixture
def processor():
    return PDFProcessor()


def test_extract_rules_empty_raises(processor):
    with pytest.raises(ValueError, match="empty"):
        processor.extract_rules(io.BytesIO(b""))


def test_parse_rules_prohibited_semantic(processor):
    text = "Do not use savings claims or discount language."
    rules = processor._parse_rules(text)
    assert len(rules) >= 1
    assert any(r.rule_type == "prohibited_semantic_claim" for r in rules)


def test_parse_rules_required_text(processor):
    text = "Required: You must include the following disclosure: This is a paid ad."
    rules = processor._parse_rules(text)
    required = [r for r in rules if r.rule_type == "required_text"]
    assert len(required) >= 1


def test_create_rule_from_text(processor):
    rule = processor._create_rule_from_text("You cannot use save money or discount claims.")
    assert rule is not None
    assert isinstance(rule, Rule)
