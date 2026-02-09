import pytest
from api.models import Rule, ExtractedText, Violation
from api.rule_engine import RuleEngine


@pytest.fixture
def engine():
    return RuleEngine()


@pytest.fixture
def sample_texts():
    return [
        ExtractedText(text="save 50% money now", source="dom", bounding_box=[0, 0, 100, 20], confidence=1.0),
        ExtractedText(text="Limited time offer", source="dom", bounding_box=[0, 25, 100, 20], confidence=1.0),
        ExtractedText(text="Contact us", source="dom", bounding_box=[0, 50, 100, 20], confidence=1.0),
    ]


def test_check_prohibited_text(engine, sample_texts):
    rule = Rule(guideline_text="no savings", rule_type="prohibited_text", params={"exact_text": "save.*money"})
    violations = engine._check_prohibited_text(rule, sample_texts)
    assert len(violations) == 1
    assert violations[0].rule_type == "prohibited_text"


def test_check_prohibited_text_no_match(engine, sample_texts):
    rule = Rule(guideline_text="x", rule_type="prohibited_text", params={"exact_text": "nonexistentxyz"})
    violations = engine._check_prohibited_text(rule, sample_texts)
    assert len(violations) == 0


def test_check_required_text_found(engine, sample_texts):
    rule = Rule(guideline_text="must have contact", rule_type="required_text", params={"required_text": "contact"})
    violations = engine._check_required_text(rule, sample_texts)
    assert len(violations) == 0


def test_check_required_text_not_found(engine, sample_texts):
    rule = Rule(guideline_text="must have disclaimer", rule_type="required_text", params={"required_text": "disclaimer"})
    violations = engine._check_required_text(rule, sample_texts)
    assert len(violations) == 1
    assert violations[0].rule_type == "required_text"


def test_check_prohibited_semantic_claim_with_threshold(engine, sample_texts):
    semantic_results = {
        "save 50% money now": ("savings_claim", 0.8),
        "save 50% money now".lower(): ("savings_claim", 0.8),
    }
    rule = Rule(
        guideline_text="no savings",
        rule_type="prohibited_semantic_claim",
        params={"banned_categories": ["savings_claim"], "confidence_threshold": 0.7},
    )
    violations = engine._check_prohibited_semantic_claim(rule, sample_texts, semantic_results)
    assert len(violations) == 1
    assert violations[0].confidence == 0.8


def test_evaluate_multiple_rules(engine, sample_texts):
    rules = [
        Rule(guideline_text="no save", rule_type="prohibited_text", params={"exact_text": "save"}),
        Rule(guideline_text="must have contact", rule_type="required_text", params={"required_text": "contact"}),
    ]
    violations = engine.evaluate(rules, sample_texts, {})
    assert any(v.rule_type == "prohibited_text" for v in violations)
    assert not any(v.rule_type == "required_text" for v in violations)
