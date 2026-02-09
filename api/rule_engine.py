import re
import logging
from typing import List, Tuple, Dict
from difflib import SequenceMatcher

from api.models import Rule, ExtractedText, Violation

logger = logging.getLogger(__name__)


class RuleEngine:
    def __init__(self):
        pass
    
    def evaluate(
        self, 
        rules: List[Rule], 
        extracted_texts: List[ExtractedText],
        semantic_results: Dict[str, Tuple[str, float]] = None
    ) -> List[Violation]:
        """
        Evaluate all rules against extracted texts and return violations.
        
        Args:
            rules: List of parsed rules from guidelines PDF
            extracted_texts: List of extracted text objects from creative
            semantic_results: Pre-computed semantic classifications mapping text -> (category, confidence)
        """
        if semantic_results is None:
            semantic_results = {}
        
        violations = []
        
        for rule in rules:
            rule_violations = self._check_rule(rule, extracted_texts, semantic_results)
            violations.extend(rule_violations)
        
        return violations
    
    def _check_rule(
        self, 
        rule: Rule, 
        extracted_texts: List[ExtractedText],
        semantic_results: Dict[str, Tuple[str, float]]
    ) -> List[Violation]:
        """Check a single rule against all extracted texts."""
        rule_type = rule.rule_type
        
        if rule_type == "prohibited_text":
            return self._check_prohibited_text(rule, extracted_texts)
        elif rule_type == "fuzzy_text_match":
            return self._check_fuzzy_text_match(rule, extracted_texts)
        elif rule_type == "prohibited_semantic_claim":
            return self._check_prohibited_semantic_claim(rule, extracted_texts, semantic_results)
        elif rule_type == "required_text":
            return self._check_required_text(rule, extracted_texts)
        elif rule_type == "prohibited_calendar_reference":
            return self._check_prohibited_calendar_reference(rule, extracted_texts)
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return []
    
    def _check_prohibited_text(self, rule: Rule, extracted_texts: List[ExtractedText]) -> List[Violation]:
        """Check for prohibited exact text (case-insensitive, regex support)."""
        violations = []
        
        prohibited_pattern = rule.params.get("exact_text", rule.guideline_text)
        
        try:
            pattern = re.compile(prohibited_pattern, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(prohibited_pattern), re.IGNORECASE)
        
        for text_obj in extracted_texts:
            if pattern.search(text_obj.text):
                violations.append(Violation(
                    rule_type="prohibited_text",
                    evidence_text=text_obj.text,
                    source=text_obj.source,
                    confidence=text_obj.confidence or 0.9
                ))
        
        return violations
    
    def _check_fuzzy_text_match(self, rule: Rule, extracted_texts: List[ExtractedText]) -> List[Violation]:
        """Check for fuzzy text matches using token similarity/Levenshtein."""
        violations = []
        
        threshold = rule.params.get("similarity_threshold", 0.8)
        target_text = rule.params.get("target_text", rule.guideline_text).lower()
        
        for text_obj in extracted_texts:
            similarity = self._text_similarity(text_obj.text.lower(), target_text)
            
            if similarity >= threshold:
                violations.append(Violation(
                    rule_type="fuzzy_text_match",
                    evidence_text=text_obj.text,
                    source=text_obj.source,
                    confidence=float(similarity)
                ))
        
        return violations
    
    def _check_prohibited_semantic_claim(
        self,
        rule: Rule,
        extracted_texts: List[ExtractedText],
        semantic_results: Dict[str, Tuple[str, float]]
    ) -> List[Violation]:
        violations = []
        banned_categories = rule.params.get("banned_categories", [])
        if not banned_categories:
            return violations
        threshold = rule.params.get("confidence_threshold", 0.5)

        for text_obj in extracted_texts:
            text_key = text_obj.text
            text_key_lower = text_key.lower() if text_key else ""

            if text_key in semantic_results:
                category, confidence = semantic_results[text_key]
            elif text_key_lower in semantic_results:
                category, confidence = semantic_results[text_key_lower]
            else:
                continue

            if category and category in banned_categories and confidence > threshold:
                violations.append(Violation(
                    rule_type="prohibited_semantic_claim",
                    evidence_text=text_obj.text,
                    source=text_obj.source,
                    confidence=confidence
                ))
        return violations
    
    def _check_required_text(self, rule: Rule, extracted_texts: List[ExtractedText]) -> List[Violation]:
        """
        Check if required text/disclosure is present.
        Returns violation if required text is NOT found.
        """
        required_text = rule.params.get("required_text", rule.guideline_text).lower()
        
        found = False
        for text_obj in extracted_texts:
            if required_text in text_obj.text.lower():
                found = True
                break
        
        if not found:
            return [Violation(
                rule_type="required_text",
                evidence_text="Required text not found in creative",
                source="dom",
                confidence=1.0
            )]
        
        return []
    
    def _check_prohibited_calendar_reference(self, rule: Rule, extracted_texts: List[ExtractedText]) -> List[Violation]:
        """Check for prohibited calendar references (dates, festivals, urgency words)."""
        violations = []
        
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(christmas|new year|thanksgiving|easter|halloween|valentine|independence day)\b',
            r'\b(expires|deadline|ends|closes|last day|final day)\b',
            r'\b(limited time|act now|hurry|urgent|soon|today|tomorrow)\b'
        ]
        
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in date_patterns]
        
        for text_obj in extracted_texts:
            for pattern in compiled_patterns:
                if pattern.search(text_obj.text):
                    violations.append(Violation(
                        rule_type="prohibited_calendar_reference",
                        evidence_text=text_obj.text,
                        source=text_obj.source,
                        confidence=text_obj.confidence or 0.85
                    ))
                    break
        
        return violations
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()

