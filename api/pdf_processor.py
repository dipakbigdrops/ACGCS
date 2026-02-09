import re
from typing import List
import io
import logging

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from api.models import Rule

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        if not PyPDF2 and not pdfplumber:
            raise ImportError("Either PyPDF2 or pdfplumber must be installed")
    
    def extract_rules(self, pdf_file: io.BytesIO) -> List[Rule]:
        """
        Extract text from PDF and convert to normalized rule schema.
        Uses simple pattern matching to identify rule types from guideline text.
        """
        text = self._extract_text(pdf_file)
        rules = self._parse_rules(text)
        return rules
    
    def _extract_text(self, pdf_file: io.BytesIO) -> str:
        """Extract all text from PDF bytes."""
        pdf_file.seek(0)
        
        if len(pdf_file.read()) == 0:
            raise ValueError("PDF file is empty")
        
        pdf_file.seek(0)
        
        if pdfplumber:
            return self._extract_with_pdfplumber(pdf_file)
        elif PyPDF2:
            return self._extract_with_pypdf2(pdf_file)
        else:
            raise RuntimeError("No PDF library available")
    
    def _extract_with_pdfplumber(self, pdf_file: io.BytesIO) -> str:
        """Extract text using pdfplumber (preferred)."""
        pdf_file.seek(0)
        text_parts = []
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}, trying PyPDF2")
            pdf_file.seek(0)
            return self._extract_with_pypdf2(pdf_file)
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_file: io.BytesIO) -> str:
        """Extract text using PyPDF2 (fallback)."""
        pdf_file.seek(0)
        text_parts = []
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _parse_rules(self, text: str) -> List[Rule]:
        """
        Parse extracted PDF text into structured rules.
        Uses pattern matching to identify different rule types.
        """
        rules = []
        lines = text.split('\n')
        
        current_rule_text = []
        current_rule_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_rule_text:
                    rule_text = ' '.join(current_rule_text)
                    rule = self._create_rule_from_text(rule_text)
                    if rule:
                        rules.append(rule)
                    current_rule_text = []
                    current_rule_type = None
                continue
            
            current_rule_text.append(line)
        
        if current_rule_text:
            rule_text = ' '.join(current_rule_text)
            rule = self._create_rule_from_text(rule_text)
            if rule:
                rules.append(rule)
        
        if not rules:
            if text.strip():
                rules = self._create_default_rules_from_text(text)
            else:
                logger.warning("No text extracted from PDF")
                return []
        
        return rules
    
    def _create_rule_from_text(self, text: str) -> Rule:
        """Convert guideline text into a Rule object."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['prohibited', 'not allowed', 'cannot', 'must not', 'do not']):
            if any(keyword in text_lower for keyword in ['savings', 'save money', 'discount', 'cheaper']):
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_semantic_claim",
                    params={"banned_categories": ["savings_claim"]}
                )
            elif any(keyword in text_lower for keyword in ['urgent', 'limited time', 'act now', 'expires']):
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_semantic_claim",
                    params={"banned_categories": ["urgency"]}
                )
            elif any(keyword in text_lower for keyword in ['government', 'federal', 'irs', 'official']):
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_semantic_claim",
                    params={"banned_categories": ["government_association"]}
                )
            elif any(keyword in text_lower for keyword in ['guarantee', 'guaranteed', 'promise', 'assure']):
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_semantic_claim",
                    params={"banned_categories": ["guaranteed_outcome"]}
                )
            elif any(keyword in text_lower for keyword in ['free', 'no cost', 'risk-free', 'misleading']):
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_semantic_claim",
                    params={"banned_categories": ["misleading_language"]}
                )
            elif any(keyword in text_lower for keyword in ['date', 'calendar', 'festival', 'holiday', 'deadline']):
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_calendar_reference"
                )
            else:
                return Rule(
                    guideline_text=text,
                    rule_type="prohibited_text",
                    params={"exact_text": text}
                )
        
        elif any(keyword in text_lower for keyword in ['required', 'must include', 'should contain', 'disclosure']):
            required_phrase = text
            if len(text) > 80:
                parts = re.split(r'[.!?]+', text)
                for p in reversed(parts):
                    p = p.strip()
                    if len(p) > 15:
                        required_phrase = p
                        break
            return Rule(
                guideline_text=text,
                rule_type="required_text",
                params={"required_text": required_phrase}
            )
        
        return None
    
    def _create_default_rules_from_text(self, text: str) -> List[Rule]:
        """Create default rules if no structured rules found."""
        sentences = re.split(r'[.!?]+', text)
        rules = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                rule = self._create_rule_from_text(sentence)
                if rule:
                    rules.append(rule)
        
        return rules if rules else [
            Rule(
                guideline_text=text[:200],
                rule_type="prohibited_text",
                params={"exact_text": text[:200]}
            )
        ]

