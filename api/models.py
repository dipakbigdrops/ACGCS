from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict


class Violation(BaseModel):
    rule_type: str = Field(..., description="Type of rule that was violated")
    evidence_text: str = Field(..., description="Text from creative that violates the rule")
    source: Literal["ocr", "dom"] = Field(..., description="Source of extracted text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of violation")


class ComplianceResponse(BaseModel):
    message: Union[str, List[Violation]] = Field(..., description="Compliance message or list of violations")


class FileAnalysisResult(BaseModel):
    file_path: str = Field(..., description="Path of the file within the ZIP")
    status: Literal["pass", "fail"] = Field(..., description="Compliance status for this file")
    violations: List[Violation] = Field(default_factory=list, description="List of violations found in this file")
    processed_at: str = Field(..., description="Timestamp when file was processed")


class ZipComplianceResponse(BaseModel):
    overall_status: Literal["pass", "fail"] = Field(..., description="Overall compliance status")
    total_files: int = Field(..., description="Total number of files processed")
    passed_files: int = Field(..., description="Number of files that passed compliance")
    failed_files: int = Field(..., description="Number of files that failed compliance")
    results: List[FileAnalysisResult] = Field(..., description="Individual results for each file")
    summary: str = Field(..., description="Summary message")


class ExtractedText(BaseModel):
    text: str
    source: Literal["ocr", "dom"]
    bounding_box: List[float]
    confidence: Optional[float] = None


class Rule(BaseModel):
    guideline_text: str
    rule_type: str
    params: dict = Field(default_factory=dict)

