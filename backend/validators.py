"""
Pydantic validators for structured output from intelligent RAG pipelines.
Enforces strict response formats and validation rules.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class SourceReference(BaseModel):
    """Reference to a source chunk in the document."""
    source_file: str
    page_number: int
    chunk_index: int
    text_snippet: str
    relevance_score: float


class ExtractionResult(BaseModel):
    """Structured result for extraction tasks."""
    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted items from the document"
    )
    total_count: int = Field(
        default=0,
        description="Total number of items found"
    )
    limit_applied: Optional[int] = Field(
        default=None,
        description="Maximum limit applied to results"
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Source references for extracted data"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    not_found_reason: Optional[str] = Field(
        default=None,
        description="Reason if items were not found"
    )

    @validator("confidence")
    def validate_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class CountingResult(BaseModel):
    """Structured result for counting tasks."""
    count: int = Field(default=0, description="The counted value")
    entity: str = Field(default="", description="What was counted")
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_status: str = Field(
        default="verified",
        description="verified, uncertain, or not_found"
    )


class VerificationResult(BaseModel):
    """Structured result for verification tasks."""
    claim: str = Field(default="", description="The claim being verified")
    is_true: bool = Field(default=False, description="Whether the claim is supported")
    evidence: str = Field(default="", description="Supporting evidence from document")
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ComparisonResult(BaseModel):
    """Structured result for comparison tasks."""
    entities: list[str] = Field(default_factory=list, description="Entities being compared")
    comparison_points: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Points of comparison with values for each entity"
    )
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QAResult(BaseModel):
    """Structured result for QA tasks."""
    answer: str = Field(default="", description="The answer to the question")
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_grounded: bool = Field(
        default=True,
        description="Whether the answer is grounded in the document"
    )


class TableExtractionResult(BaseModel):
    """Structured result for table parsing tasks."""
    headers: list[str] = Field(default_factory=list, description="Table column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Table rows")
    source_page: Optional[int] = Field(default=None, description="Page number where table was found")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def create_not_found_response(intent: str, query: str) -> dict:
    """Create a standardized 'not found' response."""
    base = {
        "answer": "Not found in document",
        "sources": [],
        "confidence": 0,
        "is_grounded": True,
    }

    if intent == "extraction":
        return {
            "items": [],
            "total_count": 0,
            "sources": [],
            "confidence": 0,
            "not_found_reason": f"No information matching '{query}' was found in the document.",
        }
    elif intent == "counting":
        return {
            "count": 0,
            "entity": query,
            "sources": [],
            "confidence": 0,
            "verification_status": "not_found",
        }
    elif intent == "verification":
        return {
            "claim": query,
            "is_true": False,
            "evidence": "No supporting evidence found in the document.",
            "sources": [],
            "confidence": 0,
        }
    else:
        return base


def validate_structured_output(data: dict, intent: str) -> dict:
    """Validate and normalize structured output based on intent."""
    try:
        if intent == "extraction":
            result = ExtractionResult(**data)
            return result.model_dump()
        elif intent == "counting":
            result = CountingResult(**data)
            return result.model_dump()
        elif intent == "verification":
            result = VerificationResult(**data)
            return result.model_dump()
        elif intent == "comparison":
            result = ComparisonResult(**data)
            return result.model_dump()
        elif intent == "qa":
            result = QAResult(**data)
            return result.model_dump()
        elif intent == "table_parsing":
            result = TableExtractionResult(**data)
            return result.model_dump()
        else:
            return data
    except Exception as e:
        # If validation fails, return raw data with warning
        return {**data, "_validation_warning": str(e)}
