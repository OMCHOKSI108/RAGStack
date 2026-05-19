"""
Verification layer for intelligent document reasoning.
Validates that LLM outputs are grounded in the retrieved context.
Performs second-pass validation to prevent hallucination.
Logs verification failures for threshold calibration.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from backend import embeddings
from backend.config import FAITHFULNESS_THRESHOLD, VERIFICATION_PROVIDER

logger = logging.getLogger(__name__)

# Verification failure log file
_VERIFICATION_LOG_PATH = Path(__file__).parent.parent / "evaluation_output" / "verification_failures.jsonl"


def _log_verification_failure(query: str, response: str, confidence: float, context_count: int):
    """Log a verification failure for threshold calibration."""
    try:
        _VERIFICATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "query": query,
            "response_preview": response[:200],
            "confidence": confidence,
            "context_count": context_count,
        }
        with open(_VERIFICATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Verification failure logged: confidence={confidence:.3f}")
    except Exception as e:
        logger.warning(f"Failed to log verification failure: {e}")


def verify_extraction(
    extracted_items: List[Dict],
    context_texts: List[str],
    min_confidence: float = FAITHFULNESS_THRESHOLD,
) -> Tuple[bool, float, List[str]]:
    """
    Verify that extracted items are grounded in the context.
    
    Returns:
        (is_valid, confidence_score, unsupported_items)
    """
    if not extracted_items:
        return True, 1.0, []
    
    unsupported = []
    scores = []
    
    for item in extracted_items:
        item_text = str(item.get("name", "")) + " " + str(item.get("details", ""))
        
        if not item_text.strip():
            continue
        
        # Check if item is supported by any context chunk
        is_supported = False
        best_score = 0.0
        
        for ctx in context_texts:
            # Simple keyword matching first (fast)
            item_words = set(item_text.lower().split())
            ctx_words = set(ctx.lower().split())
            overlap = len(item_words & ctx_words)
            
            if overlap >= 2:  # At least 2 words match
                # Deeper semantic check
                try:
                    item_embedding = embeddings.encode([item_text], normalize=True)
                    ctx_embedding = embeddings.encode([ctx], normalize=True)
                    sim = float(item_embedding @ ctx_embedding.T)
                    
                    if sim > best_score:
                        best_score = sim
                    
                    if sim >= min_confidence:
                        is_supported = True
                        break
                except Exception:
                    if overlap >= 3:
                        is_supported = True
                        best_score = overlap / len(item_words)
                        break
        
        scores.append(best_score)
        if not is_supported:
            unsupported.append(item_text)
    
    if not scores:
        return True, 1.0, []
    
    avg_score = sum(scores) / len(scores)
    is_valid = len(unsupported) == 0
    
    logger.info(
        f"Extraction verification: {len(extracted_items)} items, "
        f"{len(unsupported)} unsupported, confidence={avg_score:.3f}"
    )
    
    return is_valid, avg_score, unsupported


def verify_answer(
    answer: str,
    context_texts: List[str],
    min_confidence: float = FAITHFULNESS_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Verify that an answer is grounded in the context.
    Uses sentence-level claim verification.
    For comprehensive answers, uses a more lenient threshold.
    """
    if not answer.strip() or not context_texts:
        return False, 0.0

    if any(kw in answer.lower() for kw in [
        "the uploaded documents do not contain",
        "not found in document",
        "no information",
        "i don't have",
    ]):
        return True, 1.0

    if VERIFICATION_PROVIDER == "HUGGINGFACE_API":
        from backend.hf_api import verify_answer_with_llm

        return verify_answer_with_llm(answer, context_texts)
    
    # Split answer into claims
    claims = _split_into_claims(answer)
    
    # Filter out meta-statements and "not found" responses
    claims = [
        c for c in claims
        if not c.lower().startswith(("not found", "i don't have", "no information", "the document does not"))
    ]
    
    if not claims:
        return True, 1.0  # "Not found" is a valid response
    
    scores = []
    
    for claim in claims:
        best_score = 0.0
        
        for ctx in context_texts:
            # Fast keyword overlap check
            claim_words = set(claim.lower().split())
            ctx_words = set(ctx.lower().split())
            overlap = len(claim_words & ctx_words)
            overlap_ratio = overlap / max(len(claim_words), 1)
            
            if overlap_ratio >= 0.3:  # At least 30% word overlap
                try:
                    claim_embedding = embeddings.encode([claim], normalize=True)
                    ctx_embedding = embeddings.encode([ctx], normalize=True)
                    sim = float(claim_embedding @ ctx_embedding.T)
                    
                    if sim > best_score:
                        best_score = sim
                except Exception:
                    best_score = max(best_score, overlap_ratio * 0.8)
        
        # For comprehensive answers, use keyword overlap as fallback
        if best_score < min_confidence:
            best_score = max(best_score, min(overlap_ratio * 0.7, 0.5))
        
        scores.append(best_score)
    
    if not scores:
        return False, 0.0
    
    # Use median instead of mean to be more robust to outliers
    scores.sort()
    median_score = scores[len(scores) // 2]
    avg_score = sum(scores) / len(scores)
    
    # For comprehensive answers, use weighted score
    final_score = 0.6 * avg_score + 0.4 * median_score
    
    # More lenient threshold for comprehensive answers
    is_faithful = final_score >= (min_confidence * 0.8)
    
    # Log verification failures for calibration
    if not is_faithful and final_score > 0.1:
        _log_verification_failure(
            query="N/A",  # Caller doesn't pass query, logged at agent level
            response=answer,
            confidence=final_score,
            context_count=len(context_texts),
        )
    
    logger.info(
        f"Answer verification: {len(claims)} claims, "
        f"avg={avg_score:.3f}, median={median_score:.3f}, final={final_score:.3f}, faithful={is_faithful}"
    )
    
    return is_faithful, final_score


def verify_count(
    claimed_count: int,
    entity: str,
    context_texts: List[str],
) -> Tuple[bool, str]:
    """
    Verify a claimed count against the context.
    Returns (is_accurate, status_message).
    """
    if claimed_count == 0:
        # Check if entity is actually absent
        entity_lower = entity.lower()
        found = any(entity_lower in ctx.lower() for ctx in context_texts)
        
        if found:
            return False, f"Entity '{entity}' appears in context but count is 0"
        return True, "Entity not found in context, count 0 is accurate"
    
    # For non-zero counts, verify entity exists in context
    entity_lower = entity.lower()
    found_in_context = any(entity_lower in ctx.lower() for ctx in context_texts)
    
    if not found_in_context:
        return False, f"Entity '{entity}' not found in context"
    
    # Count is plausible if entity exists
    return True, f"Entity '{entity}' found in context, count {claimed_count} is plausible"


def _split_into_claims(text: str) -> List[str]:
    """Split text into individual claim sentences."""
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in ".!?" and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    
    if current.strip() and len(current.strip()) > 10:
        sentences.append(current.strip())
    
    if not sentences:
        sentences = [text.strip()]
    
    return sentences


def extract_citations_from_response(response: str) -> List[int]:
    """Extract citation numbers from response text (e.g., [1], [2])."""
    return [int(x) for x in re.findall(r'\[(\d+)\]', response)]


def validate_citations(
    response: str,
    num_sources: int,
) -> Tuple[bool, List[str]]:
    """Validate that citations in response reference valid sources."""
    citations = extract_citations_from_response(response)
    
    if not citations:
        return False, ["No citations found in response"]
    
    invalid = [str(c) for c in citations if c < 1 or c > num_sources]
    
    if invalid:
        return False, [f"Invalid citation references: [{', '.join(invalid)}]"]
    
    return True, []
