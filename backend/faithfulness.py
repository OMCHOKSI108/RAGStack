"""
Answer faithfulness verification using cross-encoder NLI model.
Checks whether the generated answer is supported by the retrieved context.
Uses microsoft/deberta-v3-base-mnli for entailment classification.
Dynamic threshold: entailment score > 0.70 = verified.
"""

import logging
from typing import List, Tuple

from backend.config import NLI_ENTAILMENT_THRESHOLD, NLI_MODEL
from backend.config import VERIFICATION_PROVIDER

logger = logging.getLogger(__name__)

_nli_model = None
_nli_tokenizer = None


def get_nli_model():
    """Load or return the cached cross-encoder NLI model."""
    global _nli_model, _nli_tokenizer
    if _nli_model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading NLI verifier ({NLI_MODEL}) on {device}...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model.to(device)
        _nli_model.eval()
        logger.info(f"NLI verifier loaded successfully on {device.upper()}")
    return _nli_model, _nli_tokenizer


def _split_into_claims(answer: str) -> List[str]:
    """
    Split an answer into individual claim sentences for verification.
    """
    sentences = []
    current = ""

    for char in answer:
        current += char
        if char in ".!?" and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""

    if current.strip() and len(current.strip()) > 10:
        sentences.append(current.strip())

    if not sentences:
        sentences = [answer.strip()]

    return sentences


def _check_entailment(claim: str, context: str) -> float:
    """
    Check if a claim is entailed by the context using the NLI model.
    Returns entailment probability (0-1).
    """
    model, tokenizer = get_nli_model()
    import torch

    # Truncate context to fit model's max length
    max_context_len = tokenizer.model_max_length - len(tokenizer(claim)["input_ids"]) - 3
    if max_context_len > 0:
        context_tokens = tokenizer(context, truncation=True, max_length=max_context_len, return_tensors="pt")
        context_text = tokenizer.decode(context_tokens["input_ids"][0], skip_special_tokens=True)
    else:
        context_text = context[:512]

    # NLI: premise = context, hypothesis = claim
    inputs = tokenizer(
        context_text,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Labels for MNLI: 0=entailment, 1=neutral, 2=contradiction
    entailment_score = probs[0].item()
    return entailment_score


def verify_faithfulness(
    answer: str,
    context_texts: List[str],
    threshold: float = NLI_ENTAILMENT_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Verify whether the generated answer is supported by the context.

    For each claim sentence in the answer, checks NLI entailment
    against the context using a cross-encoder model.

    Returns:
        (is_faithful, average_entailment_score)
    """
    if not answer.strip() or not context_texts:
        return False, 0.0

    if any(kw in answer.lower() for kw in [
        "not found in document", "no information", "i don't have",
        "the uploaded documents do not contain", "no relevant information"
    ]):
        return True, 1.0

    if VERIFICATION_PROVIDER == "HUGGINGFACE_API":
        from backend.hf_api import verify_answer_with_llm

        return verify_answer_with_llm(answer, context_texts)

    # Skip verification for "not found" responses
    if any(kw in answer.lower() for kw in [
        "not found in document", "no information", "i don't have",
        "the uploaded documents do not contain", "no relevant information"
    ]):
        return True, 1.0

    claims = _split_into_claims(answer)
    full_context = "\n\n".join(context_texts)

    scores = []
    for claim in claims:
        try:
            score = _check_entailment(claim, full_context)
            scores.append(score)
        except Exception as e:
            logger.warning(f"NLI check failed for claim: {e}")
            scores.append(0.0)

    if not scores:
        return False, 0.0

    avg_score = sum(scores) / len(scores)
    is_faithful = avg_score >= threshold

    logger.info(
        f"NLI faithfulness check: {len(claims)} claims, "
        f"avg_entailment={avg_score:.3f}, threshold={threshold}, "
        f"faithful={is_faithful}"
    )

    return is_faithful, avg_score


# The replacement message when answer is not faithful
UNFAITHFUL_RESPONSE = (
    "I don't have enough information in the provided documents "
    "to answer this question reliably."
)
