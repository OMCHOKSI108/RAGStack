"""
Small Hugging Face Inference Providers/API adapter.

The rest of the app calls stable local functions and does not need to know
whether the backing provider is local or remote.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Generator, Iterable, List

import httpx
import numpy as np

from backend.config import (
    HF_API_BASE,
    HF_API_URL,
    HF_CHAT_MODEL_ID,
    HF_EMBEDDING_MODEL_ID,
    HF_PROVIDER,
    HF_RERANKER_MODEL_ID,
    HF_ROUTER_URL,
    HF_TIMEOUT,
    HF_TOKEN,
    HF_VERIFIER_MODEL_ID,
)


def _router_base() -> str:
    base = HF_ROUTER_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


logger = logging.getLogger(__name__)


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def _raise_for_missing_token() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required for Hugging Face API providers")


def _chat_payload(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if HF_PROVIDER:
        payload["provider"] = HF_PROVIDER
    return payload


def chat_stream(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    model: str = HF_CHAT_MODEL_ID,
) -> Generator[str, None, None]:
    """Yield generated chat tokens from the HF OpenAI-compatible endpoint."""
    _raise_for_missing_token()
    payload = _chat_payload(messages, model, max_tokens, temperature, stream=True)
    payloads = [payload]
    if "provider" in payload:
        fallback = dict(payload)
        fallback.pop("provider", None)
        payloads.append(fallback)

    last_error = ""
    for attempt in payloads:
        with httpx.stream(
            "POST",
            HF_API_URL,
            headers=_headers(),
            json=attempt,
            timeout=httpx.Timeout(connect=10.0, read=HF_TIMEOUT, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                last_error = response.read().decode("utf-8", errors="replace")
                continue

            for line in response.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    token = delta.get("content")
                    if token:
                        yield token
            return

    raise RuntimeError(f"HF chat API error: {last_error}")


def chat_complete(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    model: str = HF_CHAT_MODEL_ID,
) -> str:
    """Return a complete generated chat response from HF."""
    _raise_for_missing_token()
    payload = _chat_payload(messages, model, max_tokens, temperature, stream=False)
    payloads = [payload]
    if "provider" in payload:
        fallback = dict(payload)
        fallback.pop("provider", None)
        payloads.append(fallback)

    response = None
    for attempt in payloads:
        response = httpx.post(
            HF_API_URL,
            headers=_headers(),
            json=attempt,
            timeout=HF_TIMEOUT,
        )
        if response.status_code == 200:
            break

    if response.status_code != 200:
        raise RuntimeError(f"HF chat API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def _feature_extraction_url(model: str = HF_EMBEDDING_MODEL_ID) -> str:
    provider = HF_PROVIDER or "hf-inference"
    return f"{_router_base()}/{provider}/models/{model}/pipeline/feature-extraction"


def _reranker_url(model: str = HF_RERANKER_MODEL_ID) -> str:
    provider = HF_PROVIDER or "hf-inference"
    return f"{_router_base()}/{provider}/models/{model}"


def embed_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Generate embeddings through the HF feature extraction endpoint."""
    _raise_for_missing_token()
    response = httpx.post(
        _feature_extraction_url(),
        headers=_headers(),
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=HF_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HF embedding API error {response.status_code}: {response.text}")

    data = response.json()
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms

    return arr.astype(np.float32)


def rerank(query: str, documents: List[str]) -> List[float]:
    """
    Score query/document pairs through the configured HF reranker model.
    Raises on unsupported model/API shapes so callers can fall back safely.
    """
    _raise_for_missing_token()
    if not documents:
        return []

    url = _reranker_url()
    payload = {
        "inputs": [{"text": query, "text_pair": doc} for doc in documents],
        "options": {"wait_for_model": True},
    }
    response = httpx.post(url, headers=_headers(), json=payload, timeout=HF_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"HF reranker API error {response.status_code}: {response.text}")

    data = response.json()
    if isinstance(data, dict):
        raise RuntimeError(f"Unexpected HF reranker response: {data}")

    # Some providers wrap the batch in an extra outer list: [[s1, s2, ...]].
    if (
        len(data) == 1
        and isinstance(data[0], list)
        and len(data[0]) == len(documents)
    ):
        data = data[0]

    scores = []
    for item in data:
        if isinstance(item, list):
            score = max(float(label.get("score", 0.0)) for label in item)
        elif isinstance(item, dict):
            score = float(item.get("score", item.get("logit", 0.0)))
        else:
            score = float(item)
        scores.append(score)
    return scores


def verify_answer_with_llm(answer: str, context_texts: Iterable[str]) -> tuple[bool, float]:
    """Ask the configured HF verifier model for a strict groundedness decision."""
    context = "\n\n".join(list(context_texts)[:5])
    messages = [
        {
            "role": "system",
            "content": (
                "You verify document-grounded answers. Respond with only YES or NO. "
                "YES means every factual claim in the answer is supported by the context."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nIs the answer fully supported by the context?",
        },
    ]
    try:
        result = chat_complete(messages, max_tokens=5, temperature=0.0, model=HF_VERIFIER_MODEL_ID).upper()
    except Exception as exc:
        logger.warning("HF verifier failed: %s", exc)
        return False, 0.0
    is_grounded = "YES" in result and "NO" not in result
    return is_grounded, 0.85 if is_grounded else 0.25


def health_check() -> Dict[str, bool | str]:
    """Cheap configuration/reachability check for health endpoint."""
    if not HF_TOKEN:
        return {"ok": False, "detail": "HF_TOKEN is not configured"}
    try:
        response = httpx.get(f"{HF_API_BASE}/status", timeout=5.0)
        if response.status_code < 500:
            return {"ok": True, "detail": "configured"}
        return {"ok": False, "detail": f"status {response.status_code}"}
    except httpx.RequestError as exc:
        return {"ok": False, "detail": str(exc)}
