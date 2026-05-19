"""
LLM wrapper supporting Ollama, local HuggingFace Transformers, and the
Hugging Face Inference Providers API.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from threading import Thread

import httpx

from backend.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL_ID,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def get_llm():
    """Load or return the cached local HuggingFace model and tokenizer."""
    global _model, _tokenizer

    if LLM_PROVIDER != "HUGGINGFACE":
        return None, None

    if _model is None or _tokenizer is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local HuggingFace LLM: %s", LLM_MODEL_ID)
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )
        _model.eval()

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        logger.info("Local HuggingFace LLM loaded")

    return _model, _tokenizer


def is_model_available() -> bool:
    """Check whether the selected model provider appears configured."""
    if LLM_PROVIDER == "OLLAMA":
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                tags = response.json().get("models", [])
                return any(
                    t.get("name") == OLLAMA_MODEL or t.get("name") == f"{OLLAMA_MODEL}:latest"
                    for t in tags
                )
        except httpx.RequestError:
            logger.warning("Could not connect to Ollama server.")
        return False

    if LLM_PROVIDER == "HUGGINGFACE_API":
        from backend.config import HF_TOKEN

        return bool(HF_TOKEN)

    return True


def build_prompt_messages(
    question: str,
    context_chunks: list[dict[str, str]],
    history: list[dict[str, str]] = None,
) -> list[dict[str, str]]:
    """Build standard chat messages from context chunks and a user question."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] Source: {source}, Page/Section: {page}\n{text}")

    context_block = "\n\n---\n\n".join(context_parts)
    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above. Cite sources using [1], [2], etc."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend({"role": msg["role"], "content": msg["content"]} for msg in history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _format_hf_prompt(messages: list[dict[str, str]]) -> str:
    """Format messages for TinyLlama ChatML local fallback."""
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<|system|>\n{msg['content']}</s>\n"
        elif msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt


def _generate_stream_hf_local(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Stream generation using local HuggingFace transformers."""
    import torch
    from transformers import TextIteratorStreamer

    model, tokenizer = get_llm()
    prompt = _format_hf_prompt(messages)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
        top_p=0.9,
        repetition_penalty=1.1,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        if new_text:
            yield new_text
    thread.join()


def _generate_stream_ollama(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Stream generation using Ollama API."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    with httpx.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120.0) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "message" in data and "content" in data["message"]:
                yield data["message"]["content"]
            if data.get("done"):
                break


def generate_stream(
    messages: list[dict[str, str]],
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> Generator[str, None, None]:
    """Stream tokens from the selected LLM provider."""
    if LLM_PROVIDER == "OLLAMA":
        yield from _generate_stream_ollama(messages, max_tokens, temperature)
    elif LLM_PROVIDER == "HUGGINGFACE_API":
        from backend.hf_api import chat_stream

        yield from chat_stream(messages, max_tokens=max_tokens, temperature=temperature)
    else:
        yield from _generate_stream_hf_local(messages, max_tokens, temperature)


def generate(
    messages: list[dict[str, str]],
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """Generate a complete response from the selected provider."""
    if LLM_PROVIDER == "OLLAMA":
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120.0)
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "").strip()
            logger.error("Ollama API error: %s - %s", response.status_code, response.text)
            return ""
        except httpx.RequestError as exc:
            logger.error("Ollama API request failed: %s", exc)
            return ""

    if LLM_PROVIDER == "HUGGINGFACE_API":
        from backend.hf_api import chat_complete

        return chat_complete(messages, max_tokens=max_tokens, temperature=temperature)

    import torch

    model, tokenizer = get_llm()
    prompt = _format_hf_prompt(messages)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
