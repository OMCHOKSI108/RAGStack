"""
Agentic orchestrator for intelligent document reasoning.
Routes queries to appropriate task pipelines and manages multi-step reasoning:
understand → retrieve → rerank → extract → validate → verify → return

Features:
- Hard rejection for out-of-scope queries (math, personal, general knowledge)
- Citation granularity: (Source: file.pdf, page X, snippet: "...")
- Post-processing: strips false "document does not provide" lines when verification is high
- LLM-based verification fallback after NLI check
"""

import logging
import json
import re
from typing import AsyncGenerator, Dict, List, Optional

from backend.intent_classifier import classify_intent
from backend.task_prompts import build_task_prompt
from backend.validators import create_not_found_response, validate_structured_output
from backend.verification import verify_extraction, verify_answer, verify_count, validate_citations
from backend import llm

logger = logging.getLogger(__name__)

# Patterns for false "not found" hallucination
_FALSE_NOT_FOUND_PATTERNS = [
    re.compile(r'^The document does not provide', re.I | re.M),
    re.compile(r'^The uploaded documents do not contain', re.I | re.M),
    re.compile(r'^Not found in document', re.I | re.M),
    re.compile(r'^I don.t have enough information', re.I | re.M),
]


class ReasoningAgent:
    """
    Agentic orchestrator for intelligent document reasoning.
    Manages the full reasoning pipeline with verification and validation.
    """
    
    def __init__(self):
        self.intent_cache: Dict[str, Dict] = {}
    
    async def process(
        self,
        query: str,
        context_chunks: List[Dict[str, str]],
        history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Process a query through the full reasoning pipeline.
        
        Yields SSE events:
        - intent: detected intent
        - token: streaming tokens
        - verification: verification results
        - citations: source references
        - structured: structured output (if applicable)
        - done: completion
        """
        # ── Step 1: Intent Classification ────────────────────────────────
        intent_info = classify_intent(query)
        intent = intent_info["intent"]
        strategy = intent_info["strategy"]
        params = intent_info["params"]
        
        yield {"event": "intent", "data": {"intent": intent, "strategy": strategy}}
        
        # ── Step 1b: Hard rejection for out-of-scope queries ─────────────
        if intent == "out_of_scope":
            scope_type = params.get("scope_type", "unknown")
            rejection_msg = (
                f"I can only answer questions about your uploaded documents. "
                f"Your request appears to be a {scope_type} query, which is outside the document scope."
            )
            for char in rejection_msg:
                yield {"event": "token", "data": {"token": char}}
            yield {
                "event": "verification",
                "data": {
                    "is_grounded": True,
                    "confidence": 1.0,
                    "issues": ["Query classified as out-of-scope"],
                    "out_of_scope": True,
                    "scope_type": scope_type,
                }
            }
            yield {"event": "done", "data": {"intent": intent, "out_of_scope": True}}
            return
        
        # ── Step 2: Build Context ────────────────────────────────────────
        context_text = self._format_context(context_chunks)
        context_texts = [c["text"] for c in context_chunks]
        
        # ── Step 3: Build Task-Specific Prompt ───────────────────────────
        messages = build_task_prompt(intent, context_text, query, params)
        
        # ── Step 4: Stream Response ──────────────────────────────────────
        full_response = ""
        
        from backend.config import LLM_MAX_TOKENS_COMPREHENSIVE, LLM_MAX_TOKENS
        max_tokens = LLM_MAX_TOKENS_COMPREHENSIVE if intent in ("qa", "summarization") else LLM_MAX_TOKENS
        
        for token in llm.generate_stream(messages, max_tokens=max_tokens):
            full_response += token
            yield {"event": "token", "data": {"token": token}}
        
        # ── Step 5: Post-process LLM output ──────────────────────────────
        full_response = self._post_process_response(full_response)
        
        # ── Step 6: Verification & Validation ────────────────────────────
        verification_result = await self._verify_response(
            intent, full_response, context_texts, params
        )
        
        yield {"event": "verification", "data": verification_result}
        
        # ── Step 7: Structured Output (if applicable) ────────────────────
        structured_output = None
        if strategy.get("structured_output"):
            structured_output = self._parse_structured_output(
                intent, full_response, context_chunks
            )
            if structured_output:
                yield {"event": "structured", "data": structured_output}
        
        # ── Step 8: Citations with granularity ───────────────────────────
        citations = self._extract_citations(context_chunks)
        if citations:
            yield {"event": "citations", "data": {"citations": citations}}
        
        # ── Step 9: Done ─────────────────────────────────────────────────
        yield {
            "event": "done",
            "data": {
                "intent": intent,
                "verification": verification_result,
                "structured": structured_output,
            },
        }
    
    def _format_context(self, context_chunks: List[Dict[str, str]]) -> str:
        """Format context chunks for prompt inclusion with full granularity."""
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source_file", "unknown")
            page = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            parts.append(f"[{i}] Source: {source}, Page: {page}\n{text}")
        
        return "\n\n---\n\n".join(parts)
    
    def _post_process_response(self, response: str) -> str:
        """
        Post-process LLM output:
        - Strip false 'document does not provide' lines when verification score is high
        - Clean up redundant not-found statements
        """
        lines = response.split("\n")
        cleaned_lines = []
        
        for line in lines:
            is_false_not_found = any(
                p.match(line.strip()) for p in _FALSE_NOT_FOUND_PATTERNS
            )
            if is_false_not_found:
                # Skip this line - it will be replaced by proper abstain message if needed
                continue
            cleaned_lines.append(line)
        
        result = "\n".join(cleaned_lines).strip()
        
        # If we stripped everything, return a proper abstain message
        if not result:
            result = "The uploaded documents do not contain this information."
        
        return result
    
    async def _verify_response(
        self,
        intent: str,
        response: str,
        context_texts: List[str],
        params: Dict,
    ) -> Dict:
        """Verify response against context with NLI + LLM fallback."""
        result = {
            "is_grounded": True,
            "confidence": 0.0,
            "issues": [],
        }
        
        # Check for abstain responses
        if any(kw in response.lower() for kw in [
            "the uploaded documents do not contain",
            "not found in document",
            "no information",
            "i don't have"
        ]):
            result["is_grounded"] = True
            result["confidence"] = 1.0
            return result
        
        # Intent-specific verification
        if intent == "extraction":
            try:
                items = self._extract_items_from_response(response)
                if items:
                    is_valid, confidence, unsupported = verify_extraction(
                        items, context_texts
                    )
                    result["is_grounded"] = is_valid
                    result["confidence"] = confidence
                    if unsupported:
                        result["issues"].append(
                            f"{len(unsupported)} items may not be fully supported by context"
                        )
            except Exception as e:
                logger.warning(f"Extraction verification failed: {e}")
        
        elif intent == "counting":
            try:
                count = self._extract_count_from_response(response)
                entity = params.get("target_entity", "items")
                is_accurate, message = verify_count(count, entity, context_texts)
                result["is_grounded"] = is_accurate
                result["confidence"] = 1.0 if is_accurate else 0.3
                if not is_accurate:
                    result["issues"].append(message)
            except Exception as e:
                logger.warning(f"Counting verification failed: {e}")
        
        else:
            # General answer verification with NLI
            is_faithful, confidence = verify_answer(response, context_texts)
            result["is_grounded"] = is_faithful
            result["confidence"] = confidence
            
            # LLM-based verification fallback
            if not is_faithful and confidence > 0.3:
                llm_verified = await self._llm_verification_fallback(response, context_texts)
                if llm_verified:
                    result["is_grounded"] = True
                    result["confidence"] = max(confidence, 0.6)
                    result["issues"] = [i for i in result["issues"] if "not fully supported" not in i]
                    logger.info("LLM fallback verification passed, overriding NLI result")
            
            if not is_faithful:
                result["issues"].append("Answer may not be fully supported by the document")
        
        return result
    
    async def _llm_verification_fallback(
        self, response: str, context_texts: List[str]
    ) -> bool:
        """
        Second verification check using a small LLM.
        Prompt: "Does the following answer come entirely from the provided context? Answer YES/NO."
        """
        try:
            full_context = "\n\n".join(context_texts[:3])  # Use top 3 chunks for speed
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a verification assistant. "
                        "Determine if the ANSWER comes entirely from the CONTEXT. "
                        "Respond with ONLY YES or NO."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{full_context}\n\n"
                        f"ANSWER:\n{response}\n\n"
                        f"Does the answer come entirely from the provided context? Answer YES/NO."
                    )
                }
            ]
            
            result = llm.generate(messages, max_tokens=5, temperature=0.0).strip().upper()
            is_verified = "YES" in result
            
            logger.info(f"LLM verification fallback: result='{result}', verified={is_verified}")
            return is_verified
            
        except Exception as e:
            logger.warning(f"LLM verification fallback failed: {e}")
            return False
    
    def _parse_structured_output(
        self,
        intent: str,
        response: str,
        context_chunks: List[Dict],
    ) -> Optional[Dict]:
        """Parse structured output from response."""
        try:
            json_match = self._extract_json(response)
            if json_match:
                data = json.loads(json_match)
                validated = validate_structured_output(data, intent)
                return validated
        except Exception as e:
            logger.warning(f"Structured output parsing failed: {e}")
        
        return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text."""
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        
        return None
    
    def _extract_items_from_response(self, response: str) -> List[Dict]:
        """Extract items from extraction response."""
        json_str = self._extract_json(response)
        if json_str:
            try:
                data = json.loads(json_str)
                return data.get("items", [])
            except json.JSONDecodeError:
                pass
        return []
    
    def _extract_count_from_response(self, response: str) -> int:
        """Extract count from counting response."""
        json_str = self._extract_json(response)
        if json_str:
            try:
                data = json.loads(json_str)
                return data.get("count", 0)
            except json.JSONDecodeError:
                pass
        
        import re
        match = re.search(r'count[:\s]*(\d+)', response.lower())
        if match:
            return int(match.group(1))
        
        return 0
    
    def _extract_citations(self, context_chunks: List[Dict]) -> List[Dict]:
        """Extract citation data with full granularity: source file, page, snippet."""
        citations = []
        for i, chunk in enumerate(context_chunks):
            text = chunk.get("text", "")
            snippet = text[:200]
            if len(text) > 200:
                snippet += "..."
            
            citations.append({
                "index": i + 1,
                "source_file": chunk.get("source_file", "unknown"),
                "page_number": chunk.get("page_number", "?"),
                "text_snippet": snippet,
                "citation_label": f"(Source: {chunk.get('source_file', 'unknown')}, page {chunk.get('page_number', '?')}, snippet: \"{snippet[:100]}...\")",
            })
        
        return citations
