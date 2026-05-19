"""
Production-Grade RAG Evaluation & Benchmarking System.

Automatically evaluates the entire RAG pipeline with comprehensive metrics,
generates visualizations, JSON/CSV reports, and updates README.md.

Usage:
    python test.py                          # Full evaluation (~20 min)
    python test.py --mode benchmark         # Quick benchmark only
    python test.py --mode debug             # Debug mode with verbose logging
    python test.py --queries 50             # Custom query count
    python test.py --no-plots               # Skip visualization generation
"""

import asyncio
import json
import logging
import os
import sys
import time
import csv
import argparse
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import psutil
import httpx
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")
HEALTH_URL = f"{API_BASE}/health"
QUERY_URL = f"{API_BASE}/query/stream"
UPLOAD_URL = f"{API_BASE}/upload"
DOCUMENTS_URL = f"{API_BASE}/documents"

OUTPUT_DIR = Path(__file__).parent / "evaluation_output"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure directories exist
for d in [OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with file and console handlers."""
    level = logging.DEBUG if debug else logging.INFO
    log_file = LOGS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("rag_eval")
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Stores results for a single evaluation query."""
    query: str
    intent: str
    expected_answer: str = ""
    retrieved_docs: List[Dict] = field(default_factory=list)
    generated_answer: str = ""
    citations: List[Dict] = field(default_factory=list)
    verification: Dict = field(default_factory=dict)
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    retrieval_scores: Dict = field(default_factory=dict)
    generation_scores: Dict = field(default_factory=dict)
    citation_scores: Dict = field(default_factory=dict)
    hallucination_detected: bool = False
    error: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    # Retrieval
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Generation
    answer_relevancy: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_rate: float = 0.0
    citation_accuracy: float = 0.0
    source_grounding: float = 0.0
    exact_match: float = 0.0
    f1_score: float = 0.0
    semantic_similarity: float = 0.0
    completeness_score: float = 0.0
    instruction_following: float = 0.0

    # Performance
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0

    # System
    cpu_usage_avg: float = 0.0
    gpu_usage_avg: float = 0.0
    memory_usage_avg: float = 0.0
    peak_memory_mb: float = 0.0

    # Counts
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    unsupported_correctly_rejected: int = 0
    unsupported_total: int = 0


# ── Synthetic Query Generator ─────────────────────────────────────────────────

class SyntheticQueryGenerator:
    """Generates diverse synthetic evaluation queries for RAG testing."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def generate_queries(self, count: int = 100) -> List[Dict]:
        """Generate a diverse set of evaluation queries."""
        queries = []

        # Category 1: Extraction tasks (20%)
        extraction_count = max(1, int(count * 0.20))
        queries.extend(self._generate_extraction_queries(extraction_count))

        # Category 2: Summarization tasks (15%)
        summary_count = max(1, int(count * 0.15))
        queries.extend(self._generate_summarization_queries(summary_count))

        # Category 3: Comparison tasks (10%)
        comparison_count = max(1, int(count * 0.10))
        queries.extend(self._generate_comparison_queries(comparison_count))

        # Category 4: QA tasks (25%)
        qa_count = max(1, int(count * 0.25))
        queries.extend(self._generate_qa_queries(qa_count))

        # Category 5: Unsupported/out-of-document questions (10%)
        unsupported_count = max(1, int(count * 0.10))
        queries.extend(self._generate_unsupported_queries(unsupported_count))

        # Category 6: Adversarial hallucination tests (5%)
        adversarial_count = max(1, int(count * 0.05))
        queries.extend(self._generate_adversarial_queries(adversarial_count))

        # Category 7: Citation verification tests (5%)
        citation_count = max(1, int(count * 0.05))
        queries.extend(self._generate_citation_queries(citation_count))

        # Category 8: Numeric constraint tests (5%)
        constraint_count = max(1, int(count * 0.05))
        queries.extend(self._generate_constraint_queries(constraint_count))

        # Category 9: Multi-turn tests (5%)
        multiturn_count = max(1, int(count * 0.05))
        queries.extend(self._generate_multiturn_queries(multiturn_count))

        # Category 10: Specific real-world test queries (always included)
        queries.extend(self._generate_specific_test_queries())

        # Pad or trim to exact count
        if len(queries) > count:
            queries = queries[:count]
        elif len(queries) < count:
            # Add more QA queries
            remaining = count - len(queries)
            queries.extend(self._generate_qa_queries(remaining))

        self.logger.info(f"Generated {len(queries)} synthetic queries across 9 categories")
        return queries

    def _generate_extraction_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "extract all company names from the document", "intent": "extraction", "category": "entity_extraction"},
            {"query": "list all dates mentioned in the document", "intent": "extraction", "category": "date_extraction"},
            {"query": "extract all email addresses found", "intent": "extraction", "category": "email_extraction"},
            {"query": "get all phone numbers from the document", "intent": "extraction", "category": "phone_extraction"},
            {"query": "extract all project names mentioned", "intent": "extraction", "category": "project_extraction"},
            {"query": "list all person names in the document", "intent": "extraction", "category": "person_extraction"},
            {"query": "extract all locations mentioned", "intent": "extraction", "category": "location_extraction"},
            {"query": "get all URLs or links from the document", "intent": "extraction", "category": "url_extraction"},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_summarization_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "summarize the main points of the document", "intent": "summarization", "category": "general_summary"},
            {"query": "give me an overview of what this document is about", "intent": "summarization", "category": "overview"},
            {"query": "what are the key takeaways from this document?", "intent": "summarization", "category": "key_takeaways"},
            {"query": "provide a brief summary of the document content", "intent": "summarization", "category": "brief_summary"},
            {"query": "what is the main topic of this document?", "intent": "summarization", "category": "main_topic"},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_comparison_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "compare the different projects mentioned in the document", "intent": "comparison", "category": "project_comparison"},
            {"query": "what are the differences between the sections?", "intent": "comparison", "category": "section_comparison"},
            {"query": "compare the advantages and disadvantages mentioned", "intent": "comparison", "category": "pros_cons"},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_qa_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "tell me about the main topic of the document", "intent": "qa", "category": "general_qa"},
            {"query": "what is discussed in this document?", "intent": "qa", "category": "content_qa"},
            {"query": "who are the key people mentioned?", "intent": "qa", "category": "people_qa"},
            {"query": "what are the important dates?", "intent": "qa", "category": "dates_qa"},
            {"query": "explain the main concepts in the document", "intent": "qa", "category": "concepts_qa"},
            {"query": "what recommendations are given in the document?", "intent": "qa", "category": "recommendations_qa"},
            {"query": "what conclusions does the document draw?", "intent": "qa", "category": "conclusions_qa"},
            {"query": "what methodology is described?", "intent": "qa", "category": "methodology_qa"},
            {"query": "what are the results or findings?", "intent": "qa", "category": "results_qa"},
            {"query": "what is the purpose of this document?", "intent": "qa", "category": "purpose_qa"},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_unsupported_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "what is the capital of France?", "intent": "qa", "category": "unsupported", "should_reject": True},
            {"query": "tell me about quantum physics", "intent": "qa", "category": "unsupported", "should_reject": True},
            {"query": "who won the 2024 world cup?", "intent": "qa", "category": "unsupported", "should_reject": True},
            {"query": "what is the meaning of life?", "intent": "qa", "category": "unsupported", "should_reject": True},
            {"query": "explain the theory of relativity", "intent": "qa", "category": "unsupported", "should_reject": True},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_adversarial_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "what does the document say about aliens?", "intent": "qa", "category": "adversarial", "should_reject": True},
            {"query": "extract the secret password from the document", "intent": "extraction", "category": "adversarial", "should_reject": True},
            {"query": "what is the author's social security number?", "intent": "extraction", "category": "adversarial", "should_reject": True},
            {"query": "list all the confidential information", "intent": "extraction", "category": "adversarial", "should_reject": True},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_citation_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "what information is on page 1 of the document?", "intent": "qa", "category": "citation_verification"},
            {"query": "cite the sources for the main claims made", "intent": "qa", "category": "citation_verification"},
            {"query": "where in the document is this information found?", "intent": "qa", "category": "citation_verification"},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_constraint_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "extract maximum 5 items from the document", "intent": "extraction", "category": "numeric_constraint", "max_limit": 5},
            {"query": "list exactly 3 key points", "intent": "extraction", "category": "numeric_constraint", "max_limit": 3},
            {"query": "give me the top 2 most important facts", "intent": "qa", "category": "numeric_constraint", "max_limit": 2},
            {"query": "extract all companies mentioned, but maximum 10", "intent": "extraction", "category": "numeric_constraint", "max_limit": 10},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_multiturn_queries(self, count: int) -> List[Dict]:
        templates = [
            {"query": "tell me about the first project mentioned", "intent": "qa", "category": "multi_turn"},
            {"query": "what about the second one?", "intent": "qa", "category": "multi_turn"},
            {"query": "can you summarize that in one sentence?", "intent": "summarization", "category": "multi_turn"},
        ]
        return [templates[i % len(templates)].copy() for i in range(count)]

    def _generate_specific_test_queries(self) -> List[Dict]:
        """Generate specific real-world test queries for evaluation."""
        return [
            {
                "query": "Which Vadodara-based companies offer Backend internships with a 'Likely' PPO, and what are their stipend ranges?",
                "intent": "extraction",
                "category": "specific_real_world",
                "expected_keywords": ["vadodara", "backend", "ppo", "stipend"],
            },
            {
                "query": "What is the typical stipend range for remote AI startups, and what tech stack is recommended for applications?",
                "intent": "qa",
                "category": "specific_real_world",
                "expected_keywords": ["stipend", "remote", "ai", "tech stack"],
            },
            {
                "query": "What are the top three recommended AI/ML internships for high resume value according to the report?",
                "intent": "extraction",
                "category": "specific_real_world",
                "expected_keywords": ["ai", "ml", "internship", "resume", "top"],
            },
        ]


# ── Retrieval Evaluator ───────────────────────────────────────────────────────

class RetrievalEvaluator:
    """Evaluates retrieval quality metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Dict],
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Compute retrieval metrics for a single query."""
        scores = {}

        # Precision@K
        k = min(5, len(retrieved_docs))
        if k > 0 and relevant_doc_ids:
            retrieved_ids = [d.get("doc_id", "") for d in retrieved_docs[:k]]
            relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant_doc_ids)
            scores["precision_at_k"] = relevant_retrieved / k
            scores["recall_at_k"] = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        else:
            scores["precision_at_k"] = 0.0
            scores["recall_at_k"] = 0.0

        # MRR (Mean Reciprocal Rank)
        scores["mrr"] = self._compute_mrr(retrieved_docs, relevant_doc_ids)

        # nDCG@K
        scores["ndcg"] = self._compute_ndcg(retrieved_docs, relevant_doc_ids, k)

        # Context relevance (based on reranker scores)
        if retrieved_docs:
            rerank_scores = [d.get("relevance_score", 0) for d in retrieved_docs]
            scores["context_precision"] = np.mean(rerank_scores) if rerank_scores else 0.0
        else:
            scores["context_precision"] = 0.0

        return scores

    def _compute_mrr(self, retrieved_docs: List[Dict], relevant_ids: Optional[List[str]]) -> float:
        """Compute Mean Reciprocal Rank."""
        if not relevant_ids:
            return 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc.get("doc_id", "") in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _compute_ndcg(self, retrieved_docs: List[Dict], relevant_ids: Optional[List[str]], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        if not relevant_ids or not retrieved_docs:
            return 0.0

        # Binary relevance
        gains = []
        for doc in retrieved_docs[:k]:
            gains.append(1.0 if doc.get("doc_id", "") in relevant_ids else 0.0)

        # DCG
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

        # Ideal DCG
        ideal_gains = sorted(gains, reverse=True)
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

        return dcg / idcg if idcg > 0 else 0.0


# ── Generation Evaluator ──────────────────────────────────────────────────────

class GenerationEvaluator:
    """Evaluates generation quality metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return self._embedding_model

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        context_chunks: List[Dict],
        expected_answer: str = "",
    ) -> Dict:
        """Compute generation quality metrics."""
        scores = {}

        # Answer relevancy (semantic similarity between query and answer)
        scores["answer_relevancy"] = self._semantic_similarity(query, generated_answer)

        # Semantic similarity to expected answer
        if expected_answer:
            scores["semantic_similarity"] = self._semantic_similarity(expected_answer, generated_answer)
            scores["exact_match"] = 1.0 if generated_answer.strip().lower() == expected_answer.strip().lower() else 0.0
            scores["f1_score"] = self._compute_f1(expected_answer, generated_answer)
        else:
            scores["semantic_similarity"] = 0.0
            scores["exact_match"] = 0.0
            scores["f1_score"] = 0.0

        # Completeness (based on answer length and structure)
        scores["completeness_score"] = self._evaluate_completeness(generated_answer)

        # Instruction following
        scores["instruction_following"] = self._evaluate_instruction_following(query, generated_answer)

        return scores

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            model = self._get_embedding_model()
            embeddings = model.encode([text1, text2], convert_to_numpy=True, normalize_embeddings=True)
            return float(embeddings[0] @ embeddings[1])
        except Exception as e:
            self.logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0

    def _compute_f1(self, expected: str, generated: str) -> float:
        """Compute token-level F1 score."""
        expected_tokens = set(expected.lower().split())
        generated_tokens = set(generated.lower().split())

        if not expected_tokens or not generated_tokens:
            return 0.0

        overlap = expected_tokens & generated_tokens
        precision = len(overlap) / len(generated_tokens)
        recall = len(overlap) / len(expected_tokens)

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _evaluate_completeness(self, answer: str) -> float:
        """Evaluate answer completeness based on structure and length."""
        score = 0.0

        # Length score (longer is generally more complete, up to a point)
        word_count = len(answer.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.2
        elif word_count > 10:
            score += 0.1

        # Structure score
        if "##" in answer or "**" in answer:
            score += 0.2  # Has markdown headings or bold

        if "\n" in answer:
            score += 0.1  # Has line breaks

        if "[" in answer and "]" in answer:
            score += 0.2  # Has citations

        if any(bullet in answer for bullet in ["-", "*", "1.", "2.", "3."]):
            score += 0.2  # Has lists

        return min(score, 1.0)

    def _evaluate_instruction_following(self, query: str, answer: str) -> float:
        """Evaluate if the answer follows instructions in the query."""
        score = 1.0

        # Check for "not found" when appropriate
        if "not found" in query.lower() or "does not exist" in query.lower():
            if "not found" in answer.lower():
                score += 0.5

        # Check for numeric constraints
        import re
        max_match = re.search(r'maximum\s+(\d+)|max\s+(\d+)', query.lower())
        if max_match:
            limit = int(max_match.group(1) or max_match.group(2))
            # Count items in answer (rough estimate)
            items = re.findall(r'\d+\.', answer)
            if len(items) <= limit:
                score += 0.3
            else:
                score -= 0.5

        return max(min(score, 1.0), 0.0)


# ── Citation Evaluator ────────────────────────────────────────────────────────

class CitationEvaluator:
    """Evaluates citation accuracy and source grounding."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def evaluate(
        self,
        generated_answer: str,
        citations: List[Dict],
        context_chunks: List[Dict],
    ) -> Dict:
        """Evaluate citation quality."""
        scores = {}

        # Citation accuracy (do citations reference valid sources?)
        scores["citation_accuracy"] = self._evaluate_citation_accuracy(generated_answer, citations)

        # Source grounding (is the answer grounded in cited sources?)
        scores["source_grounding"] = self._evaluate_source_grounding(generated_answer, context_chunks)

        return scores

    def _evaluate_citation_accuracy(self, answer: str, citations: List[Dict]) -> float:
        """Check if citations in answer reference valid sources."""
        import re
        cited_numbers = [int(x) for x in re.findall(r'\[(\d+)\]', answer)]

        if not cited_numbers:
            return 0.5  # Neutral - no citations

        valid_citations = sum(1 for n in cited_numbers if 1 <= n <= len(citations))
        return valid_citations / len(cited_numbers) if cited_numbers else 0.0

    def _evaluate_source_grounding(self, answer: str, context_chunks: List[Dict]) -> float:
        """Check if answer content is grounded in context."""
        if not context_chunks:
            return 0.0

        context_text = " ".join([c.get("text", "") for c in context_chunks]).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())

        if not answer_words:
            return 0.0

        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)


# ── Hallucination Evaluator ───────────────────────────────────────────────────

class HallucinationEvaluator:
    """Evaluates hallucination rate and unsupported answer detection."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        context_chunks: List[Dict],
        should_reject: bool = False,
    ) -> Dict:
        """Evaluate hallucination and unsupported answer detection."""
        scores = {}

        # Hallucination detection
        scores["hallucination_detected"] = self._detect_hallucination(generated_answer, context_chunks)

        # Unsupported answer rejection
        if should_reject:
            scores["correctly_rejected"] = self._check_rejection(generated_answer)
        else:
            scores["correctly_rejected"] = None

        return scores

    def _detect_hallucination(self, answer: str, context_chunks: List[Dict]) -> bool:
        """Detect if answer contains hallucinated content."""
        if not context_chunks:
            return True  # No context = likely hallucination

        context_text = " ".join([c.get("text", "") for c in context_chunks]).lower()
        answer_sentences = self._split_sentences(answer)

        hallucinated_count = 0
        for sentence in answer_sentences:
            if len(sentence.split()) < 5:
                continue
            # Check if sentence content exists in context
            sentence_words = set(sentence.lower().split())
            context_words = set(context_text.split())
            overlap = len(sentence_words & context_words)
            if overlap < len(sentence_words) * 0.2:  # Less than 20% overlap
                hallucinated_count += 1

        hallucination_rate = hallucinated_count / max(len(answer_sentences), 1)
        return hallucination_rate > 0.3

    def _check_rejection(self, answer: str) -> bool:
        """Check if the model correctly rejected an unsupported question."""
        rejection_phrases = [
            "not found",
            "don't have",
            "do not have",
            "no information",
            "not mentioned",
            "not in the document",
            "cannot answer",
            "not provided",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in rejection_phrases)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


# ── Performance Evaluator ─────────────────────────────────────────────────────

class PerformanceEvaluator:
    """Evaluates system performance metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process = psutil.Process(os.getpid())

    def get_system_metrics(self) -> Dict:
        """Capture current system metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": self.process.memory_info().rss / (1024 * 1024),
        }

        # GPU metrics if available
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpu_utils = []
                gpu_mem = []
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        gpu_utils.append(float(parts[0]))
                        gpu_mem.append(float(parts[1]))
                metrics["gpu_utilization"] = np.mean(gpu_utils) if gpu_utils else 0.0
                metrics["gpu_memory_mb"] = np.mean(gpu_mem) if gpu_mem else 0.0
        except Exception:
            pass

        return metrics


# ── Visualization Manager ─────────────────────────────────────────────────────

class VisualizationManager:
    """Generates all evaluation visualizations."""

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

    def generate_all(self, results_df: pd.DataFrame, metrics: EvaluationMetrics):
        """Generate all visualization plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualizations")
            return

        self.logger.info("Generating visualizations...")

        # Flatten nested score dictionaries
        flat_df = self._flatten_scores(results_df)

        # Generate single comprehensive dashboard
        self._plot_comprehensive_dashboard(flat_df, metrics)

        self.logger.info(f"Visualizations saved to {self.output_dir}")

    def _flatten_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested score dictionaries into DataFrame columns."""
        import re
        
        flat_rows = []
        for _, row in df.iterrows():
            flat_row = {k: v for k, v in row.items() if k not in ["retrieval_scores", "generation_scores", "citation_scores"]}
            
            for col_name in ["retrieval_scores", "generation_scores", "citation_scores"]:
                prefix = col_name.split("_")[0][:3]
                try:
                    raw = row[col_name]
                    if isinstance(raw, str):
                        raw = re.sub(r'np\.float64\(([^)]+)\)', r'\1', raw)
                        raw = ast.literal_eval(raw)
                    if isinstance(raw, dict):
                        flat_row.update({f"{prefix}_{k}": v for k, v in raw.items()})
                except:
                    pass
            
            flat_rows.append(flat_row)
        
        return pd.DataFrame(flat_rows)

    def _plot_comprehensive_dashboard(self, df: pd.DataFrame, metrics: EvaluationMetrics):
        """Plot single comprehensive evaluation dashboard with real values."""
        fig, axes = self.plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle("RAG Pipeline Evaluation Dashboard", fontsize=20, fontweight="bold", y=0.99)

        palette = {
            "primary": "#6366f1",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "info": "#3b82f6",
            "purple": "#8b5cf6",
            "pink": "#ec4899",
            "teal": "#14b8a6",
            "cyan": "#06b6d4",
        }

        # 1. Latency Distribution (top-left)
        if "total_time" in df.columns:
            axes[0, 0].hist(df["total_time"], bins=30, color=palette["primary"], alpha=0.85, edgecolor="white", linewidth=0.5)
            axes[0, 0].axvline(df["total_time"].median(), color=palette["danger"], linestyle="--", linewidth=2.5, label=f"Median: {df['total_time'].median():.1f}s")
            axes[0, 0].axvline(df["total_time"].mean(), color=palette["warning"], linestyle="--", linewidth=2.5, label=f"Mean: {df['total_time'].mean():.1f}s")
            axes[0, 0].set_xlabel("Response Time (seconds)", fontsize=12)
            axes[0, 0].set_ylabel("Frequency", fontsize=12)
            axes[0, 0].set_title("1. Response Latency Distribution", fontsize=14, fontweight="bold")
            axes[0, 0].legend(fontsize=10, framealpha=0.9)
            axes[0, 0].grid(True, alpha=0.25)

        # 2. Generation Quality Metrics (top-center)
        gen_metrics_data = {
            "Answer Relevancy": ("gen_answer_relevancy", palette["info"]),
            "Faithfulness": ("gen_faithfulness_score", palette["purple"]),
            "Completeness": ("gen_completeness_score", palette["success"]),
            "Instruction Following": ("gen_instruction_following", palette["teal"]),
            "Source Grounding": ("cit_source_grounding", palette["pink"]),
        }
        labels, values, colors = [], [], []
        for label, (col, color) in gen_metrics_data.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    labels.append(label)
                    values.append(vals.mean())
                    colors.append(color)
        
        if values:
            bars = axes[0, 1].bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)
            for bar, val in zip(bars, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            axes[0, 1].set_ylabel("Score (0-1)", fontsize=12)
            axes[0, 1].set_title("2. Generation Quality (Average)", fontsize=14, fontweight="bold")
            axes[0, 1].set_ylim(0, 1.15)
            axes[0, 1].tick_params(axis="x", rotation=25, labelsize=10)
            axes[0, 1].grid(True, alpha=0.25, axis="y")

        # 3. Generation Speed (top-right)
        if "tokens_per_second" in df.columns:
            valid = pd.to_numeric(df["tokens_per_second"], errors="coerce")
            valid = valid[valid > 0]
            if not valid.empty:
                axes[0, 2].hist(valid, bins=25, color=palette["cyan"], alpha=0.85, edgecolor="white", linewidth=0.5)
                axes[0, 2].axvline(valid.median(), color=palette["danger"], linestyle="--", linewidth=2.5, label=f"Median: {valid.median():.0f} tok/s")
                axes[0, 2].axvline(valid.mean(), color=palette["warning"], linestyle="--", linewidth=2.5, label=f"Mean: {valid.mean():.0f} tok/s")
                axes[0, 2].set_xlabel("Tokens per Second", fontsize=12)
                axes[0, 2].set_ylabel("Frequency", fontsize=12)
                axes[0, 2].set_title("3. Generation Speed", fontsize=14, fontweight="bold")
                axes[0, 2].legend(fontsize=10, framealpha=0.9)
                axes[0, 2].grid(True, alpha=0.25)

        # 4. Retrieval & Context Metrics (bottom-left)
        ret_data = [
            ("Avg Docs Retrieved", "ret_retrieved_count", palette["primary"], False),
            ("Citation Accuracy", "cit_citation_accuracy", palette["success"], True),
            ("Source Grounding", "cit_source_grounding", palette["pink"], True),
        ]
        ret_labels, ret_values, ret_colors = [], [], []
        for label, col, color, as_pct in ret_data:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    ret_labels.append(label)
                    ret_values.append(vals.mean())
                    ret_colors.append(color)
        
        if ret_values:
            bars = axes[1, 0].bar(ret_labels, ret_values, color=ret_colors, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)
            for bar, val in zip(bars, ret_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + (val * 0.03 if val > 1 else 0.015), f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            axes[1, 0].set_title("4. Retrieval & Context Metrics", fontsize=14, fontweight="bold")
            axes[1, 0].tick_params(axis="x", rotation=25, labelsize=10)
            axes[1, 0].grid(True, alpha=0.25, axis="y")

        # 5. Latency Breakdown (bottom-center)
        breakdown_metrics = ["avg_retrieval_time", "avg_generation_time", "avg_total_time"]
        breakdown_labels = ["Retrieval", "Generation", "Total"]
        breakdown_values = [getattr(metrics, m, 0) * 1000 for m in breakdown_metrics]
        breakdown_colors = [palette["info"], palette["purple"], palette["warning"]]
        
        bars = axes[1, 1].bar(breakdown_labels, breakdown_values, color=breakdown_colors, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)
        for bar, val in zip(bars, breakdown_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, f"{val:.0f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[1, 1].set_ylabel("Time (milliseconds)", fontsize=12)
        axes[1, 1].set_title("5. Latency Breakdown (Average)", fontsize=14, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.25, axis="y")

        # 6. System Performance (bottom-right)
        perf_categories = ["Success Rate", "Citation Accuracy", "Hallucination Rate"]
        perf_values = [
            metrics.successful_queries / max(metrics.total_queries, 1) * 100,
            getattr(metrics, "citation_accuracy", 0) * 100,
            metrics.hallucination_rate * 100
        ]
        perf_colors = [palette["success"], palette["info"], palette["danger"]]
        
        bars = axes[1, 2].bar(perf_categories, perf_values, color=perf_colors, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)
        for bar, val in zip(bars, perf_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[1, 2].set_ylabel("Percentage (%)", fontsize=12)
        axes[1, 2].set_title("6. System Performance Overview", fontsize=14, fontweight="bold")
        axes[1, 2].set_ylim(0, 115)
        axes[1, 2].grid(True, alpha=0.25, axis="y")

        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "evaluation_dashboard.png", dpi=150, bbox_inches="tight")
        self.plt.close()


# ── Report Generator ──────────────────────────────────────────────────────────

class ReportGenerator:
    """Generates evaluation reports in multiple formats."""

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

    def generate_json_reports(self, metrics: EvaluationMetrics, results: List[Dict]):
        """Generate JSON report files."""
        # evaluation_results.json
        eval_results = {
            "evaluation_date": datetime.now().isoformat(),
            "total_queries": metrics.total_queries,
            "successful_queries": metrics.successful_queries,
            "failed_queries": metrics.failed_queries,
            "metrics": asdict(metrics),
            "results": results[:100],  # First 100 for readability
        }
        self._save_json(eval_results, "evaluation_results.json")

        # metrics_summary.json
        metrics_summary = {
            "retrieval": {
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "mrr": metrics.mrr,
                "ndcg": metrics.ndcg,
                "context_precision": metrics.context_precision,
                "context_recall": metrics.context_recall,
            },
            "generation": {
                "answer_relevancy": metrics.answer_relevancy,
                "faithfulness_score": metrics.faithfulness_score,
                "hallucination_rate": metrics.hallucination_rate,
                "citation_accuracy": metrics.citation_accuracy,
                "source_grounding": metrics.source_grounding,
                "exact_match": metrics.exact_match,
                "f1_score": metrics.f1_score,
                "semantic_similarity": metrics.semantic_similarity,
                "completeness_score": metrics.completeness_score,
                "instruction_following": metrics.instruction_following,
            },
            "performance": {
                "avg_retrieval_time": metrics.avg_retrieval_time,
                "avg_generation_time": metrics.avg_generation_time,
                "avg_total_time": metrics.avg_total_time,
                "avg_tokens_per_second": metrics.avg_tokens_per_second,
                "p50_latency": metrics.p50_latency,
                "p95_latency": metrics.p95_latency,
                "p99_latency": metrics.p99_latency,
                "throughput": metrics.throughput,
            },
            "system": {
                "cpu_usage_avg": metrics.cpu_usage_avg,
                "gpu_usage_avg": metrics.gpu_usage_avg,
                "memory_usage_avg": metrics.memory_usage_avg,
                "peak_memory_mb": metrics.peak_memory_mb,
            },
        }
        self._save_json(metrics_summary, "metrics_summary.json")

        # benchmark_results.json
        benchmark = {
            "benchmark_date": datetime.now().isoformat(),
            "total_queries": metrics.total_queries,
            "total_time_seconds": metrics.avg_total_time * metrics.total_queries,
            "queries_per_minute": metrics.throughput * 60,
            "avg_latency_ms": metrics.avg_total_time * 1000,
            "p50_latency_ms": metrics.p50_latency * 1000,
            "p95_latency_ms": metrics.p95_latency * 1000,
            "p99_latency_ms": metrics.p99_latency * 1000,
        }
        self._save_json(benchmark, "benchmark_results.json")

    def generate_csv_report(self, results: List[Dict]):
        """Generate CSV report."""
        if not results:
            return

        csv_path = self.output_dir / "evaluation_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"CSV report saved to {csv_path}")

    def generate_markdown_report(self, metrics: EvaluationMetrics, system_info: Dict):
        """Generate markdown evaluation report."""
        report_path = REPORTS_DIR / "evaluation_report.md"

        md = f"""# RAG Pipeline Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Queries | {metrics.total_queries} |
| Success Rate | {metrics.successful_queries / max(metrics.total_queries, 1) * 100:.1f}% |
| Avg Latency | {metrics.avg_total_time * 1000:.0f}ms |
| P95 Latency | {metrics.p95_latency * 1000:.0f}ms |
| Throughput | {metrics.throughput:.2f} queries/sec |

## Retrieval Metrics

| Metric | Score |
|--------|-------|
| Precision@K | {metrics.precision_at_k:.4f} |
| Recall@K | {metrics.recall_at_k:.4f} |
| MRR | {metrics.mrr:.4f} |
| nDCG | {metrics.ndcg:.4f} |
| Context Precision | {metrics.context_precision:.4f} |
| Context Recall | {metrics.context_recall:.4f} |

## Generation Metrics

| Metric | Score |
|--------|-------|
| Answer Relevancy | {metrics.answer_relevancy:.4f} |
| Faithfulness Score | {metrics.faithfulness_score:.4f} |
| Hallucination Rate | {metrics.hallucination_rate:.4f} |
| Citation Accuracy | {metrics.citation_accuracy:.4f} |
| Source Grounding | {metrics.source_grounding:.4f} |
| Exact Match | {metrics.exact_match:.4f} |
| F1 Score | {metrics.f1_score:.4f} |
| Semantic Similarity | {metrics.semantic_similarity:.4f} |
| Completeness Score | {metrics.completeness_score:.4f} |
| Instruction Following | {metrics.instruction_following:.4f} |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Avg Retrieval Time | {metrics.avg_retrieval_time * 1000:.0f}ms |
| Avg Generation Time | {metrics.avg_generation_time * 1000:.0f}ms |
| Avg Total Time | {metrics.avg_total_time * 1000:.0f}ms |
| P50 Latency | {metrics.p50_latency * 1000:.0f}ms |
| P95 Latency | {metrics.p95_latency * 1000:.0f}ms |
| P99 Latency | {metrics.p99_latency * 1000:.0f}ms |
| Throughput | {metrics.throughput:.2f} queries/sec |
| Avg Tokens/sec | {metrics.avg_tokens_per_second:.1f} |

## System Metrics

| Metric | Value |
|--------|-------|
| CPU Usage (avg) | {metrics.cpu_usage_avg:.1f}% |
| GPU Usage (avg) | {metrics.gpu_usage_avg:.1f}% |
| Memory Usage (avg) | {metrics.memory_usage_avg:.1f}% |
| Peak Memory | {metrics.peak_memory_mb:.0f} MB |

## System Information

| Property | Value |
|----------|-------|
| Platform | {system_info.get('platform', 'N/A')} |
| Python Version | {system_info.get('python_version', 'N/A')} |
| CPU Cores | {system_info.get('cpu_count', 'N/A')} |
| Total RAM | {system_info.get('total_ram_gb', 'N/A')} GB |
| GPU Available | {system_info.get('gpu_available', 'N/A')} |

## Evaluation Methodology

This evaluation uses a comprehensive multi-dimensional approach:

1. **Retrieval Quality**: Precision@K, Recall@K, MRR, nDCG measure how well the system finds relevant documents.
2. **Generation Quality**: Answer relevancy, faithfulness, completeness, and instruction following assess response quality.
3. **Citation Accuracy**: Verifies that citations reference valid sources and content is grounded.
4. **Hallucination Detection**: Measures rate of unsupported claims and ability to reject out-of-document questions.
5. **Performance**: Latency percentiles, throughput, and system resource utilization.

## Visualizations

![Evaluation Dashboard](../plots/evaluation_dashboard.png)
![Latency Distribution](../plots/latency_histogram.png)
![Retrieval Scores](../plots/retrieval_scores.png)
![Faithfulness](../plots/faithfulness_chart.png)
![Citation Accuracy](../plots/citation_accuracy.png)
![Token Speed](../plots/token_speed.png)
![Confusion Matrix](../plots/confusion_matrix.png)
"""
        report_path.write_text(md, encoding="utf-8")
        self.logger.info(f"Markdown report saved to {report_path}")

    def _save_json(self, data: Dict, filename: str):
        """Save data as JSON file."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info(f"JSON report saved to {path}")


# ── Main Evaluator ────────────────────────────────────────────────────────────

class RAGEvaluator:
    """Main orchestrator for RAG evaluation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = setup_logging(args.debug)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0))

        # Evaluators
        self.retrieval_eval = RetrievalEvaluator(self.logger)
        self.generation_eval = GenerationEvaluator(self.logger)
        self.citation_eval = CitationEvaluator(self.logger)
        self.hallucination_eval = HallucinationEvaluator(self.logger)
        self.performance_eval = PerformanceEvaluator(self.logger)
        self.query_generator = SyntheticQueryGenerator(self.logger)
        self.viz_manager = VisualizationManager(PLOTS_DIR, self.logger)
        self.report_gen = ReportGenerator(OUTPUT_DIR, self.logger)

        # Results storage
        self.results: List[QueryResult] = []
        self.metrics = EvaluationMetrics()

    async def run(self):
        """Run the complete evaluation pipeline."""
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("  RAG Pipeline Evaluation System")
        self.logger.info("=" * 60)

        # Check backend health
        if not await self._check_health():
            self.logger.error("Backend is not available. Aborting evaluation.")
            return

        # Get system info
        system_info = self._get_system_info()
        self.logger.info(f"System: {system_info['platform']} | Python: {system_info['python_version']}")
        self.logger.info(f"GPU: {'Available' if system_info['gpu_available'] else 'Not available'}")

        # Generate queries
        query_count = self.args.queries
        self.logger.info(f"Generating {query_count} synthetic evaluation queries...")
        queries = self.query_generator.generate_queries(query_count)

        # Run evaluation
        self.logger.info(f"Starting evaluation of {len(queries)} queries...")
        await self._evaluate_queries(queries)

        # Compute aggregated metrics
        self._compute_aggregated_metrics()

        # Generate reports
        self.logger.info("Generating reports and visualizations...")
        results_dicts = [asdict(r) for r in self.results]
        results_df = pd.DataFrame(results_dicts)

        self.report_gen.generate_json_reports(self.metrics, results_dicts)
        self.report_gen.generate_csv_report(results_dicts)
        self.report_gen.generate_markdown_report(self.metrics, system_info)

        if not self.args.no_plots:
            self.viz_manager.generate_all(results_df, self.metrics)

        # Update README
        self._update_readme(system_info)

        # Print summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        self.logger.info(f"\nEvaluation complete in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
        self.logger.info(f"Results saved to: {OUTPUT_DIR}")

    async def _check_health(self) -> bool:
        """Check if backend is healthy."""
        try:
            response = await self.client.get(HEALTH_URL, timeout=10.0)
            if response.status_code == 200:
                health = response.json()
                self.logger.info(f"Backend healthy: {health.get('document_count', 0)} documents, {health.get('total_chunks', 0)} chunks")
                return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
        return False

    def _get_system_info(self) -> Dict:
        """Get system information."""
        import platform
        import torch

        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "gpu_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()

        return info

    async def _evaluate_queries(self, queries: List[Dict]):
        """Evaluate all queries with progress tracking."""
        for i, query_info in enumerate(tqdm(queries, desc="Evaluating queries")):
            try:
                result = await self._evaluate_single_query(query_info)
                self.results.append(result)
            except Exception as e:
                self.logger.warning(f"Query {i+1} failed: {e}")
                self.results.append(QueryResult(
                    query=query_info.get("query", ""),
                    intent=query_info.get("intent", "qa"),
                    error=str(e),
                ))

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)

    async def _evaluate_single_query(self, query_info: Dict) -> QueryResult:
        """Evaluate a single query through the RAG pipeline."""
        query = query_info.get("query", "")
        intent = query_info.get("intent", "qa")
        should_reject = query_info.get("should_reject", False)
        max_limit = query_info.get("max_limit", None)

        result = QueryResult(query=query, intent=intent)
        start_time = time.time()

        # Capture system metrics before query
        sys_before = self.performance_eval.get_system_metrics()

        # Send query to backend
        async with self.client.stream(
            "POST",
            QUERY_URL,
            json={"question": query},
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}"
                return result

            full_answer = ""
            citations = []
            verification = {}
            token_count = 0
            retrieval_done = False
            retrieval_time = 0.0
            first_token_time = None

            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    if current_event == "token":
                        token = data.get("token", "")
                        full_answer += token
                        token_count += 1
                        if first_token_time is None:
                            retrieval_time = time.time() - start_time
                            retrieval_done = True
                            first_token_time = time.time()
                    elif current_event == "citations":
                        citations = data.get("citations", [])
                    elif current_event == "verification":
                        verification = data
                    elif current_event == "intent":
                        pass  # Intent detected
                    elif current_event == "done":
                        break

        # Calculate times
        total_time = time.time() - start_time
        generation_time = total_time - retrieval_time if retrieval_done else total_time
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0

        result.retrieved_docs = citations
        result.generated_answer = full_answer
        result.citations = citations
        result.verification = verification
        result.retrieval_time = retrieval_time
        result.generation_time = generation_time
        result.total_time = total_time
        result.tokens_generated = token_count
        result.tokens_per_second = tokens_per_second

        # Evaluate retrieval
        context_chunks = [
            {"text": c.get("text_snippet", ""), "source_file": c.get("source_file", "")}
            for c in citations
        ]
        result.retrieval_scores = self.retrieval_eval.evaluate(query, citations)

        # Evaluate generation
        result.generation_scores = self.generation_eval.evaluate(
            query, full_answer, context_chunks
        )

        # Evaluate citations
        result.citation_scores = self.citation_eval.evaluate(
            full_answer, citations, context_chunks
        )

        # Evaluate hallucination
        hallucination_result = self.hallucination_eval.evaluate(
            query, full_answer, context_chunks, should_reject
        )
        result.hallucination_detected = hallucination_result.get("hallucination_detected", False)
        if hallucination_result.get("correctly_rejected") is not None:
            result.retrieval_scores["correctly_rejected"] = hallucination_result["correctly_rejected"]

        # Capture system metrics after query
        sys_after = self.performance_eval.get_system_metrics()

        # Store additional metrics for aggregation
        result.retrieval_scores["retrieved_count"] = len(citations)
        result.generation_scores["faithfulness_score"] = verification.get("confidence", 0)

        return result

    def _compute_aggregated_metrics(self):
        """Compute aggregated metrics from all results."""
        valid_results = [r for r in self.results if not r.error]

        if not valid_results:
            self.logger.warning("No valid results to aggregate")
            return

        self.metrics.total_queries = len(self.results)
        self.metrics.successful_queries = len(valid_results)
        self.metrics.failed_queries = len(self.results) - len(valid_results)

        # Retrieval metrics
        self.metrics.precision_at_k = np.mean([r.retrieval_scores.get("precision_at_k", 0) for r in valid_results])
        self.metrics.recall_at_k = np.mean([r.retrieval_scores.get("recall_at_k", 0) for r in valid_results])
        self.metrics.mrr = np.mean([r.retrieval_scores.get("mrr", 0) for r in valid_results])
        self.metrics.ndcg = np.mean([r.retrieval_scores.get("ndcg", 0) for r in valid_results])
        self.metrics.context_precision = np.mean([r.retrieval_scores.get("context_precision", 0) for r in valid_results])

        # Generation metrics
        self.metrics.answer_relevancy = np.mean([r.generation_scores.get("answer_relevancy", 0) for r in valid_results])
        self.metrics.faithfulness_score = np.mean([r.generation_scores.get("faithfulness_score", 0) for r in valid_results])
        self.metrics.citation_accuracy = np.mean([r.citation_scores.get("citation_accuracy", 0) for r in valid_results])
        self.metrics.source_grounding = np.mean([r.citation_scores.get("source_grounding", 0) for r in valid_results])
        self.metrics.exact_match = np.mean([r.generation_scores.get("exact_match", 0) for r in valid_results])
        self.metrics.f1_score = np.mean([r.generation_scores.get("f1_score", 0) for r in valid_results])
        self.metrics.semantic_similarity = np.mean([r.generation_scores.get("semantic_similarity", 0) for r in valid_results])
        self.metrics.completeness_score = np.mean([r.generation_scores.get("completeness_score", 0) for r in valid_results])
        self.metrics.instruction_following = np.mean([r.generation_scores.get("instruction_following", 0) for r in valid_results])

        # Hallucination rate
        hallucinated = sum(1 for r in valid_results if r.hallucination_detected)
        self.metrics.hallucination_rate = hallucinated / len(valid_results) if valid_results else 0.0

        # Unsupported question rejection
        unsupported_results = [r for r in valid_results if r.retrieval_scores.get("correctly_rejected") is not None]
        if unsupported_results:
            self.metrics.unsupported_total = len(unsupported_results)
            self.metrics.unsupported_correctly_rejected = sum(
                1 for r in unsupported_results if r.retrieval_scores.get("correctly_rejected", False)
            )

        # Performance metrics
        total_times = [r.total_time for r in valid_results]
        self.metrics.avg_retrieval_time = np.mean([r.retrieval_time for r in valid_results])
        self.metrics.avg_generation_time = np.mean([r.generation_time for r in valid_results])
        self.metrics.avg_total_time = np.mean(total_times)
        self.metrics.avg_tokens_per_second = np.mean([r.tokens_per_second for r in valid_results if r.tokens_per_second > 0])

        if total_times:
            sorted_times = sorted(total_times)
            self.metrics.p50_latency = sorted_times[len(sorted_times) // 2]
            self.metrics.p95_latency = sorted_times[int(len(sorted_times) * 0.95)]
            self.metrics.p99_latency = sorted_times[int(len(sorted_times) * 0.99)]
            self.metrics.throughput = len(valid_results) / sum(total_times) if sum(total_times) > 0 else 0.0

    def _update_readme(self, system_info: Dict):
        """Update README.md with evaluation results."""
        readme_path = Path(__file__).parent / "README.md"

        if not readme_path.exists():
            self.logger.warning("README.md not found, skipping update")
            return

        content = readme_path.read_text(encoding="utf-8")

        # Check if evaluation section already exists
        if "## Evaluation Results" in content:
            # Remove existing section
            sections = content.split("## Evaluation Results")
            content = sections[0]

        # Append evaluation section
        eval_section = f"""## Evaluation Results

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Benchmark Summary

| Metric | Value |
|--------|-------|
| Total Queries | {self.metrics.total_queries} |
| Success Rate | {self.metrics.successful_queries / max(self.metrics.total_queries, 1) * 100:.1f}% |
| Avg Latency | {self.metrics.avg_total_time * 1000:.0f}ms |
| P95 Latency | {self.metrics.p95_latency * 1000:.0f}ms |
| Throughput | {self.metrics.throughput:.2f} queries/sec |
| Hallucination Rate | {self.metrics.hallucination_rate:.2%} |

### Retrieval Performance

| Metric | Score |
|--------|-------|
| Precision@K | {self.metrics.precision_at_k:.4f} |
| Recall@K | {self.metrics.recall_at_k:.4f} |
| MRR | {self.metrics.mrr:.4f} |
| nDCG | {self.metrics.ndcg:.4f} |

### Generation Quality

| Metric | Score |
|--------|-------|
| Answer Relevancy | {self.metrics.answer_relevancy:.4f} |
| Faithfulness | {self.metrics.faithfulness_score:.4f} |
| Citation Accuracy | {self.metrics.citation_accuracy:.4f} |
| Semantic Similarity | {self.metrics.semantic_similarity:.4f} |

### System Information

| Property | Value |
|----------|-------|
| Platform | {system_info.get('platform', 'N/A')} |
| Python | {system_info.get('python_version', 'N/A')} |
| CPU Cores | {system_info.get('cpu_count', 'N/A')} |
| RAM | {system_info.get('total_ram_gb', 'N/A')} GB |
| GPU | {'Available' if system_info.get('gpu_available') else 'Not available'} |

### Evaluation Plots

![Dashboard](evaluation_output/plots/evaluation_dashboard.png)
![Latency](evaluation_output/plots/latency_histogram.png)
![Retrieval](evaluation_output/plots/retrieval_scores.png)
![Faithfulness](evaluation_output/plots/faithfulness_chart.png)

### Methodology

- **Synthetic Queries**: {self.metrics.total_queries} queries across 9 categories
- **Categories**: Extraction, Summarization, Comparison, QA, Unsupported, Adversarial, Citation, Constraints, Multi-turn
- **Metrics**: 22+ metrics covering retrieval, generation, citations, hallucination, and performance
- **Duration**: ~{int(self.metrics.total_queries * 12 / 60)} minutes estimated

---
"""
        readme_path.write_text(content + eval_section, encoding="utf-8")
        self.logger.info("README.md updated with evaluation results")

    def _print_summary(self, elapsed: float):
        """Print terminal summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("  EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"  Total Queries:        {self.metrics.total_queries}")
        self.logger.info(f"  Success Rate:         {self.metrics.successful_queries / max(self.metrics.total_queries, 1) * 100:.1f}%")
        self.logger.info(f"  Avg Latency:          {self.metrics.avg_total_time * 1000:.0f}ms")
        self.logger.info(f"  P95 Latency:          {self.metrics.p95_latency * 1000:.0f}ms")
        self.logger.info(f"  Throughput:           {self.metrics.throughput:.2f} queries/sec")
        self.logger.info(f"  Hallucination Rate:   {self.metrics.hallucination_rate:.2%}")
        self.logger.info(f"  Faithfulness:         {self.metrics.faithfulness_score:.4f}")
        self.logger.info(f"  Citation Accuracy:    {self.metrics.citation_accuracy:.4f}")
        self.logger.info(f"  Precision@K:          {self.metrics.precision_at_k:.4f}")
        self.logger.info(f"  Recall@K:             {self.metrics.recall_at_k:.4f}")
        self.logger.info(f"  MRR:                  {self.metrics.mrr:.4f}")
        self.logger.info(f"  nDCG:                 {self.metrics.ndcg:.4f}")
        self.logger.info(f"  Tokens/sec:           {self.metrics.avg_tokens_per_second:.1f}")
        self.logger.info(f"  Duration:             {elapsed:.0f}s ({elapsed/60:.1f} min)")
        self.logger.info("=" * 60)
        self.logger.info(f"\n  Reports: {OUTPUT_DIR}")
        self.logger.info(f"  Plots:   {PLOTS_DIR}")
        self.logger.info("=" * 60)


# ── CLI Arguments ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation & Benchmarking System")
    parser.add_argument(
        "--mode",
        choices=["full", "benchmark", "debug"],
        default="full",
        help="Evaluation mode: full (default), benchmark (quick), debug (verbose)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of synthetic queries to generate (default: 100)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE,
        help="Backend API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def main():
    """Main entry point."""
    args = parse_args()

    # Adjust settings based on mode
    if args.mode == "benchmark":
        args.queries = 20
        args.no_plots = True
    elif args.mode == "debug":
        args.queries = 10
        args.no_plots = False

    # Override API URL if provided
    global API_BASE, HEALTH_URL, QUERY_URL
    if args.api_url != API_BASE:
        API_BASE = args.api_url
        HEALTH_URL = f"{API_BASE}/health"
        QUERY_URL = f"{API_BASE}/query/stream"

    evaluator = RAGEvaluator(args)
    await evaluator.run()


if __name__ == "__main__":
    asyncio.run(main())
