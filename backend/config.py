"""
Centralized configuration for the RAG pipeline.
All paths, model names, providers, and hyperparameters are defined here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OLLAMA").upper()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "LOCAL").upper()
VECTOR_STORE = os.getenv("VECTOR_STORE", "FAISS").upper()
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "LOCAL").upper()
VERIFICATION_PROVIDER = os.getenv("VERIFICATION_PROVIDER", "LOCAL").upper()
SPARSE_RETRIEVAL = _env_bool("SPARSE_RETRIEVAL", VECTOR_STORE != "PINECONE")

# Local Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Hugging Face Inference Providers/API
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1")
HF_API_BASE = os.getenv("HF_API_BASE", "https://api-inference.huggingface.co")
HF_CHAT_MODEL_ID = os.getenv("HF_CHAT_MODEL_ID", os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
HF_EMBEDDING_MODEL_ID = os.getenv("HF_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
HF_RERANKER_MODEL_ID = os.getenv("HF_RERANKER_MODEL_ID", "BAAI/bge-reranker-base")
HF_VERIFIER_MODEL_ID = os.getenv("HF_VERIFIER_MODEL_ID", HF_CHAT_MODEL_ID)
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))
HF_API_URL = os.getenv("HF_API_URL", f"{HF_ROUTER_URL}/chat/completions")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

# Frontend/backend wiring
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Directory paths
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
INDEX_DIR = PROJECT_ROOT / "index"
MODELS_DIR = PROJECT_ROOT / "models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", HF_EMBEDDING_MODEL_ID)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", HF_RERANKER_MODEL_ID)

# Local Hugging Face Transformers fallback
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS_COMPREHENSIVE = int(os.getenv("LLM_MAX_TOKENS_COMPREHENSIVE", "2048"))

# Document parsing
OCR_CORRECTIONS_ENABLED = _env_bool("OCR_CORRECTIONS_ENABLED", False)

# Query preprocessing
QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", False)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RETRIEVAL_NO_ANSWER_THRESHOLD = float(os.getenv("RETRIEVAL_NO_ANSWER_THRESHOLD", "0.1"))

# Verification
NLI_MODEL = os.getenv("NLI_MODEL", "microsoft/deberta-v3-base-mnli")
NLI_ENTAILMENT_THRESHOLD = float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.70"))
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.25"))

# MMR
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))
MMR_ENABLED = _env_bool("MMR_ENABLED", True)

# Context compression
COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.7"))
MIN_CONTEXT_CHUNKS = int(os.getenv("MIN_CONTEXT_CHUNKS", "3"))

# API
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}

# Rate limiting
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "10/minute")
QUERY_RATE_LIMIT = os.getenv("QUERY_RATE_LIMIT", "30/minute")

SYSTEM_PROMPT = """You are an expert document analyst and research assistant. Provide COMPREHENSIVE, WELL-STRUCTURED, and DETAILED answers based ONLY on the provided document context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context. NEVER use outside knowledge.
2. If the answer is not in the context, say: "Not found in document."
3. Be THOROUGH and COMPREHENSIVE. Cover ALL relevant information from the context.
4. Use MARKDOWN FORMATTING: headings (##, ###), bullet points, numbered lists, bold text.
5. Cite sources inline using [1], [2], etc. matching the context passage numbers.
6. Structure your answer: direct answer first, then detailed sections, then summary.
7. If the question asks about multiple items, describe EACH ONE in its own section.
8. NEVER skip relevant information. Include ALL details from the context.
9. Do NOT use emojis."""
