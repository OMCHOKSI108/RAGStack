"""
FastAPI application for the RAG pipeline.
Provides endpoints for document upload, streaming queries,
document management, and health checks.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse

from backend.config import (
    ALLOWED_EXTENSIONS,
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
    QUERY_RATE_LIMIT,
    RERANK_PROVIDER,
    UPLOAD_DIR,
    UPLOAD_RATE_LIMIT,
    VECTOR_STORE,
    VERIFICATION_PROVIDER,
)
from backend.models import HealthResponse, QueryRequest, UploadResponse
from backend.rag_pipeline import RAGPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy httpx INFO logs (404s for optional HuggingFace files)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Pipeline singleton ────────────────────────────────────────────────────────
pipeline = RAGPipeline()


# ── Lifespan: load models and indices on startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG pipeline...")
    pipeline.load_indices()

    # Load models in a thread to avoid blocking
    await asyncio.to_thread(pipeline.load_models)
    logger.info("RAG pipeline ready")
    yield
    logger.info("Shutting down RAG pipeline")


# ── App creation ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline API",
    description="Local RAG system with hybrid search, re-ranking, and faithfulness verification",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down."},
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and model status."""
    try:
        docs = pipeline.get_documents()
    except Exception as exc:
        logger.warning("Document metadata health lookup failed: %s", exc)
        docs = []
    total_chunks = sum(d.chunk_count for d in docs)
    providers = {
        "llm": LLM_PROVIDER,
        "embedding": EMBEDDING_PROVIDER,
        "vector_store": VECTOR_STORE,
        "rerank": RERANK_PROVIDER,
        "verification": VERIFICATION_PROVIDER,
    }

    if LLM_PROVIDER == "HUGGINGFACE_API" or EMBEDDING_PROVIDER == "HUGGINGFACE_API":
        from backend.hf_api import health_check as hf_health_check

        providers["huggingface"] = hf_health_check()

    if VECTOR_STORE == "PINECONE":
        try:
            providers["pinecone"] = {
                "ok": True,
                "total_chunks": pipeline.vector_store.total_chunks,
                "total_documents": pipeline.vector_store.total_documents,
            }
        except Exception as exc:
            providers["pinecone"] = {"ok": False, "detail": str(exc)}

    return HealthResponse(
        status="healthy",
        models_loaded=pipeline.models_loaded,
        document_count=len(docs),
        total_chunks=total_chunks,
        providers=providers,
    )


# ── Document upload ──────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a document (PDF, MD, or TXT) for indexing.
    Computes a file hash to detect duplicates and handle updates.
    """
    # Validate file extension
    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save uploaded file
    file_path = UPLOAD_DIR / filename
    content = await file.read()
    file_path.write_bytes(content)

    logger.info(f"File saved: {filename} ({len(content)} bytes)")

    # Run ingestion in a thread
    try:
        result = await asyncio.to_thread(pipeline.ingest, file_path, filename)
        return result
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── Streaming query ──────────────────────────────────────────────────────────
@app.post("/query/stream")
@limiter.limit(QUERY_RATE_LIMIT)
async def query_stream(request: Request, body: QueryRequest):
    """
    Query the RAG pipeline with real-time SSE token streaming.
    Returns an EventSourceResponse that emits:
    - rewritten_query: the rewritten search query
    - token: individual generated tokens
    - faithfulness: verification result
    - citations: source references
    - done: stream completion signal
    """
    if not pipeline.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please wait.",
        )

    if pipeline.vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload documents first.",
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = body.history if body.history else []

    async def event_generator():
        try:
            async for event in pipeline.query_stream(question, history):
                event_type = event.get("event", "message")
                event_data = event.get("data", {})
                yield {
                    "event": event_type,
                    "data": json.dumps(event_data),
                }
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


# ── Document management ──────────────────────────────────────────────────────
@app.get("/documents")
async def list_documents():
    """List all ingested documents with metadata."""
    docs = pipeline.get_documents()
    return {"documents": [d.model_dump() for d in docs]}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Remove a document from the index and delete its file."""
    success = await asyncio.to_thread(pipeline.delete_document, filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    return {"status": "deleted", "filename": filename}
