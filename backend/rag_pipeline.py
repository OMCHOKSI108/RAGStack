"""
RAG Pipeline orchestrator.
Ties together ingestion (parse -> chunk -> embed -> index)
and intelligent query processing (classify -> retrieve -> reason -> verify → stream).

Production-grade features:
- Intent classification for task routing
- Multi-query retrieval for improved recall
- MMR for result diversity
- Context compression for focused LLM input
- True token-by-token streaming via async generators
- Query result caching for repeated questions
- Agentic reasoning with verification layer
- Structured output validation
"""

import asyncio
import hashlib
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from pathlib import Path

from backend import embeddings, llm
from backend.agent import ReasoningAgent
from backend.cache import get_cache
from backend.chunker import chunk_sections
from backend.config import (
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
    RERANK_PROVIDER,
    SPARSE_RETRIEVAL,
    UPLOAD_DIR,
    VECTOR_STORE,
    VERIFICATION_PROVIDER,
)
from backend.context_compressor import compress_context
from backend.document_parser import parse_document
from backend.faithfulness import verify_faithfulness
from backend.hybrid_search import hybrid_search, multi_query_hybrid_search
from backend.models import (
    DocumentInfo,
    UploadResponse,
)
from backend.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)


def create_vector_store():
    """Create the configured vector store without importing unused backends."""
    if VECTOR_STORE == "PINECONE":
        from backend.vector_store_pinecone import PineconeVectorStore

        return PineconeVectorStore()

    from backend.vector_store import VectorStore

    return VectorStore()


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


class RAGPipeline:
    """
    Main orchestrator for the RAG pipeline.
    Manages document ingestion, indexing, and query processing.
    """

    def __init__(self):
        self.vector_store = create_vector_store()
        if SPARSE_RETRIEVAL:
            from backend.bm25_store import BM25Store

            self.bm25_store = BM25Store()
        else:
            self.bm25_store = None
        self._models_loaded = False
        self.reasoning_agent = ReasoningAgent()

    def load_indices(self) -> None:
        """Load persisted indices or connect to remote stores."""
        self.vector_store.load()
        if self.bm25_store is not None:
            self.bm25_store.load()

    def save_indices(self) -> None:
        """Persist local indices. Remote stores are no-ops."""
        self.vector_store.save()
        if self.bm25_store is not None:
            self.bm25_store.save()

    def load_models(self) -> None:
        """Pre-load local ML models, or mark remote providers ready."""
        remote_mode = all(
            provider == "HUGGINGFACE_API"
            for provider in (LLM_PROVIDER, EMBEDDING_PROVIDER, RERANK_PROVIDER, VERIFICATION_PROVIDER)
        )
        if remote_mode:
            logger.info("Remote Hugging Face providers configured; skipping local model preload")
            self._models_loaded = True
            return

        logger.info("Pre-loading ML models...")
        embeddings.get_model()
        logger.info("Embedding model ready")

        llm.get_llm()
        logger.info("LLM ready")

        self._models_loaded = True
        logger.info("All models loaded")

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    def ingest(self, file_path: Path, filename: str) -> UploadResponse:
        """
        Ingest a document: parse, chunk, embed, and index.
        Handles deduplication via file hashing and document versioning.
        """
        # Step 1: Compute file hash
        file_hash = _compute_file_hash(file_path)
        logger.info(f"Ingesting '{filename}' (hash: {file_hash[:12]}...)")

        # Step 2: Check for duplicates
        if self.vector_store.has_document(file_hash):
            logger.info(f"File '{filename}' is unchanged (hash match), skipping")
            existing_chunks = len(self.vector_store.doc_index.get(file_hash, []))
            return UploadResponse(
                filename=filename,
                chunk_count=existing_chunks,
                status="unchanged",
                file_hash=file_hash,
            )

        # Step 3: If same filename exists with different hash, remove old version
        existing_doc_id = self.vector_store.get_doc_id_by_filename(filename)
        status = "indexed"
        if existing_doc_id and existing_doc_id != file_hash:
            logger.info(f"Updating '{filename}': removing old version")
            self.vector_store.remove_document(existing_doc_id)
            if self.bm25_store is not None:
                self.bm25_store.remove_document(existing_doc_id)
            status = "updated"

        # Step 4: Parse document into sections
        sections = parse_document(file_path)
        if not sections:
            return UploadResponse(
                filename=filename,
                chunk_count=0,
                status="empty",
                file_hash=file_hash,
            )

        # Step 5: Chunk sections semantically
        chunks = chunk_sections(sections, doc_id=file_hash)
        if not chunks:
            return UploadResponse(
                filename=filename,
                chunk_count=0,
                status="empty",
                file_hash=file_hash,
            )

        # Step 6: Embed chunks
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = embeddings.encode(chunk_texts)

        # Step 7: Add to both indices
        self.vector_store.add(chunk_embeddings, chunks, doc_id=file_hash)
        if self.bm25_store is not None:
            self.bm25_store.add(chunks, doc_id=file_hash)

        # Step 8: Persist to disk
        self.save_indices()

        logger.info(f"Ingestion complete: '{filename}' -> {len(chunks)} chunks")
        return UploadResponse(
            filename=filename,
            chunk_count=len(chunks),
            status=status,
            file_hash=file_hash,
        )

    def delete_document(self, filename: str) -> bool:
        """Remove a document from both indices."""
        doc_id = self.vector_store.get_doc_id_by_filename(filename)
        if not doc_id:
            return False

        self.vector_store.remove_document(doc_id)
        if self.bm25_store is not None:
            self.bm25_store.remove_document(doc_id)
        self.save_indices()

        # Remove the file from uploads
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()

        logger.info(f"Deleted document: {filename}")
        return True

    def get_documents(self) -> list[DocumentInfo]:
        """List all ingested documents."""
        doc_info = self.vector_store.get_document_info()
        return [
            DocumentInfo(
                filename=info["filename"],
                file_hash=info["file_hash"],
                chunk_count=info["chunk_count"],
            )
            for info in doc_info.values()
        ]

    async def _stream_tokens_async(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """
        Convert the synchronous LLM generator into a true async generator.
        Uses a thread-safe queue to bridge sync generator -> async yields.
        Tokens are yielded immediately as they arrive, no buffering.
        """
        q: queue.Queue = queue.Queue()

        def _producer():
            try:
                for token in llm.generate_stream(messages):
                    q.put(token)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)  # sentinel

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, q.get)
            if token is None:
                break
            if isinstance(token, Exception):
                logger.error(f"LLM streaming error: {token}")
                raise token
            yield token

    async def query_stream(
        self, question: str, history: list[dict] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Process a query through the INTELLIGENT RAG pipeline:
        1. Check cache for repeated questions
        2. Intent classification for task routing
        3. Query rewriting (typo fix only)
        4. Adaptive retrieval (multi-query if needed)
        5. MMR for diversity
        6. Context compression
        7. Agentic reasoning with verification
        8. Stream response (token-by-token)
        9. Post-generation verification
        10. Emit citations and structured output
        11. Cache response
        """
        # ── Step 0: Check cache ──────────────────────────────────────────
        doc_count = len(self.vector_store.get_document_info())
        cached = get_cache().get(question, doc_count)
        if cached:
            logger.info("Serving cached response")
            if cached.get("rewritten_query"):
                yield {"event": "rewritten_query", "data": {"query": cached["rewritten_query"]}}
            for token in cached.get("tokens", []):
                yield {"event": "token", "data": {"token": token}}
            if cached.get("verification"):
                yield {"event": "verification", "data": cached["verification"]}
            if cached.get("structured"):
                yield {"event": "structured", "data": cached["structured"]}
            if cached.get("citations"):
                yield {"event": "citations", "data": {"citations": cached["citations"]}}
            yield {"event": "done", "data": {}}
            return

        # ── Step 1: Intent Classification ────────────────────────────────
        from backend.intent_classifier import classify_intent
        intent_info = classify_intent(question)
        intent = intent_info["intent"]
        strategy = intent_info["strategy"]
        params = intent_info["params"]

        yield {"event": "intent", "data": {
            "intent": intent,
            "strategy_summary": {
                "multi_query": strategy["use_multi_query"],
                "top_k": strategy["top_k"],
                "structured": strategy["structured_output"],
            }
        }}

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
                }
            }
            yield {"event": "done", "data": {}}
            return

        # ── Step 2: Query Rewriting ──────────────────────────────────────
        rewritten = await asyncio.to_thread(rewrite_query, question)
        if rewritten != question:
            yield {"event": "rewritten_query", "data": {"query": rewritten}}

        # ── Step 3: Adaptive Retrieval ───────────────────────────────────
        base_kwargs = {
            "top_k": strategy["top_k"],
            "rerank_top_k": strategy["rerank_top_k"],
        }
        mmr_kwargs = {
            "use_mmr": True,
            "mmr_lambda": strategy.get("mmr_lambda", 0.5),
        }

        if strategy["use_multi_query"]:
            from backend.multi_query import generate_alternate_queries

            queries = await asyncio.to_thread(generate_alternate_queries, rewritten)
            yield {"event": "rewritten_query", "data": {"query": f"Expanded to {len(queries)} search queries"}}

            search_results = await asyncio.to_thread(
                multi_query_hybrid_search,
                queries,
                self.vector_store,
                self.bm25_store,
                **base_kwargs,
            )
        else:
            search_results = await asyncio.to_thread(
                hybrid_search,
                rewritten,
                self.vector_store,
                self.bm25_store,
                **base_kwargs,
                **mmr_kwargs,
            )

        if not search_results:
            yield {
                "event": "token",
                "data": {"token": "No relevant documents found. Please upload documents first."},
            }
            yield {"event": "done", "data": {}}
            return

        # ── Step 3b: No-answer threshold on retrieval score ─────────────
        from backend.config import RETRIEVAL_NO_ANSWER_THRESHOLD
        top_score = search_results[0].score if search_results else 0.0
        logger.info(
            "Top retrieval score=%.4f, threshold=%.4f, candidates=%d",
            top_score, RETRIEVAL_NO_ANSWER_THRESHOLD, len(search_results),
        )
        if top_score < RETRIEVAL_NO_ANSWER_THRESHOLD:
            yield {
                "event": "token",
                "data": {"token": "No relevant information found in the uploaded documents."},
            }
            yield {
                "event": "verification",
                "data": {
                    "is_grounded": True,
                    "confidence": 1.0,
                    "issues": [f"Top retrieval score {top_score:.3f} below threshold {RETRIEVAL_NO_ANSWER_THRESHOLD}"],
                    "retrieval_score": round(top_score, 4),
                }
            }
            yield {"event": "done", "data": {}}
            return

        # ── Step 4: Context Compression ──────────────────────────────────
        context_chunks = [
            {
                "text": r.chunk.text,
                "source_file": r.chunk.source_file,
                "page_number": str(r.chunk.page_number),
            }
            for r in search_results
        ]

        # For QA and summarization, keep ALL context (no compression)
        compression_ratio = strategy.get("compression_ratio", 0.7)
        if intent in ("qa", "summarization"):
            compression_ratio = 1.0  # Keep everything

        compressed_context = await asyncio.to_thread(
            compress_context,
            context_chunks,
            rewritten,
            compression_ratio,
        )

        # ── Step 5: Agentic Reasoning with Streaming ─────────────────────
        cached_tokens = []
        verification_data = None
        structured_data = None
        full_response = ""

        async for event in self.reasoning_agent.process(
            question, compressed_context, history
        ):
            event_type = event.get("event", "message")
            event_data = event.get("data", {})

            if event_type == "token":
                token = event_data.get("token", "")
                full_response += token
                cached_tokens.append(token)
                yield {"event": "token", "data": {"token": token}}
            elif event_type in ("verification", "structured"):
                if event_type == "verification":
                    verification_data = event_data
                elif event_type == "structured":
                    structured_data = event_data
                yield {"event": event_type, "data": event_data}
            elif event_type == "intent":
                yield {"event": event_type, "data": event_data}

        # ── Step 6: Emit Citations ───────────────────────────────────────
        citations = []
        for i, result in enumerate(search_results):
            snippet = result.chunk.text[:200]
            if len(result.chunk.text) > 200:
                snippet += "..."

            citations.append({
                "index": i + 1,
                "source_file": result.chunk.source_file,
                "page_number": result.chunk.page_number,
                "chunk_index": result.chunk.chunk_index,
                "text_snippet": snippet,
                "relevance_score": round(result.score, 4),
            })

        yield {"event": "citations", "data": {"citations": citations}}

        # ── Step 7: Faithfulness Verification (backup) ───────────────────
        if not verification_data:
            context_texts = [c["text"] for c in compressed_context]
            is_faithful, faith_score = await asyncio.to_thread(
                verify_faithfulness, full_response, context_texts
            )
            verification_data = {
                "is_grounded": is_faithful,
                "confidence": round(faith_score, 3),
                "issues": [] if is_faithful else ["Answer may not be fully supported"],
            }
            yield {"event": "verification", "data": verification_data}

        yield {"event": "done", "data": {}}

        # ── Step 8: Cache Response ───────────────────────────────────────
        get_cache().put(question, doc_count, {
            "rewritten_query": rewritten if rewritten != question else None,
            "tokens": cached_tokens,
            "verification": verification_data,
            "structured": structured_data,
            "citations": citations,
        })
