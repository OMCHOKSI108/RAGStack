"""
Pydantic schemas for request/response models and internal data structures.
"""

from typing import Optional

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A single chunk of text from a parsed document."""
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    doc_id: str  # SHA-256 hash of the source file


class DocumentSection(BaseModel):
    """A section extracted from a document before chunking."""
    text: str
    source_file: str
    page_or_section: int
    raw_text: Optional[str] = None  # Original text before OCR normalization


class UploadResponse(BaseModel):
    """Response after uploading and indexing a document."""
    filename: str
    chunk_count: int
    status: str  # "indexed", "unchanged", "updated"
    file_hash: str


class QueryRequest(BaseModel):
    """Request body for querying the RAG pipeline."""
    question: str
    history: Optional[list[dict[str, str]]] = None  # Conversation history for context


class Citation(BaseModel):
    """A citation referencing a specific chunk in a source document."""
    source_file: str
    page_number: int
    chunk_index: int
    text_snippet: str
    relevance_score: float


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""
    filename: str
    file_hash: str
    chunk_count: int


class SearchResult(BaseModel):
    """A single search result from the vector or BM25 store."""
    chunk: DocumentChunk
    score: float
    global_index: int  # Position in the FAISS/BM25 index


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    document_count: int
    total_chunks: int
    providers: Optional[dict] = None
