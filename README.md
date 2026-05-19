# RAG Pipeline

A production-grade Retrieval-Augmented Generation system designed for fully remote, free-tier-friendly deployment. The pipeline parses uploaded documents, indexes them in a managed vector database, and answers natural-language questions with cited, grounded responses streamed token-by-token to the browser. Every heavy compute step — embedding, generation, reranking, and verification — runs on Hugging Face Inference Providers, and every persisted vector lives in Pinecone, so the application itself ships as two lightweight containers that fit comfortably inside Render's free plan.

## Live Deployment

| Service        | URL                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------ |
| Frontend       | [rag-pipeline-frontend-aqk9.onrender.com](https://rag-pipeline-frontend-aqk9.onrender.com) |
| Backend        | [rag-pipeline-backend-odlr.onrender.com](https://rag-pipeline-backend-odlr.onrender.com)   |
| Backend health | [/health](https://rag-pipeline-backend-odlr.onrender.com/health)                           |

Both services run on Render's free tier and sleep after roughly fifteen minutes of inactivity, so the first request after an idle period takes about thirty to sixty seconds while the container warms back up. The frontend includes an **API configuration** panel in the sidebar that lets you point the UI at a different backend (for example a local instance during development) without redeploying.

## What the System Does

At a high level, the pipeline performs four things in order whenever a user asks a question:

1. **Ingest** documents (PDF, Markdown, plain text) by parsing them into sections, chunking each section with overlap, embedding each chunk through Hugging Face, and upserting the resulting vectors into Pinecone with chunk metadata.
2. **Classify intent** of the incoming question (QA, summarization, extraction, comparison, out-of-scope, etc.) and route it to an adaptive retrieval strategy.
3. **Retrieve and rerank** the most relevant chunks using dense search in Pinecone, optional Maximum Marginal Relevance for diversity, and a Hugging Face cross-encoder reranker.
4. **Generate, verify, and stream** the answer through Hugging Face chat completion, then check the answer against the retrieved context with a verifier LLM, and finally stream tokens, citations, and a grounding signal back to the browser over Server-Sent Events.

## Architecture

```
┌──────────────────────┐         ┌──────────────────────────┐
│  Streamlit Frontend  │  HTTP   │     FastAPI Backend      │
│  (Render container)  │ ──────▶ │   (Render container)     │
│   - upload / chat    │         │   /upload, /query/stream │
│   - SSE consumer     │ ◀────── │   /documents, /health    │
└──────────────────────┘   SSE   └────────────┬─────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ▼                                     ▼                                     ▼
┌────────────────────┐               ┌────────────────────┐               ┌────────────────────┐
│  HF Inference      │               │  Pinecone          │               │  HF Inference      │
│  Providers         │               │  Serverless        │               │  Providers         │
│  (chat + verify)   │               │  (dense vectors)   │               │  (embeddings +     │
│  Llama-3.1-8B-Inst │               │  ragapp, 384-dim   │               │   rerank)          │
└────────────────────┘               └────────────────────┘               └────────────────────┘
```

The backend is intentionally stateless: anything persistent lives in Pinecone, the model weights live in Hugging Face, and Render's ephemeral disk is only used as a scratch space for the uploaded file during parsing. Restarting either service does not lose data.

### Request Lifecycle

The diagram below tracks a single query from the moment the user hits send to the moment the final `done` event arrives in the browser:

```
Browser
   │ POST /query/stream  { question, history }
   ▼
FastAPI ── intent_classifier ── route to strategy
   │
   ├── query_rewriter (typo fix)
   │
   ├── HF embeddings.encode_query() ─────────────▶ HF Inference Providers
   │
   ├── Pinecone dense search (top_k = 20)  ─────▶ Pinecone index
   │
   ├── MMR diversification (optional)
   │
   ├── HF cross-encoder rerank (top 5)  ─────────▶ HF Inference Providers
   │
   ├── context_compressor (keeps focused passages)
   │
   ├── ReasoningAgent
   │      │
   │      ├── build chat messages with cited context
   │      └── HF chat_stream() ────── tokens ────▶ Browser  (SSE: token)
   │
   ├── citation extraction      ──────────────────▶ Browser  (SSE: citations)
   │
   ├── HF verifier prompt       ──────────────────▶ Browser  (SSE: verification)
   │
   └── cache result             ──────────────────▶ Browser  (SSE: done)
```

Every step that touches a remote provider is wrapped so that a transient failure surfaces as a clean SSE `error` event instead of a stack trace, and the verification step degrades gracefully to a textual YES/NO grounding check when the more sophisticated NLI path is not configured.

## Components

The backend is split into small, single-responsibility modules. The table below describes the role each module plays in the system as it is deployed today.

| Module                                | Role                                                                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `backend/main.py`                     | FastAPI application, CORS, rate limiting, lifespan hooks, and the four public routes                       |
| `backend/rag_pipeline.py`             | Orchestrator that wires ingestion and query streaming together; holds the vector store and reasoning agent |
| `backend/config.py`                   | Single source of truth for every env var, model id, threshold, and provider switch                         |
| `backend/hf_api.py`                   | Hugging Face Inference Providers client: chat (stream + complete), embeddings, rerank, verifier            |
| `backend/vector_store_pinecone.py`    | Pinecone-backed implementation of the vector-store contract                                                |
| `backend/vector_store.py`             | FAISS-backed implementation (used only when `VECTOR_STORE=FAISS` for local dev)                            |
| `backend/bm25_store.py`               | BM25 sparse retrieval (gated behind `SPARSE_RETRIEVAL=true`; disabled by default in cloud mode)            |
| `backend/embeddings.py`               | Embedding adapter that routes to HF when configured, otherwise to a local sentence-transformers model      |
| `backend/llm.py`                      | LLM adapter that routes to HF, Ollama, or local Transformers depending on `LLM_PROVIDER`                   |
| `backend/hybrid_search.py`            | Dense + optional sparse retrieval, RRF fusion, MMR, and reranking                                          |
| `backend/mmr.py`                      | Maximum Marginal Relevance for diversity-aware reranking                                                   |
| `backend/multi_query.py`              | Generates alternate phrasings of a query when the intent strategy asks for it                              |
| `backend/query_rewriter.py`           | Lightweight typo and spelling fix before retrieval                                                         |
| `backend/intent_classifier.py`        | Classifies the user's question and chooses retrieval, compression, and verification parameters             |
| `backend/agent.py`                    | The ReasoningAgent that builds prompts, streams tokens, and emits verification events                      |
| `backend/task_prompts.py`             | Task-specific prompt templates (QA, summarization, extraction, comparison, etc.)                           |
| `backend/context_compressor.py`       | Trims and reorders retrieved chunks so the LLM sees the most relevant slice                                |
| `backend/verification.py`             | Sentence-level keyword + embedding grounding check, plus a HF-LLM verifier branch                          |
| `backend/faithfulness.py`             | NLI-based faithfulness verifier (local Transformers); also routes to HF verifier when remote               |
| `backend/validators.py`               | Structured-output validation for extraction tasks                                                          |
| `backend/document_parser.py`          | PDF (PyMuPDF), Markdown, and TXT parsing into sections with page metadata                                  |
| `backend/chunker.py`                  | Section-aware chunker that respects sentence boundaries and overlap                                        |
| `backend/cache.py`                    | On-disk query cache for repeated questions                                                                 |
| `backend/models.py`                   | Pydantic schemas shared between the API, pipeline, and stores                                              |
| `frontend/app.py`                     | Streamlit chat UI: upload panel, document list, streaming chat, citations, runtime API config              |
| `test.py`                             | Evaluation harness: synthetic queries across nine categories, twenty-plus metrics                          |
| `run.py`                              | Local launcher that spins up backend and frontend on the same machine                                      |

## Provider Selection

Every dependency is selected through a single environment variable, which makes it possible to run the same code locally against Ollama and FAISS or in the cloud against Hugging Face and Pinecone without code changes.

| Variable                  | Cloud value         | Local default | Effect                                                            |
| ------------------------- | ------------------- | ------------- | ----------------------------------------------------------------- |
| `LLM_PROVIDER`            | `HUGGINGFACE_API`   | `OLLAMA`      | Chooses generation backend                                        |
| `EMBEDDING_PROVIDER`      | `HUGGINGFACE_API`   | `LOCAL`       | Chooses embedding backend                                         |
| `RERANK_PROVIDER`         | `HUGGINGFACE_API`   | `LOCAL`       | Chooses reranker backend                                          |
| `VERIFICATION_PROVIDER`   | `HUGGINGFACE_API`   | `LOCAL`       | Chooses faithfulness/verifier backend                             |
| `VECTOR_STORE`            | `PINECONE`          | `FAISS`       | Chooses dense-vector store                                        |
| `SPARSE_RETRIEVAL`        | `false`             | derived       | Toggles BM25; auto-off when `VECTOR_STORE=PINECONE`               |

The full environment matrix — Hugging Face model ids, timeouts, Pinecone region, chunk size, retrieval `top_k`, RRF and MMR knobs, faithfulness thresholds, rate limits — lives in [`backend/config.py`](backend/config.py) and is documented inline. Anything declared there can be overridden through environment variables without editing the file.

## API Surface

The backend exposes four routes. They are intentionally narrow so the frontend (or any other consumer) only has to understand a small contract.

- `GET /health` — returns service status, model-loading flag, document and chunk counts, and a `providers` block reporting Hugging Face token configuration and Pinecone index stats. Used as Render's health-check.
- `POST /upload` — accepts a multipart file (PDF, MD, TXT), hashes it, deduplicates against existing vectors, parses, chunks, embeds, and upserts to Pinecone. Returns a status of `indexed`, `updated`, `unchanged`, or `empty`.
- `POST /query/stream` — accepts a JSON body `{ "question": "...", "history": [...] }` and returns an SSE stream. Event names are stable: `intent`, `rewritten_query`, `token`, `citations`, `verification`, `structured`, `error`, `done`.
- `GET /documents` and `DELETE /documents/{filename}` — list and remove indexed documents.

Both `/upload` and `/query/stream` are rate-limited per client IP through `slowapi`.

## Streaming Events

The frontend consumes the SSE stream emitted by `/query/stream` and uses each event type to update a different region of the chat surface. The contract is:

- `intent` — fires once at the start with the chosen intent label and a strategy summary.
- `rewritten_query` — fires if the typo fixer or multi-query expander changed the original question.
- `token` — fires repeatedly with one or a few characters of the model's answer at a time. The frontend appends these directly to the visible bubble.
- `citations` — fires once after the answer is complete, with one citation object per retrieved chunk (`index`, `source_file`, `page_number`, `chunk_index`, `text_snippet`, `relevance_score`).
- `verification` — fires after the answer, reporting `is_grounded`, `confidence`, and any `issues` raised by the verifier.
- `structured` — fires only when the intent expected a structured payload (extraction tasks).
- `error` — fires once if any upstream call fails. The frontend renders it as a banner.
- `done` — fires last; the frontend uses this to stop the streaming indicator and unlock the input.

## Quick Start (Local)

Local development still works against either Ollama or Hugging Face. The instructions below use the cloud-style configuration so the local run matches production.

### Prerequisites

- Python 3.11 or newer
- A Hugging Face token with Inference Providers access
- A Pinecone account, an API key, and one serverless index of dimension 384 with cosine metric (the project uses `sentence-transformers/all-MiniLM-L6-v2` by default)

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file at the project root with at minimum:

```ini
LLM_PROVIDER=HUGGINGFACE_API
EMBEDDING_PROVIDER=HUGGINGFACE_API
VECTOR_STORE=PINECONE
RERANK_PROVIDER=HUGGINGFACE_API
VERIFICATION_PROVIDER=HUGGINGFACE_API
SPARSE_RETRIEVAL=false

HF_TOKEN=hf_xxx
HF_CHAT_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
HF_EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
HF_RERANKER_MODEL_ID=BAAI/bge-reranker-base
HF_VERIFIER_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

PINECONE_API_KEY=pcsk_xxx
PINECONE_INDEX=ragapp
PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

BACKEND_URL=http://localhost:8000
```

### Run

```bash
# Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend (in a second terminal)
streamlit run frontend/app.py --server.port 8501

# Or both at once
python run.py
```

The Streamlit app is then available at `http://localhost:8501` and will talk to the backend at `http://localhost:8000`.

### Evaluate

```bash
python test.py --mode debug       # quick smoke run, ~10 queries
python test.py --queries 100      # full evaluation suite
python test.py --queries 50       # custom size
```

Outputs land in [`evaluation_output/`](evaluation_output/) including JSON metrics, plots, and a Markdown summary.

## Deploying to Render

The repository ships with everything Render needs to stand up both services from a single commit.

1. **Push the repo to GitHub** (or any git host Render supports). Ensure `.env` is in `.gitignore` so secrets do not leak.
2. **Create a new Blueprint** in Render and point it at the repo. Render reads [`render.yaml`](render.yaml) and creates two web services: `rag-pipeline-backend` and `rag-pipeline-frontend`.
3. **Set secrets in the backend service** (`sync: false` entries in `render.yaml`): `HF_TOKEN`, `HF_CHAT_MODEL_ID`, `HF_VERIFIER_MODEL_ID`, `PINECONE_API_KEY`, `PINECONE_INDEX`.
4. **Set `BACKEND_URL` on the frontend service** to the full HTTPS URL of the backend (for example `https://rag-pipeline-backend-odlr.onrender.com`). Render's automatic `fromService` reference returns only the bare hostname, so this must be a manual entry.
5. **Wait for the health-check at `/health` to go green** on the backend, then open the frontend URL.

The two Dockerfiles ([`Dockerfile.backend`](Dockerfile.backend) and [`Dockerfile.frontend`](Dockerfile.frontend)) are minimal `python:3.11-slim` images that install `requirements.txt` and start the respective process. No GPU, no model weights, no extra build steps.

## Operational Notes

- **Cold starts**: Render free-tier services sleep after fifteen minutes of inactivity. The first request after a sleep can take up to a minute to return. The frontend's health-check timeout is generous (forty-five seconds) so a normal cold start does not produce a false "offline" banner.
- **Idempotent uploads**: Uploading the same file twice returns `unchanged`. Uploading a modified version of an existing filename returns `updated` and replaces all of the previous document's vectors in Pinecone.
- **Embedding dimension is locked**: The Pinecone index dimension must match the embedding model. The default pairing is `sentence-transformers/all-MiniLM-L6-v2` (384 dim) with a 384-dim cosine index named `ragapp`. Changing the embedding model means recreating the index.
- **Pinecone metadata limit**: Each vector stores its chunk text in metadata. Pinecone's 40 KB per-vector limit is well above the configured `CHUNK_SIZE` of 1024 characters, but if you raise the chunk size dramatically you may need to switch to storing chunks in an external blob store.
- **Rate limits**: Defaults are ten uploads per minute and thirty queries per minute per client IP. Override with `UPLOAD_RATE_LIMIT` and `QUERY_RATE_LIMIT`.

## Evaluation Snapshot

The `test.py` harness produces synthetic queries across nine categories (extraction, summarization, comparison, QA, unsupported, adversarial, citation, constraints, multi-turn) and records over twenty metrics covering retrieval, generation, citations, hallucination, and performance.

| Section              | Metric              | Value                       |
| -------------------- | ------------------- | --------------------------- |
| Benchmark            | Total queries       | 10                          |
| Benchmark            | Success rate        | 100.0%                      |
| Benchmark            | Average latency     | 15369 ms                    |
| Benchmark            | P95 latency         | 23863 ms                    |
| Benchmark            | Throughput          | 0.07 queries / second       |
| Benchmark            | Hallucination rate  | 50.0%                       |
| Retrieval            | Precision@K         | 0.0000                      |
| Retrieval            | Recall@K            | 0.0000                      |
| Retrieval            | MRR                 | 0.0000                      |
| Retrieval            | nDCG                | 0.0000                      |
| Generation           | Answer relevancy    | 0.2955                      |
| Generation           | Faithfulness        | 0.3412                      |
| Generation           | Citation accuracy   | 0.8000                      |
| Generation           | Semantic similarity | 0.0000                      |
| System               | Platform            | Windows-11-10.0.26200-SP0   |
| System               | Python              | 3.13.9                      |
| System               | CPU cores           | 12                          |
| System               | RAM                 | 15.7 GB                     |
| System               | GPU                 | Not available               |

Last updated: 2026-05-19.

### Plots

![Evalutions](evaluation_output/plots/evaluation_dashboard.png)

## Repository Layout

```
.
├── backend/                       # FastAPI service and all RAG modules
│   ├── main.py                    # API routes, lifespan, CORS, rate limiting
│   ├── rag_pipeline.py            # Orchestrator
│   ├── config.py                  # All configuration and provider switches
│   ├── hf_api.py                  # Hugging Face Inference Providers client
│   ├── vector_store_pinecone.py   # Pinecone vector store
│   ├── vector_store.py            # FAISS vector store (local dev only)
│   ├── bm25_store.py              # BM25 sparse retriever (opt-in)
│   ├── embeddings.py              # Embedding router
│   ├── llm.py                     # LLM router
│   ├── hybrid_search.py           # Dense + sparse + RRF + MMR + rerank
│   ├── mmr.py                     # MMR implementation
│   ├── multi_query.py             # Query expansion
│   ├── query_rewriter.py          # Typo fix
│   ├── intent_classifier.py       # Intent + strategy
│   ├── agent.py                   # ReasoningAgent
│   ├── task_prompts.py            # Per-task prompt templates
│   ├── context_compressor.py      # Context trimming
│   ├── verification.py            # Grounding checks
│   ├── faithfulness.py            # NLI faithfulness
│   ├── validators.py              # Structured-output validation
│   ├── document_parser.py         # PDF / MD / TXT parsing
│   ├── chunker.py                 # Section-aware chunking
│   ├── cache.py                   # Query cache
│   └── models.py                  # Pydantic schemas
├── frontend/
│   └── app.py                     # Streamlit chat UI
├── data/uploads/                  # Local upload sink (ignored in cloud)
├── evaluation_output/             # Plots, metrics, and reports
├── Dockerfile.backend             # Backend container
├── Dockerfile.frontend            # Frontend container
├── render.yaml                    # Render blueprint (two services)
├── requirements.txt               # Cloud-friendly minimal deps
├── run.py                         # Local launcher
├── test.py                        # Evaluation harness
└── README.md
```

## License

See [`LICENSE`](LICENSE) if present, otherwise this code is provided as-is for educational and internal use.
