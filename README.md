# RAG Pipeline

Production-grade Retrieval-Augmented Generation pipeline with intelligent document reasoning, real-time SSE streaming, GPU auto-detection, strict hallucination prevention, and comprehensive evaluation/benchmarking.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit UI   │────▶│  FastAPI Backend │────▶│  LLM (Ollama)   │
│  (Frontend)     │◀────│  (SSE Streaming) │◀────│  / HF TinyLlama │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ FAISS    │ │  BM25    │ │Cross-Enc │
              │ Embedding│ │ Sparse   │ │ Reranker │
              └──────────┘ └──────────┘ └──────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `backend/main.py` | FastAPI app with `/query/stream` SSE endpoint |
| `backend/rag_pipeline.py` | Core orchestrator: async streaming, caching, compression |
| `backend/agent.py` | Agentic reasoning workflow (understand → retrieve → verify → stream) |
| `backend/intent_classifier.py` | Intent detection & adaptive strategy routing |
| `backend/hybrid_search.py` | BM25 + FAISS + RRF + MMR + cross-encoder reranking |
| `backend/verification.py` | Hallucination detection & source grounding checks |
| `backend/faithfulness.py` | Sentence similarity verification |
| `backend/llm.py` | Ollama/HF wrapper with GPU auto-detection |
| `backend/document_parser.py` | PDF/MD/text parser using PyMuPDF |
| `backend/config.py` | Centralized hyperparameters and model paths |
| `frontend/app.py` | Streamlit UI with custom CSS, live streaming indicators |
| `test.py` | Evaluation system: 9 query categories, 22+ metrics, visualization |

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama running with `llama3.2:3b` (or HF fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
# Start backend + frontend
python run.py

# Backend:  http://localhost:8000
# Frontend: http://localhost:8501
```

### Evaluate
```bash
# Quick debug run (10 queries)
python test.py --mode debug

# Full evaluation (100 queries)
python test.py --queries 100

# Custom query count
python test.py --queries 50
```

## Evaluation Results

**Last Updated:** 2026-05-19 17:01:29

### Benchmark Summary

| Metric | Value |
|--------|-------|
| Total Queries | 10 |
| Success Rate | 100.0% |
| Avg Latency | 15369ms |
| P95 Latency | 23863ms |
| Throughput | 0.07 queries/sec |
| Hallucination Rate | 50.00% |

### Retrieval Performance

| Metric | Score |
|--------|-------|
| Precision@K | 0.0000 |
| Recall@K | 0.0000 |
| MRR | 0.0000 |
| nDCG | 0.0000 |

### Generation Quality

| Metric | Score |
|--------|-------|
| Answer Relevancy | 0.2955 |
| Faithfulness | 0.3412 |
| Citation Accuracy | 0.8000 |
| Semantic Similarity | 0.0000 |

### System Information

| Property | Value |
|----------|-------|
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.13.9 |
| CPU Cores | 12 |
| RAM | 15.7 GB |
| GPU | Not available |

### Evaluation Plots

![Dashboard](evaluation_output/plots/evaluation_dashboard.png)
![Latency](evaluation_output/plots/latency_histogram.png)
![Retrieval](evaluation_output/plots/retrieval_scores.png)
![Faithfulness](evaluation_output/plots/faithfulness_chart.png)

### Methodology

- **Synthetic Queries**: 10 queries across 9 categories
- **Categories**: Extraction, Summarization, Comparison, QA, Unsupported, Adversarial, Citation, Constraints, Multi-turn
- **Metrics**: 22+ metrics covering retrieval, generation, citations, hallucination, and performance
- **Duration**: ~2 minutes estimated

---
