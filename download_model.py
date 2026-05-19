"""
Pre-download all HuggingFace models used by the RAG pipeline.
This ensures the first startup is faster since models are cached locally.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def download_models():
    """Pre-download all models from HuggingFace Hub."""

    from huggingface_hub import snapshot_download

    print("=" * 60)
    print("  RAG Pipeline -- Model Download")
    print("=" * 60)
    print()

    # ──────────────────────────────────────────────────────────────────────
    # 1. Embedding model
    # ──────────────────────────────────────────────────────────────────────
    print("[1/4] Downloading embedding model...")

    try:
        from sentence_transformers import SentenceTransformer
        from backend.config import EMBEDDING_MODEL

        t0 = time.time()

        model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  OK: {EMBEDDING_MODEL}")
        print(f"  Time: {time.time() - t0:.1f}s")

        del model

    except Exception as e:
        print(f"  FAILED: {e}")

    print()

    # ──────────────────────────────────────────────────────────────────────
    # 2. TinyLlama LLM
    # ──────────────────────────────────────────────────────────────────────
    print("[2/4] Downloading LLM (TinyLlama-1.1B-Chat)...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from backend.config import LLM_MODEL_ID

        t0 = time.time()

        print(f"  Repository: {LLM_MODEL_ID}")
        print("  Downloading with live progress...\n")

        # LIVE progress bar download
        local_dir = snapshot_download(
            repo_id=LLM_MODEL_ID,
            local_dir="./models/tinyllama",
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print("\n  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        print("  Loading model into memory...")
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="auto"
        )

        print(f"  OK: {LLM_MODEL_ID}")
        print(f"  Saved to: {local_dir}")
        print(f"  Time: {time.time() - t0:.1f}s")

        del model, tokenizer

    except Exception as e:
        print(f"  FAILED: {e}")

    print()

    # ──────────────────────────────────────────────────────────────────────
    # 3. Re-ranker model
    # ──────────────────────────────────────────────────────────────────────
    print("[3/4] Downloading re-ranker model...")

    try:
        from sentence_transformers import CrossEncoder
        from backend.config import RERANKER_MODEL

        t0 = time.time()

        model = CrossEncoder(RERANKER_MODEL)

        print(f"  OK: {RERANKER_MODEL}")
        print(f"  Time: {time.time() - t0:.1f}s")

        del model

    except Exception as e:
        print(f"  FAILED: {e}")

    print()

    # ──────────────────────────────────────────────────────────────────────
    # 4. Faithfulness verifier
    # ──────────────────────────────────────────────────────────────────────
    print("[4/4] Downloading faithfulness verifier...")

    try:
        from transformers import pipeline
        from backend.config import VERIFIER_MODEL

        t0 = time.time()

        pipe = pipeline(
            "question-answering",
            model=VERIFIER_MODEL
        )

        print(f"  OK: {VERIFIER_MODEL}")
        print(f"  Time: {time.time() - t0:.1f}s")

        del pipe

    except Exception as e:
        print(f"  FAILED: {e}")

    print()
    print("=" * 60)
    print("  All models downloaded successfully.")
    print("  Run: python run.py")
    print("=" * 60)


if __name__ == "__main__":
    download_models()