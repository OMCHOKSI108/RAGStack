"""
Launch script for the RAG Pipeline.
Starts both the FastAPI backend and Streamlit frontend as subprocesses.
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).parent

# Add to path
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("  RAG Pipeline Launcher")
    print("=" * 60)
    print()

    print("Models will be auto-downloaded from HuggingFace on first run.")
    print("To pre-download, run: python download_model.py")
    print()

    # Determine the Python executable - prefer venv if it exists
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python = str(venv_python)
        print(f"Using venv Python: {python}")
    else:
        python = sys.executable
        print(f"Using system Python: {python}")

    # Start FastAPI backend
    print("[1/2] Starting FastAPI backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [
            python, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
        ],
        cwd=str(ROOT),
    )

    # Wait a moment for backend to start
    time.sleep(3)

    # Start Streamlit frontend
    print("[2/2] Starting Streamlit frontend on http://localhost:8501 ...")
    frontend_proc = subprocess.Popen(
        [
            python, "-m", "streamlit", "run",
            "frontend/app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(ROOT),
    )

    print()
    print("=" * 60)
    print("  Backend:  http://localhost:8000")
    print("  Frontend: http://localhost:8501")
    print("  API Docs: http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop both servers.")
    print("=" * 60)

    # Wait for Ctrl+C
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")

        backend_proc.terminate()
        frontend_proc.terminate()

        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()

        try:
            frontend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_proc.kill()

        print("Both servers stopped.")


if __name__ == "__main__":
    main()
