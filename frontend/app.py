"""
Streamlit frontend for the RAG Pipeline.
Provides a chat interface with SSE streaming, file upload, and citation display.
"""

from __future__ import annotations

import html
import json
import os
import time
from typing import Any

import httpx
import streamlit as st


def _normalize_backend_url(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return "http://localhost:8000"
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    return raw


_DEFAULT_BACKEND_URL = _normalize_backend_url(
    os.getenv("BACKEND_URL", "http://localhost:8000")
)


def get_backend_url() -> str:
    """Return the active backend URL, preferring an in-page override."""
    return st.session_state.get("api_base", _DEFAULT_BACKEND_URL)


def health_url() -> str:
    return f"{get_backend_url()}/health"


def upload_url() -> str:
    return f"{get_backend_url()}/upload"


def query_url() -> str:
    return f"{get_backend_url()}/query/stream"


def documents_url() -> str:
    return f"{get_backend_url()}/documents"


st.set_page_config(
    page_title="RAG Pipeline - Document Q&A",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-page: #08090d;
        --bg-panel: #111318;
        --bg-panel-soft: rgba(255, 255, 255, 0.035);
        --bg-panel-hover: rgba(255, 255, 255, 0.06);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.16);
        --text-primary: #f3f4f6;
        --text-secondary: #a7adb8;
        --text-muted: #747b89;
        --accent: #38bdf8;
        --accent-alt: #a3e635;
        --accent-warning: #f59e0b;
        --accent-danger: #f87171;
        --shadow-panel: 0 16px 40px rgba(0, 0, 0, 0.28);
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: 0 !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 32rem),
            linear-gradient(135deg, var(--bg-page), #101116 62%, #090a0f);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: var(--text-primary) !important;
        font-weight: 650 !important;
    }

    p, li, span, label {
        color: var(--text-secondary);
    }

    hr {
        border-color: var(--border-subtle) !important;
        margin: 1.1rem 0 !important;
    }

    code {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        color: var(--text-primary);
        padding: 0.1rem 0.35rem;
    }

    .main-header {
        background:
            linear-gradient(135deg, rgba(56, 189, 248, 0.13), rgba(163, 230, 53, 0.055)),
            var(--bg-panel-soft);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        box-shadow: var(--shadow-panel);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    .main-header h1 {
        font-size: clamp(1.55rem, 2.6vw, 2.35rem) !important;
        line-height: 1.1 !important;
        margin: 0 !important;
    }

    .main-header p {
        color: var(--text-secondary) !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
        max-width: 760px;
        margin: 0.55rem 0 0 0 !important;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0 0 1.15rem 0;
    }

    .metric-card {
        background: var(--bg-panel-soft);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.8rem 0.95rem;
        min-height: 4.25rem;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 650;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 650;
        margin-top: 0.22rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101218, #0a0b10);
        border-right: 1px solid var(--border-subtle);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }

    .sidebar-brand {
        color: var(--text-primary);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sidebar-copy {
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.55;
        margin-bottom: 0.9rem;
    }

    .section-heading {
        background: var(--bg-panel-soft);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        color: var(--text-primary);
        font-weight: 650;
        padding: 0.7rem 0.8rem;
        margin: 1rem 0 0.65rem 0;
    }

    .section-heading h4 {
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
    }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.58rem 0.7rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        margin-bottom: 0.55rem;
    }

    .status-dot {
        display: inline-block;
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 999px;
        flex: 0 0 auto;
    }

    .status-dot.online {
        background: var(--accent-alt);
        box-shadow: 0 0 14px rgba(163, 230, 53, 0.48);
        animation: pulse 1.8s ease-in-out infinite;
    }

    .status-dot.offline {
        background: var(--accent-danger);
        box-shadow: 0 0 14px rgba(248, 113, 113, 0.42);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.62; transform: scale(0.86); }
    }

    .stChatMessage {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        animation: fadeSlideIn 0.22s ease-out;
    }

    .stChatMessage:hover {
        border-color: var(--border-strong);
        background: rgba(255, 255, 255, 0.048);
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatInput > div {
        background: rgba(17, 19, 24, 0.94) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 8px !important;
        box-shadow: var(--shadow-panel);
        padding: 0.35rem !important;
    }

    .stChatInput > div:focus-within {
        border-color: rgba(56, 189, 248, 0.72) !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.12) !important;
    }

    .stChatInput textarea {
        color: var(--text-primary) !important;
        font-size: 0.98rem !important;
    }

    .stButton > button {
        background: rgba(255, 255, 255, 0.045);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        min-height: 2.35rem;
        padding: 0.45rem 0.8rem;
        font-size: 0.86rem;
        font-weight: 600;
        transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
    }

    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.42);
        transform: translateY(-1px);
    }

    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.035);
        border: 1px dashed var(--border-strong);
        border-radius: 8px;
        padding: 1rem;
    }

    .stFileUploader > div:hover {
        border-color: rgba(56, 189, 248, 0.62);
        background: rgba(56, 189, 248, 0.06);
    }

    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 0.65rem 0.85rem;
    }

    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-subtle);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 0.85rem;
    }

    .doc-item,
    .citation-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.55rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    .citation-card {
        border-left: 3px solid var(--accent);
    }

    .doc-item:hover,
    .citation-card:hover {
        background: var(--bg-panel-hover);
        border-color: var(--border-strong);
    }

    .doc-item strong,
    .citation-card strong {
        color: var(--text-primary);
        font-weight: 650;
    }

    .citation-index {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.25rem;
        height: 1.25rem;
        background: rgba(56, 189, 248, 0.18);
        border: 1px solid rgba(56, 189, 248, 0.36);
        border-radius: 999px;
        color: #dff7ff;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.45rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.74rem;
        font-weight: 650;
        margin: 0.25rem 0 0.55rem 0;
    }

    .badge-info {
        background: rgba(56, 189, 248, 0.13);
        border: 1px solid rgba(56, 189, 248, 0.32);
        color: #b7ecff;
    }

    .faith-ok,
    .faith-warning,
    .rewritten-query,
    .error-panel {
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-top: 0.7rem;
    }

    .faith-ok {
        background: rgba(163, 230, 53, 0.1);
        border: 1px solid rgba(163, 230, 53, 0.28);
        border-left: 3px solid var(--accent-alt);
        color: #d8f99c;
    }

    .faith-warning {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-left: 3px solid var(--accent-warning);
        color: #ffd89a;
    }

    .rewritten-query {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-left: 3px solid var(--accent);
        color: #b7ecff;
        margin-bottom: 0.75rem;
    }

    .error-panel {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.3);
        color: #fecaca;
    }

    .empty-state {
        color: var(--text-muted);
        font-size: 0.84rem;
        text-align: center;
        padding: 1rem 0;
    }

    @media (max-width: 760px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }

        .main-header {
            padding: 1.1rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def esc(value: Any) -> str:
    """Escape text before placing it in an unsafe HTML block."""
    return html.escape(str(value), quote=True)


def check_backend_health() -> dict | None:
    """Check if the backend is running and return health info."""
    try:
        response = httpx.get(health_url(), timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None


def fetch_documents() -> list:
    """Fetch the list of indexed documents from the backend."""
    try:
        response = httpx.get(documents_url(), timeout=10.0)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return []


def upload_file(uploaded_file) -> dict | None:
    """Upload a file to the backend for indexing."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = httpx.post(upload_url(), files=files, timeout=120.0)
        if response.status_code == 200:
            return response.json()
        st.error(f"Upload failed: {response.text}")
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        st.error(f"Connection error: {e}")
    return None


def delete_doc(filename: str) -> bool:
    """Delete a document from the backend index."""
    try:
        response = httpx.delete(f"{documents_url()}/{filename}", timeout=30.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def stream_query_live(question: str, placeholder, conversation_history: list | None = None):
    """Stream a query response and update the UI as tokens arrive."""
    events = {
        "rewritten_query": None,
        "full_answer": "",
        "faithfulness": None,
        "verification": None,
        "structured": None,
        "intent": None,
        "citations": [],
        "error": None,
        "unfaithful_replacement": None,
        "search_results_count": 0,
    }

    try:
        payload = {"question": question}
        if conversation_history:
            payload["history"] = conversation_history

        with httpx.stream(
            "POST",
            query_url(),
            json=payload,
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                response.read()
                try:
                    detail = response.json().get("detail")
                    events["error"] = detail if detail else f"Server error: {response.status_code}"
                except Exception:
                    events["error"] = f"Server error: {response.status_code}"
                return events

            current_event = "message"
            last_update_time = 0.0

            for line in response.iter_lines():
                line = line.strip()

                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if current_event == "token":
                    token = data.get("token", "")
                    events["full_answer"] += token

                    now = time.time()
                    if now - last_update_time > 0.1:
                        placeholder.markdown(events["full_answer"] + " |")
                        last_update_time = now
                elif current_event == "rewritten_query":
                    events["rewritten_query"] = data.get("query", "")
                elif current_event == "faithfulness":
                    events["faithfulness"] = data
                elif current_event == "verification":
                    events["verification"] = data
                elif current_event == "structured":
                    events["structured"] = data
                elif current_event == "intent":
                    events["intent"] = data
                elif current_event == "citations":
                    events["citations"] = data.get("citations", [])
                    events["search_results_count"] = len(events["citations"])
                elif current_event == "unfaithful_replacement":
                    events["unfaithful_replacement"] = data.get("message", "")
                elif current_event == "error":
                    events["error"] = data.get("error", "Unknown error")
                elif current_event == "done":
                    break

            placeholder.markdown(events["full_answer"])

    except httpx.ConnectError:
        events["error"] = "Cannot connect to backend. Is the server running?"
    except httpx.TimeoutException:
        events["error"] = "Request timed out. The server may be overloaded."

    return events


def render_status(health: dict | None) -> None:
    if health:
        doc_count = health.get("document_count", 0)
        chunk_count = health.get("total_chunks", 0)
        st.markdown(
            '<div class="status-indicator">'
            '<span class="status-dot online"></span>'
            '<span style="color: var(--text-primary); font-size: 0.875rem;">Backend online</span>'
            '</div>'
            f'<div style="color: var(--text-muted); font-size: 0.8rem;">'
            f'{doc_count} documents / {chunk_count} chunks'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="status-indicator">'
        '<span class="status-dot offline"></span>'
        '<span style="color: var(--text-secondary); font-size: 0.875rem;">Backend offline</span>'
        '</div>'
        '<div style="color: var(--text-muted); font-size: 0.8rem;">'
        'Start with <code>python run.py</code>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_metrics(health: dict | None) -> None:
    doc_count = health.get("document_count", 0) if health else 0
    chunk_count = health.get("total_chunks", 0) if health else 0
    status = "Online" if health else "Offline"

    st.markdown(
        '<div class="metric-grid">'
        '<div class="metric-card"><div class="metric-label">Backend</div>'
        f'<div class="metric-value">{status}</div></div>'
        '<div class="metric-card"><div class="metric-label">Documents</div>'
        f'<div class="metric-value">{doc_count}</div></div>'
        '<div class="metric-card"><div class="metric-label">Indexed chunks</div>'
        f'<div class="metric-value">{chunk_count}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_intent(intent: dict | None) -> None:
    if not intent:
        return
    intent_type = esc(intent.get("intent", ""))
    st.markdown(
        f'<div class="badge badge-info">Intent: {intent_type}</div>',
        unsafe_allow_html=True,
    )


def render_citations(citations: list) -> None:
    if not citations:
        return

    with st.expander(f"Sources ({len(citations)} references)"):
        for cit in citations:
            index = esc(cit.get("index", ""))
            source_file = esc(cit.get("source_file", "Unknown source"))
            page_number = esc(cit.get("page_number", "N/A"))
            snippet = esc(cit.get("text_snippet", ""))
            st.markdown(
                '<div class="citation-card">'
                f'<span class="citation-index">{index}</span>'
                f'<strong>{source_file}</strong> - '
                f'<span style="color: var(--text-muted);">Page {page_number}</span><br>'
                f'<span style="color: var(--text-secondary); font-size: 0.8rem;">{snippet}</span>'
                '</div>',
                unsafe_allow_html=True,
            )


def render_verification(verification: dict | None, faithfulness: dict | None = None) -> None:
    if verification:
        if verification.get("is_grounded"):
            confidence = verification.get("confidence", 0)
            st.markdown(
                f'<div class="faith-ok">Verified: grounded in document (confidence: {confidence:.0%})</div>',
                unsafe_allow_html=True,
            )
        else:
            issues = verification.get("issues", [])
            issue_text = " | ".join(str(issue) for issue in issues) if issues else "Answer may not be fully supported"
            st.markdown(
                f'<div class="faith-warning">Verification warning: {esc(issue_text)}</div>',
                unsafe_allow_html=True,
            )
        return

    if faithfulness is None:
        return

    if faithfulness.get("is_faithful"):
        score = faithfulness.get("score", 0)
        st.markdown(
            f'<div class="faith-ok">Answer verified (confidence: {score:.0%})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="faith-warning">Low confidence: answer may not be fully supported by sources</div>',
            unsafe_allow_html=True,
        )


def render_structured(structured: dict | None) -> None:
    if not structured or not structured.get("items"):
        return

    with st.expander(f"Extracted items ({structured.get('total_count', 0)} found)"):
        for item in structured["items"]:
            st.markdown(
                '<div class="citation-card">'
                f'<strong>{esc(item.get("name", "Unknown"))}</strong><br>'
                f'{esc(item.get("details", ""))}'
                '</div>',
                unsafe_allow_html=True,
            )


def render_rewritten_query(query: str | None) -> None:
    if not query:
        return
    st.markdown(
        f'<div class="rewritten-query">Rewritten query: {esc(query)}</div>',
        unsafe_allow_html=True,
    )


def render_assistant_metadata(message: dict) -> None:
    render_intent(message.get("intent"))
    render_citations(message.get("citations", []))
    render_verification(message.get("verification"), message.get("faithfulness"))
    render_structured(message.get("structured"))
    render_rewritten_query(message.get("rewritten_query"))


with st.sidebar:
    st.markdown('<div class="sidebar-brand">RAG Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-copy">Upload source documents, monitor indexing, and keep the chat workspace tidy.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading"><h4>API configuration</h4></div>', unsafe_allow_html=True)
    with st.expander(f"Backend: {get_backend_url()}", expanded=False):
        candidate = st.text_input(
            "Backend URL",
            value=get_backend_url(),
            help="Full URL of the FastAPI backend, e.g. https://rag-pipeline-backend-odlr.onrender.com",
            key="api_base_input",
        )
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("Save", use_container_width=True):
                st.session_state.api_base = _normalize_backend_url(candidate)
                st.rerun()
        with col_reset:
            if st.button("Reset", use_container_width=True):
                st.session_state.api_base = _DEFAULT_BACKEND_URL
                st.rerun()

    health = check_backend_health()

    st.markdown('<div class="section-heading"><h4>System status</h4></div>', unsafe_allow_html=True)
    render_status(health)

    st.markdown('<div class="section-heading"><h4>Upload documents</h4></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag and drop PDF, Markdown, or text files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                result = upload_file(uploaded_file)
                if not result:
                    continue

                status = result.get("status", "unknown")
                chunks = result.get("chunk_count", 0)
                if status == "unchanged":
                    st.info(f"{uploaded_file.name}: Already indexed ({chunks} chunks)")
                elif status == "updated":
                    st.success(f"{uploaded_file.name}: Updated ({chunks} chunks)")
                elif status == "indexed":
                    st.success(f"{uploaded_file.name}: Indexed ({chunks} chunks)")
                else:
                    st.warning(f"{uploaded_file.name}: {status}")

    st.markdown('<div class="section-heading"><h4>Indexed documents</h4></div>', unsafe_allow_html=True)
    docs = fetch_documents() if health else []
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    '<div class="doc-item">'
                    f'<strong>{esc(doc.get("filename", "Untitled"))}</strong><br>'
                    f'<span style="color: var(--text-muted); font-size: 0.75rem;">{esc(doc.get("chunk_count", 0))} chunks</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                filename = doc.get("filename", "")
                if st.button("Delete", key=f"del_{filename}", help=f"Delete {filename}"):
                    if delete_doc(filename):
                        st.rerun()
    else:
        st.markdown('<div class="empty-state">No documents indexed yet.</div>', unsafe_allow_html=True)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()


st.markdown(
    '<div class="main-header">'
    '<h1>Production RAG Pipeline</h1>'
    '<p>Ask questions about uploaded documents and review grounded answers with citations, verification signals, and extracted structured results.</p>'
    '</div>',
    unsafe_allow_html=True,
)
render_metrics(health)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_assistant_metadata(message)


if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.container()

        with response_container:
            status_placeholder = st.empty()
            status_placeholder.markdown(
                '<div style="color: var(--text-muted); font-style: italic; padding: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">'
                '<span class="status-dot online" style="animation-duration: 0.5s;"></span>'
                'Searching documents and generating a grounded response...'
                '</div>',
                unsafe_allow_html=True,
            )

            text_placeholder = st.empty()

            if not health:
                status_placeholder.empty()
                text_placeholder.markdown(
                    '<div class="error-panel">Backend is not available. Please start the server with <code>python run.py</code>.</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Backend is not available. Please start the server.",
                    }
                )
            else:
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.conversation_history[-5:]
                ]

                events = stream_query_live(prompt, text_placeholder, history)
                status_placeholder.empty()

                if events.get("error"):
                    text_placeholder.markdown(
                        f'<div class="error-panel">Error: {esc(events["error"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Error: {events['error']}",
                        }
                    )
                else:
                    answer = events.get("full_answer", "")

                    if events.get("unfaithful_replacement"):
                        answer = events["unfaithful_replacement"]
                        text_placeholder.markdown(answer)

                    assistant_msg = {
                        "role": "assistant",
                        "content": answer,
                        "citations": events.get("citations", []),
                        "faithfulness": events.get("faithfulness"),
                        "verification": events.get("verification"),
                        "structured": events.get("structured"),
                        "intent": events.get("intent"),
                        "rewritten_query": events.get("rewritten_query"),
                        "search_results_count": events.get("search_results_count", 0),
                    }
                    st.session_state.messages.append(assistant_msg)

                    st.session_state.conversation_history.append({"role": "user", "content": prompt})
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

                    render_assistant_metadata(assistant_msg)
