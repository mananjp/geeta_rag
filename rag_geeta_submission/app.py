#!/usr/bin/env python3
"""
Streamlit UI for the Multilingual RAG Assistant (Yatharthgeeta)
"""

import os
import sys
import time
import html as _html
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Gita RAG Assistant",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS for premium look ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.main-header h1 {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 2rem;
    margin: 0 0 0.3rem 0;
}
.main-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* Chat bubbles */
.user-msg {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.5rem 0;
    max-width: 85%;
    margin-left: auto;
    box-shadow: 0 2px 12px rgba(99,102,241,0.3);
    font-size: 0.95rem;
    line-height: 1.5;
}
.assistant-msg {
    background: linear-gradient(135deg, #1e293b, #334155);
    color: #e2e8f0;
    padding: 1rem 1.25rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.5rem 0;
    max-width: 85%;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Source cards */
.source-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.source-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.source-meta {
    color: #818cf8;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.source-text {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* Stats pills */
.stat-pill {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: white;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #94a3b8 !important;
}

/* Keyword results */
.kw-result {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    color: #cbd5e1;
    font-size: 0.85rem;
}

/* Loading animation */
.loading-text {
    color: #818cf8;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Import core modules from rag_geeta ───────────────────────────────────────
# Add current dir to path so we can import
sys.path.insert(0, str(Path(__file__).parent))

from rag_geeta import (
    extract_text_from_pdf,
    detect_structure,
    ChunkingStrategy,
    MultilingualEmbedder,
    VectorStore,
    RAGAssistant,
)


# ── Constants ────────────────────────────────────────────────────────────────
PDF_PATH = Path(__file__).parent / "GEN AI TASK REF FILE Geeta-demo-1-10.pdf"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ── Pipeline builder (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(api_key: str):
    """Build the full RAG pipeline — cached across reruns."""
    pages = extract_text_from_pdf(str(PDF_PATH))
    pages = detect_structure(pages)

    chunker = ChunkingStrategy(max_chars=800, overlap_chars=150)
    chunks = chunker.chunk_pages(pages)

    embedder = MultilingualEmbedder()
    texts = [c["text"] for c in chunks]
    vecs = embedder.embed(texts, batch_size=64)

    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, chunks)

    assistant = RAGAssistant(store, embedder, api_key)
    stats = store.stats
    return assistant, stats


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    show_sources = st.toggle("Show retrieved sources", value=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "RAG assistant over **Yatharthgeeta** (Chapters 1-10). "
        "Ask questions in **English, Hindi, Gujarati, or Sanskrit** — "
        "the multilingual embedder retrieves across all languages."
    )

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "What is Dhritarashtra asking Sanjaya?",
        "Who are the warriors in the Kaurava army?",
        "Significance of Dharmashetra and Kurukshetra?",
        "What does Duryodhana say about the Pandava army?",
        "Inner meaning of Bhishma in the commentary",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["prefill_q"] = q

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Built with PyMuPDF · sentence-transformers · "
        "FAISS · Groq · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🙏 Multilingual RAG Assistant</h1>
        <p>Yatharthgeeta · Chapters 1-10 · Gujarati + Sanskrit + English</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── API key ──────────────────────────────────────────────────────────────────
api_key = GROQ_API_KEY

# ── Guard: PDF must exist ───────────────────────────────────────────────────
if not PDF_PATH.exists():
    st.error(f"PDF not found at `{PDF_PATH}`")
    st.stop()


# ── Load pipeline ───────────────────────────────────────────────────────────
with st.spinner("🔄 Loading pipeline — extracting PDF, embedding chunks, building FAISS index…"):
    assistant, stats = load_pipeline(api_key)

# Stats bar
col1, col2, col3 = st.columns(3)
col1.metric("📄 Total Chunks", stats["total_chunks"])
col2.metric("📚 Chapters", f"{min(stats['chapters'])}–{max(stats['chapters'])}")
col3.metric("📏 Avg Chunk Length", f"{stats['avg_chunk_len']} chars")


# ── Source rendering helper ──────────────────────────────────────────────────
def _deduplicate_lines(text: str, max_lines: int = 12) -> str:
    """Remove consecutive duplicate lines from text (anti-hallucination cleanup)."""
    lines = text.split('\n')
    deduped = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            continue  # skip consecutive duplicate
        deduped.append(line)
        prev = stripped
    # Limit to max_lines
    if len(deduped) > max_lines:
        deduped = deduped[:max_lines] + ["…"]
    return '\n'.join(deduped)


def _render_sources(hits):
    """Render retrieved source passages as clean cards."""
    for i, s in enumerate(hits):
        verse = f"Verse {s['verse_id']}" if s.get("verse_id") else "—"
        score_pct = f"{s['score'] * 100:.1f}%"

        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-meta">'
            f'📖 Chapter {s["chapter"]} &nbsp;|&nbsp; 📄 Page {s["page"]} &nbsp;|&nbsp; '
            f'🔢 {verse} &nbsp;|&nbsp; 🎯 Relevance: {score_pct}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"View passage #{i+1}", expanded=False):
            cleaned = _deduplicate_lines(s["text"])
            st.text(cleaned)


# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Render chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        # show sources if stored
        if show_sources and msg.get("sources"):
            with st.expander("📑 Retrieved Sources", expanded=False):
                _render_sources(msg["sources"])


# ── Chat input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_q", None)
user_input = st.chat_input("Ask a question about the Bhagavad Gita…") or prefill

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-msg">{user_input}</div>', unsafe_allow_html=True)

    # Keyword search shortcut
    if user_input.strip().lower().startswith("/keyword "):
        kw = user_input.strip()[9:]
        results = assistant.keyword_search(kw)
        reply = f"**Keyword search for** `{kw}` — **{len(results)}** matches found."
        if results:
            reply += "\n\n"
            for r in results[:5]:
                verse = f"Verse {r['verse_id']}" if r.get("verse_id") else ""
                reply += f"- **Ch.{r['chapter']} Pg.{r['page']}** {verse}: {r['text'][:120]}…\n"
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
        st.rerun()
    else:
        # RAG QA
        with st.spinner("🤔 Thinking…"):
            result = assistant.ask(user_input)

        answer = result["answer"]
        latency = result["latency_s"]
        tokens = result["tokens"]
        sources = result["sources"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        st.markdown(f'<div class="assistant-msg">{answer}</div>', unsafe_allow_html=True)
        st.caption(f"⏱ {latency}s · {tokens} tokens")

        if show_sources and sources:
            with st.expander("📑 Retrieved Sources", expanded=False):
                _render_sources(sources)
