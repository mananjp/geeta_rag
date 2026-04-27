#!/usr/bin/env python3
"""
=============================================================================
  Multilingual Knowledge Extraction & Exploration Assistant
  Task: RAG over Yatharthgeeta (Gujarati/Sanskrit Bhagavad Gita, Ch 1-10)
  Author: Manan Panchal | B.Tech AIML, Charusat
  Stack : PyMuPDF · sentence-transformers · FAISS · Groq LLM
=============================================================================
"""

import os
import re
import sys
import time
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ── dependency check ─────────────────────────────────────────────────────────
REQUIRED = {
    "fitz"                  : "pymupdf",
    "sentence_transformers" : "sentence-transformers",
    "faiss"                 : "faiss-cpu",
    "groq"                  : "groq",
    "numpy"                 : "numpy",
    "rich"                  : "rich",
}

import importlib, subprocess
for mod, pkg in REQUIRED.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"[INSTALL] {pkg} …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# ── real imports ──────────────────────────────────────────────────────────────
import fitz                              # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.prompt  import Prompt
from rich         import print as rprint
from rich.markup  import escape

import io as _io
console = Console(file=_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Document Understanding + Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of the PDF.
    Handles mixed Gujarati / Sanskrit / Devanagari / English content.
    Returns a list of {page, raw_text, clean_text}.
    """
    import json
    
    # Check for pre-processed JSON cache (for legacy font decoding)
    json_cache = Path(pdf_path).with_suffix('.json')
    if json_cache.exists():
        console.print(f"[cyan]Loading cleaned text from cache:[/cyan] {json_cache.name}")
        with open(json_cache, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    doc   = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        raw   = page.get_text("text")          # UTF-8 text layer
        clean = _clean_text(raw)
        if clean.strip():
            pages.append({
                "page"      : i + 1,
                "raw_text"  : raw,
                "clean_text": clean,
            })

    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """
    Light cleanup that preserves Gujarati / Devanagari Unicode.
    - Collapse blank lines > 2
    - Strip trailing whitespace per line
    - Remove page-artifact characters (\x00 etc.)
    """
    # remove null / control chars (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # normalise multiple blank lines → max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing spaces per line
    lines = [l.rstrip() for l in text.splitlines()]
    return "\n".join(lines).strip()


def detect_structure(pages: List[Dict]) -> List[Dict]:
    """
    Identify structural elements:
      - Sanskrit shloka (Devanagari lines ending with JJ…JJ)
      - Chapter heading
      - Gujarati commentary paragraph
    Adds 'section_type' and 'chapter' to each page entry.
    """
    SHLOKA_RE  = re.compile(r"JJ\d+JJ|\|\|\d+\|\|")
    CHAPTER_RE = re.compile(r"(અધ્યાય|आध्याय|chapter)", re.IGNORECASE)
    current_chapter = 1

    for p in pages:
        text = p["clean_text"]
        if CHAPTER_RE.search(text):
            # try to extract chapter number
            m = re.search(r"(\d+)", text[:100])
            if m:
                current_chapter = int(m.group(1))
        p["chapter"] = current_chapter

        if SHLOKA_RE.search(text):
            p["section_type"] = "shloka+commentary"
        else:
            p["section_type"] = "commentary"

    return pages


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — Chunking + Embedding Strategy
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkingStrategy:
    """
    Structure-aware chunker for multilingual scripture text.

    Strategy chosen: STRUCTURE-BASED + SLIDING-WINDOW FALLBACK
    ─────────────────────────────────────────────────────────────
    • Primary  : Split on verse/shloka markers (JJ<num>JJ) so each chunk
                 contains exactly one shloka + its Gujarati commentary.
                 This preserves semantic integrity (verse ↔ explanation).
    • Fallback : If a block is > max_chars, apply a sliding window with
                 50 % overlap so no context is lost at boundary.
    • Metadata : Each chunk carries page, chapter, verse_id, section_type.

    Why not purely semantic chunking?
    ──────────────────────────────────
    Semantic chunking (embedding-based) needs clean English; it breaks on
    mixed-script text.  Structure-based is deterministic & language-agnostic.
    """

    VERSE_SPLIT_RE = re.compile(r"(JJ\d+JJ|\|\|\d+\|\|)")

    def __init__(self, max_chars: int = 800, overlap_chars: int = 150):
        self.max_chars    = max_chars
        self.overlap_chars = overlap_chars

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        chunks = []
        chunk_id = 0

        for page in pages:
            text    = page["clean_text"]
            chapter = page.get("chapter", 1)
            page_no = page["page"]
            s_type  = page.get("section_type", "commentary")

            # Split by verse markers
            parts  = self.VERSE_SPLIT_RE.split(text)
            blocks = self._merge_parts(parts)

            for block, verse_id in blocks:
                block = block.strip()
                if len(block) < 30:          # skip noise / empty
                    continue

                sub_chunks = self._sliding_window(block) if len(block) > self.max_chars else [block]

                for sc in sub_chunks:
                    sc = sc.strip()
                    if sc:
                        chunks.append({
                            "id"          : chunk_id,
                            "text"        : sc,
                            "page"        : page_no,
                            "chapter"     : chapter,
                            "verse_id"    : verse_id,
                            "section_type": s_type,
                            "char_len"    : len(sc),
                        })
                        chunk_id += 1

        return chunks

    def _merge_parts(self, parts: List[str]) -> List[Tuple[str, Optional[str]]]:
        """Pair each text block with the verse marker that precedes it."""
        result = []
        current_verse = None
        buffer = []

        for part in parts:
            if self.VERSE_SPLIT_RE.match(part):
                if buffer:
                    result.append(("".join(buffer), current_verse))
                    buffer = []
                current_verse = part
            else:
                buffer.append(part)

        if buffer:
            result.append(("".join(buffer), current_verse))

        return result

    def _sliding_window(self, text: str) -> List[str]:
        chunks = []
        step   = self.max_chars - self.overlap_chars
        start  = 0
        while start < len(text):
            end = start + self.max_chars
            chunks.append(text[start:end])
            start += step
        return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

class MultilingualEmbedder:
    """
    Wraps a multilingual sentence-transformer model.

    Model chosen: paraphrase-multilingual-MiniLM-L12-v2
    ────────────────────────────────────────────────────
    • Supports 50+ languages including Gujarati, Hindi (Devanagari), Sanskrit.
    • 384-dim vectors — small enough to be fast on CPU in < 45 min.
    • Strong cross-lingual alignment: query in English retrieves Gujarati passages.

    Alternative considered: LaBSE (language-agnostic BERT sentence embedding)
    — better for low-resource scripts but ~470 MB vs 118 MB here.
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        console.print(f"[cyan]Loading embedding model:[/cyan] {self.MODEL_NAME} …")
        self.model = SentenceTransformer(self.MODEL_NAME)
        console.print("[green]✔ Model loaded.[/green]")

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size      = batch_size,
            show_progress_bar = show_progress,
            normalize_embeddings = True,      # cosine sim → dot product
            convert_to_numpy     = True,
        )
        return vecs.astype("float32")


# ── FAISS Vector Store ─────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory FAISS index (IndexFlatIP = exact inner-product / cosine search).
    Stores chunk metadata alongside for retrieval.
    """

    def __init__(self, dim: int):
        self.dim    = dim
        self.index  = faiss.IndexFlatIP(dim)    # cosine (vectors normalised)
        self.chunks : List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results

    def size(self) -> int:
        return self.index.ntotal

    @property
    def stats(self) -> Dict:
        chapters = set(c["chapter"] for c in self.chunks)
        return {
            "total_chunks" : self.size(),
            "chapters"     : sorted(chapters),
            "avg_chunk_len": int(np.mean([c["char_len"] for c in self.chunks])),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — Question Answering with Groq
# ═══════════════════════════════════════════════════════════════════════════════

class RAGAssistant:
    """
    Full RAG pipeline: Retrieve → Augment → Generate.

    LLM  : Groq  llama-3.3-70b-versatile  (fast, free-tier friendly)
    Retrieval: top-5 FAISS chunks, re-ranked by chapter relevance if hinted.
    """

    SYSTEM_PROMPT = """You are a knowledgeable assistant specialising in the Bhagavad Gita
and its Gujarati commentary (Yatharthgeeta by Swami Adgadanand).
You have been given context passages extracted from the book (Chapters 1-10).
The context text is primarily in Gujarati and Sanskrit, but you MUST respond in
whatever language the user is writing in.

Rules:
1. Base your answer ONLY on the provided context.
2. **CRITICAL RULE — You MUST reply in the SAME language as the user's question.**
   - User writes in Hindi → You MUST reply ENTIRELY in Hindi (Devanagari script).
   - User writes in Gujarati → You MUST reply ENTIRELY in Gujarati.
   - User writes in English → You MUST reply ENTIRELY in English.
   - User writes in Sanskrit → You MUST reply in Sanskrit.
   - The context passages may be in a DIFFERENT language than the user's question.
     That is fine — translate/interpret the context and reply in the USER's language.
   - NEVER reply in a different language than the question, even if the context is
     in another language.
3. When quoting shlokas, keep them in original script and add a translation in the
   user's language if needed.
4. Cite the chapter and verse number if available.
5. If the context is insufficient, say so honestly in the user's language.
6. Keep answers concise but complete (3-6 sentences unless asked for more)."""

    def __init__(self, vector_store: VectorStore, embedder: MultilingualEmbedder, api_key: str):
        self.vs       = vector_store
        self.embedder = embedder
        self.client   = Groq(api_key=api_key)
        self.model    = "llama-3.3-70b-versatile"
        self.history  : List[Dict] = []

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        q_vec = self.embedder.embed([query], show_progress=False)
        return self.vs.search(q_vec, top_k=top_k)

    def _build_context_block(self, hits: List[Dict]) -> str:
        parts = []
        for h in hits:
            verse = f"Verse {h['verse_id']}" if h.get("verse_id") else ""
            header = f"[Chapter {h['chapter']} | Page {h['page']} {verse} | Score: {h['score']:.3f}]"
            parts.append(f"{header}\n{h['text']}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect dominant script/language of a text string."""
        # Count characters in different Unicode script ranges
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')  # Hindi/Sanskrit
        gujarati   = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')  # Gujarati
        latin      = sum(1 for c in text if 'A' <= c <= 'z')            # English
        total = devanagari + gujarati + latin
        if total == 0:
            return "English"
        if devanagari / max(total, 1) > 0.3:
            return "Hindi"
        if gujarati / max(total, 1) > 0.3:
            return "Gujarati"
        return "English"

    def ask(self, question: str, top_k: int = 5) -> Dict:
        t0   = time.time()
        hits = self.retrieve(question, top_k=top_k)
        ctx  = self._build_context_block(hits)

        # Detect user's language for explicit instruction
        user_lang = self._detect_language(question)
        lang_instruction = (
            f"\n\n⚠️ IMPORTANT: The user is writing in {user_lang}. "
            f"You MUST reply ENTIRELY in {user_lang}. "
            f"Do NOT reply in any other language."
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        # last 4 turns of conversation history
        messages.extend(self.history[-4:])
        messages.append({
            "role"   : "user",
            "content": f"""Context passages from the book:
\n{ctx}\n\nQuestion: {question}{lang_instruction}""",
        })

        resp = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = 0.3,
            max_tokens  = 1024,
        )
        answer = resp.choices[0].message.content.strip()

        # update history (store clean Q&A without context)
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer"   : answer,
            "sources"  : hits,
            "latency_s": round(time.time() - t0, 2),
            "tokens"   : resp.usage.total_tokens,
        }

    def keyword_search(self, keyword: str) -> List[Dict]:
        """Simple BM25-style keyword search over chunk texts."""
        kw = keyword.lower()
        return [
            c for c in self.vs.chunks
            if kw in c["text"].lower()
        ][:10]


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]Multilingual RAG Assistant[/bold cyan]\n"
        "[dim]Yatharthgeeta · Chapters 1-10 · Gujarati + Sanskrit[/dim]",
        border_style="cyan"
    ))


def print_sources(hits: List[Dict]):
    t = Table(title="Retrieved Passages", show_header=True, header_style="bold magenta")
    t.add_column("Ch.", width=4)
    t.add_column("Pg.", width=4)
    t.add_column("Verse", width=8)
    t.add_column("Score", width=7)
    t.add_column("Preview", max_width=60)
    for h in hits:
        preview = escape(h["text"][:80].replace("\n", " "))
        t.add_row(
            str(h["chapter"]),
            str(h["page"]),
            str(h.get("verse_id") or "–"),
            f"{h['score']:.3f}",
            preview + "…",
        )
    console.print(t)


def run_cli(assistant: RAGAssistant):
    print_banner()

    stats = assistant.vs.stats
    console.print(
        f"[green]✔ Index ready:[/green] {stats['total_chunks']} chunks | "
        f"Chapters: {stats['chapters']} | "
        f"Avg chunk: {stats['avg_chunk_len']} chars"
    )
    console.print()
    console.print("[dim]Commands:  /keyword <word>  /sources  /quit[/dim]")
    console.print("[dim]Just type any question to ask the LLM.[/dim]")
    console.print()

    show_sources = False

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/sources":
            show_sources = not show_sources
            console.print(f"[cyan]Source display:[/cyan] {'ON' if show_sources else 'OFF'}")
            continue

        if user_input.lower().startswith("/keyword "):
            kw    = user_input[9:].strip()
            found = assistant.keyword_search(kw)
            console.print(f"[cyan]Keyword '{kw}':[/cyan] {len(found)} matches")
            for f in found[:3]:
                console.print(Panel(
                    escape(f["text"][:300]),
                    title=f"Ch.{f['chapter']} | Pg.{f['page']}",
                    border_style="dim"
                ))
            continue

        # Normal QA
        with console.status("[dim]Thinking …[/dim]"):
            result = assistant.ask(user_input)

        console.print()
        console.print(Panel(
            result["answer"],
            title="[bold green]Assistant[/bold green]",
            border_style="green"
        ))
        console.print(
            f"[dim]  ↳ {result['latency_s']}s | {result['tokens']} tokens used[/dim]"
        )

        if show_sources:
            print_sources(result["sources"])

        console.print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  — wire everything together
# ═══════════════════════════════════════════════════════════════════════════════

def build_rag_pipeline(pdf_path: str, groq_api_key: str) -> RAGAssistant:
    console.rule("[bold]Module 1 — Document Extraction & Preprocessing[/bold]")
    pages = extract_text_from_pdf(pdf_path)
    pages = detect_structure(pages)
    console.print(f"[green]✔[/green] Extracted {len(pages)} pages")

    console.rule("[bold]Module 2 — Chunking Strategy[/bold]")
    chunker = ChunkingStrategy(max_chars=800, overlap_chars=150)
    chunks  = chunker.chunk_pages(pages)
    console.print(
        f"[green]✔[/green] Created {len(chunks)} chunks "
        f"(structure-based + sliding-window fallback)"
    )

    console.rule("[bold]Module 2b — Multilingual Embeddings[/bold]")
    embedder = MultilingualEmbedder()
    texts    = [c["text"] for c in chunks]
    vecs     = embedder.embed(texts, batch_size=64)
    console.print(f"[green]✔[/green] Embedded {len(vecs)} chunks → {vecs.shape[1]}-dim vectors")

    console.rule("[bold]Module 2c — FAISS Vector Index[/bold]")
    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, chunks)
    console.print(f"[green]✔[/green] FAISS IndexFlatIP built — {store.size()} vectors")

    console.rule("[bold]Module 3 — Groq RAG QA Engine[/bold]")
    assistant = RAGAssistant(store, embedder, groq_api_key)
    console.print("[green]✔[/green] Groq LLM ready (llama-3.3-70b-versatile)")

    return assistant


def demo_run(assistant: RAGAssistant):
    """
    Run a few demo questions and print structured output.
    Used in non-interactive / notebook mode.
    """
    demo_questions = [
        "What is Dhritarashtra asking Sanjaya at the beginning of the Gita?",
        "Who are the main warriors mentioned in the Kaurava army?",
        "What is the significance of Dharmashetra and Kurukshetra?",
        "What does Duryodhana say after seeing the Pandava army?",
        "Explain the inner meaning of Bhishma in the Geeta commentary.",
    ]

    console.rule("[bold cyan]Demo Q&A Session[/bold cyan]")
    for i, q in enumerate(demo_questions, 1):
        console.print(f"\n[bold yellow]Q{i}:[/bold yellow] {q}")
        result = assistant.ask(q)
        console.print(Panel(
            result["answer"],
            title=f"[green]A{i}[/green]",
            border_style="green"
        ))
        console.print(f"[dim]  Latency: {result['latency_s']}s | Tokens: {result['tokens']}[/dim]")
        time.sleep(0.5)   # avoid rate-limit on free Groq tier


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multilingual RAG Assistant for Yatharthgeeta"
    )
    parser.add_argument(
        "--pdf",
        default="GEN-AI-TASK-REF-FILE-Geeta-demo-1-10.pdf",
        help="Path to the reference PDF (default: GEN-AI-TASK-REF-FILE-Geeta-demo-1-10.pdf)"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GROQ_API_KEY", ""),
        help="Groq API key (or set GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run 5 demo questions instead of interactive mode"
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        console.print(f"[red]ERROR:[/red] PDF not found: {args.pdf}")
        sys.exit(1)

    if not args.api_key:
        console.print("[red]ERROR:[/red] Groq API key missing. "
                      "Set --api-key or export GROQ_API_KEY=...")
        sys.exit(1)

    assistant = build_rag_pipeline(args.pdf, args.api_key)

    if args.demo:
        demo_run(assistant)
    else:
        run_cli(assistant)
