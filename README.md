# 🕉️ Geeta RAG — Multilingual Knowledge Extraction & Exploration Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20llama--3.3--70b-orange?logo=meta" />
  <img src="https://img.shields.io/badge/Embeddings-multilingual--MiniLM--L12-green?logo=huggingface" />
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS-red" />
  <img src="https://img.shields.io/badge/Language-Gujarati%20%7C%20Sanskrit%20%7C%20English-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> A production-style RAG (Retrieval-Augmented Generation) pipeline built over the **Yatharthgeeta** — a multilingual Gujarati/Sanskrit commentary on the Bhagavad Gita (Chapters 1–10). Extract, chunk, embed, and query a multilingual scripture using Groq's LLM.

***

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Modules](#-modules)
- [Quick Start](#-quick-start)
- [CLI Usage](#-cli-usage)
- [Design Decisions](#-design-decisions)
- [Trade-offs & Limitations](#-trade-offs--limitations)
- [Future Improvements](#-future-improvements)
- [Project Structure](#-project-structure)

***

## 🔍 Overview

This project implements a **3-module AI assistant** that:

1. **Extracts and cleans** multilingual text (Gujarati Unicode + Devanagari Sanskrit) from a digitized PDF using `PyMuPDF`
2. **Chunks and embeds** the content using a structure-aware strategy + multilingual sentence embeddings stored in a FAISS index
3. **Answers questions** in English over the Gujarati/Sanskrit source text using Groq's `llama-3.3-70b-versatile` via RAG

The reference document is the **Yatharthgeeta by Swami Adgadanand** — a verse-by-verse Gujarati commentary on the Bhagavad Gita (Chapters 1–10). Each chunk preserves the pairing of a Sanskrit shloka with its Gujarati explanation.

***

## 🏗️ Architecture

```
PDF (Gujarati + Sanskrit)
        │
        ▼
┌─────────────────────────┐
│  Module 1: Extraction   │  ← PyMuPDF UTF-8 extraction, Unicode-aware
│  & Preprocessing        │    cleaning, structure detection
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 2a: Chunking    │  ← Structure-based (verse markers) +
│  Strategy               │    Sliding-window fallback (800c, 150 overlap)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 2b: Embeddings  │  ← paraphrase-multilingual-MiniLM-L12-v2
│  (384-dim, 50+ langs)   │    L2-normalised → cosine similarity
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 2c: FAISS Index │  ← IndexFlatIP (exact cosine, in-memory)
└────────────┬────────────┘
             │
        query vec
             │
             ▼
┌────────────────────────────────────┐
│  Module 3: Groq RAG Engine         │
│  Retrieve (top-5) → Augment →      │
│  Generate (llama-3.3-70b)          │
│  + 4-turn conversation history     │
└────────────────────────────────────┘
```

***

## 📦 Modules

### Module 1 — Document Understanding & Preprocessing

| Step | Tool | Detail |
|---|---|---|
| Text extraction | `PyMuPDF (fitz)` | Reads embedded Unicode — no OCR needed for this PDF |
| Cleaning | `re` (regex) | Removes control chars, normalises blank lines, strips trailing spaces |
| Structure detection | Pattern matching | Detects Sanskrit shloka markers (`JJ1JJ`, `JJ2JJ`…) + Gujarati/Devanagari chapter headings |

> **Multilingual note:** The Yatharthgeeta PDF has embedded Unicode for both Gujarati and Devanagari scripts. `PyMuPDF` extracts these cleanly. For truly scanned or handwritten pages, `surya-ocr` or Google Cloud Vision would be required as a drop-in replacement.

***

### Module 2 — Chunking + Embedding Strategy

#### Chunking: Structure-Based + Sliding-Window Fallback

```
Verse marker detected (JJnJJ)?
        │
        ├── YES → One chunk = Sanskrit shloka + Gujarati commentary
        │          (preserves semantic integrity of verse ↔ explanation)
        │
        └── Block > 800 chars?
                   ├── YES → Sliding window (800 chars, 150-char overlap)
                   └── NO  → Single chunk as-is
```

Each chunk carries metadata: `page`, `chapter`, `verse_id`, `section_type`, `char_len`.

#### Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|---|---|
| Languages supported | 50+ (Gujarati, Hindi, Sanskrit, English…) |
| Vector dimension | 384 |
| Model size on disk | ~118 MB |
| Cross-lingual retrieval | ✅ English query retrieves Gujarati passage |
| Normalisation | L2 → inner product = cosine similarity |

> **Why not LaBSE?** LaBSE (768-dim, ~470 MB) offers slightly better accuracy on low-resource scripts but is 4× slower on CPU. For this task scope, MiniLM-L12 is sufficient and stays well within a 45-minute build window.

#### Vector Store: FAISS `IndexFlatIP`

- In-memory, no server setup required
- Exact nearest-neighbour search (not approximate / HNSW)
- Inner product on L2-normalised vectors = cosine similarity

***

### Module 3 — Question Answering (RAG + Groq)

**Pipeline per query:**

1. Embed question using the same multilingual model
2. Retrieve **top-5 chunks** via FAISS cosine search
3. Inject retrieved passages as context into a crafted system prompt
4. `llama-3.3-70b-versatile` on Groq generates an answer with chapter/verse citations
5. **4-turn conversation history** maintained for natural follow-up questions

**Bonus — keyword search:** `/keyword <term>` performs direct substring match across all chunks with zero LLM cost.

***

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/mananjp/geeta_rag.git
cd geeta_rag/rag_geeta_submission

pip install -r requirements.txt
```

### 2. Get a Free Groq API Key

Sign up at [console.groq.com](https://console.groq.com) → **API Keys** → Create Key.

```bash
export GROQ_API_KEY=gsk_your_key_here
```

### 3. Run

```bash
# Interactive chat mode (default)
python rag_geeta.py \
  --pdf GEN-AI-TASK-REF-FILE-Geeta-demo-1-10.pdf \
  --api-key $GROQ_API_KEY

# 5 automated demo questions — good for submission output
python rag_geeta.py \
  --pdf GEN-AI-TASK-REF-FILE-Geeta-demo-1-10.pdf \
  --api-key $GROQ_API_KEY \
  --demo
```

> ⏱️ **First run** downloads the embedding model (~118 MB via HuggingFace). All subsequent runs are instant.

***

## 💻 CLI Usage

| Input | Action |
|---|---|
| Any natural language question | Full RAG answer with chapter/verse citation |
| `/keyword <word>` | Direct substring search across all chunks (no LLM) |
| `/sources` | Toggle retrieved source passage display ON / OFF |
| `/quit` | Exit the assistant |

### Example Session

```
You: What is Dhritarashtra asking Sanjaya at the start of the Gita?

╭──────────────── Assistant ─────────────────╮
│ In Chapter 1, Verse 1, Dhritarashtra asks  │
│ Sanjaya: "What did my sons and the Pandavas│
│ do after gathering on the sacred field of  │
│ Kurukshetra, eager for battle?"            │
│                                            │
│ The Gujarati commentary explains that      │
│ Dhritarashtra represents ignorance         │
│ (अज्ञान) and Sanjaya represents           │
│ self-restraint (संयम) — the ignorant mind  │
│ perceives reality only through the lens of │
│ controlled observation.                    │
╰────────────────────────────────────────────╯
  ↳ 1.83s | 412 tokens used

You: /keyword ભે
Keyword 'ભે': 7 matches
...

You: /sources     ← toggles source table on
You: /quit
```

***

## 🧠 Design Decisions

### Why structure-based chunking over semantic chunking?

Semantic chunking (embedding-based boundary detection) requires clean, predominantly single-language text to produce meaningful split points. Mixed Gujarati–Sanskrit–Devanagari content confuses sentence-boundary models. Structure-based splitting on verse markers (`JJ1JJ`) is:
- **Language-agnostic** — works on any script
- **Semantically meaningful** — verse + commentary is the natural unit of this text
- **Deterministic** — reproducible without model inference at indexing time

### Why FAISS over ChromaDB / Qdrant?

For a < 45-minute build with a small corpus (< 500 chunks), FAISS `IndexFlatIP` requires zero server, zero configuration, and zero persistent storage. ChromaDB or Qdrant would be preferred for production with persistent storage and filtered search.

### Why Groq over OpenAI?

Groq's free tier provides the fastest inference available for `llama-3.3-70b` (often < 1s TTFT), which is ideal for a timed corporate task demo.

***

## ⚖️ Trade-offs & Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| FAISS index rebuilt every run | ~30s startup on large PDFs | Add `faiss.write_index()` / `read_index()` for persistence |
| MiniLM-L12 vs LaBSE | Slightly lower recall on rare Sanskrit terms | Swap model name to `LaBSE` for better coverage |
| No scanned-image support | Purely image-based pages yield empty text | Integrate `surya-ocr` as a pre-pass |
| Exact FAISS search | O(n) at query time | Switch to `IndexHNSWFlat` for 1M+ chunks |
| Groq free tier rate limits | ~30 req/min | Add retry logic with `tenacity` |

***

## 🔮 Future Improvements

- [ ] **Persistent index** — `faiss.write_index()` + pickle chunk metadata to disk; skip rebuild on re-run
- [ ] **Scanned page support** — integrate `surya-ocr` for handwritten/image-only Gujarati pages
- [ ] **Hybrid search** — BM25 (exact term matching) + dense retrieval, merged via Reciprocal Rank Fusion
- [ ] **Cross-encoder re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-20 → top-5 for higher precision
- [ ] **Streamlit UI** — replace CLI with a web interface including source highlighting
- [ ] **Chapter-level filtering** — pre-filter FAISS results by chapter before LLM call
- [ ] **IndicBERT embeddings** — `ai4bharat/indic-bert` for richer Gujarati/Sanskrit semantic space
- [ ] **Translation layer** — auto-translate retrieved Gujarati chunks to English before LLM context injection

***

## 📁 Project Structure

```
rag_geeta_submission/
├── rag_geeta.py                          # Main pipeline (all 3 modules)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── GEN-AI-TASK-REF-FILE-Geeta-demo-1-10.pdf  # Reference PDF (not tracked in git)
```

### Key classes in `rag_geeta.py`

| Class | Responsibility |
|---|---|
| `extract_text_from_pdf()` | Module 1 — PyMuPDF extraction + cleaning |
| `detect_structure()` | Module 1 — labels pages with chapter/section_type |
| `ChunkingStrategy` | Module 2a — structure-based + sliding-window chunker |
| `MultilingualEmbedder` | Module 2b — wraps `SentenceTransformer` |
| `VectorStore` | Module 2c — FAISS IndexFlatIP wrapper |
| `RAGAssistant` | Module 3 — Retrieve → Augment → Generate via Groq |
| `run_cli()` | Interactive Rich terminal UI |
| `demo_run()` | Non-interactive 5-question demo mode |

***

## 📄 Requirements

```
pymupdf>=1.23.0
sentence-transformers>=2.6.0
faiss-cpu>=1.7.4
groq>=0.9.0
numpy>=1.24.0
rich>=13.0.0
```

Python **3.10+** required.

***

## 🙏 Credits

- **Yatharthgeeta** — Swami Adgadanand (Gujarati commentary on Bhagavad Gita)
- **Groq** — Ultra-fast LLM inference API
- **sentence-transformers** — `paraphrase-multilingual-MiniLM-L12-v2`
- **FAISS** — Facebook AI Similarity Search

***

*Built by [Manan Panchal](https://github.com/mananjp) — B.Tech AIML, Charusat University*
