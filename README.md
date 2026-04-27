<![CDATA[# 🙏 Multilingual Knowledge Extraction & Exploration Assistant

### RAG over Yatharthgeeta (Gujarati / Sanskrit Bhagavad Gita, Chapters 1–10)

**Author:** Manan Panchal · B.Tech AIML, Charusat  
**Stack:** PyMuPDF · sentence-transformers · FAISS · Groq LLM · Streamlit

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Models, Tools & Techniques Tried](#models-tools--techniques-tried)
4. [What Worked Well](#what-worked-well)
5. [What Didn't Work / Challenges](#what-didnt-work--challenges)
6. [Handling Multilingual & Handwritten Content](#handling-multilingual--handwritten-content)
7. [Trade-offs & Design Decisions](#trade-offs--design-decisions)
8. [Improvements & Extensions with More Time](#improvements--extensions-with-more-time)
9. [Quick Start](#quick-start)
10. [CLI Commands](#cli-commands)

---

## Project Overview

This project builds a **Retrieval-Augmented Generation (RAG)** assistant that lets users ask questions about the *Yatharthgeeta* — a Gujarati commentary on the Bhagavad Gita (Chapters 1–10) by Swami Adgadanand. The system handles **multilingual content** spanning Sanskrit shlokas, Gujarati commentary, and English — and responds in the **same language the user asks in**.

### Key Capabilities

| Feature | Description |
|---|---|
| **Cross-lingual QA** | Ask in English → retrieve Gujarati passages → answer in English (or any supported language) |
| **Structure-aware chunking** | Preserves shloka ↔ commentary pairs as semantic units |
| **Conversational memory** | Maintains last 4 turns for follow-up questions |
| **Dual interface** | Rich CLI (via `rich`) + Streamlit web UI (`app.py`) |
| **Keyword search** | Exact substring matching across all indexed chunks |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION PIPELINE                     │
│                                                                     │
│   PDF ──► PyMuPDF Text Extraction ──► Regex Cleanup & Normalisation │
│                                              │                      │
│                                    Structure Detection              │
│                               (shloka markers, chapter headings)    │
│                                              │                      │
│                                    Chunking Strategy                │
│                           (Structure-based + Sliding Window)        │
│                                              │                      │
│                                 Multilingual Embedder               │
│                        (paraphrase-multilingual-MiniLM-L12-v2)      │
│                                              │                      │
│                               FAISS IndexFlatIP                     │
│                            (exact cosine similarity)                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                        RAG QUERY PIPELINE                           │
│                                                                     │
│   User Query ──► Embed with same model ──► Top-5 FAISS Retrieval   │
│                                                     │               │
│                                            Context Assembly         │
│                                      (passages + metadata)          │
│                                                     │               │
│                                          Groq LLM Generation        │
│                                     (llama-3.3-70b-versatile)       │
│                                                     │               │
│                                          Answer + Sources           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Models, Tools & Techniques Tried

### 1. Document Extraction

| Tool / Approach | Tried? | Outcome |
|---|---|---|
| **PyMuPDF (`fitz`)** | ✅ Used | Excellent — extracted embedded Unicode (Gujarati / Devanagari) cleanly from the PDF's text layer without OCR |
| Tesseract OCR | ❌ Considered | Not needed — the PDF has an embedded text layer, so OCR would add latency without benefit |
| Surya OCR | ❌ Considered | Would be needed for truly scanned / handwritten pages, but overkill for this digital PDF |
| Google Cloud Vision | ❌ Considered | Best for handwritten script recognition, but requires API costs and network dependency |

### 2. Text Preprocessing & Structure Detection

| Technique | Details |
|---|---|
| **Regex-based cleanup** | Removes null/control characters (`\x00–\x1f`), collapses excess blank lines, strips trailing whitespace — all while preserving Gujarati/Devanagari Unicode codepoints |
| **Shloka marker detection** | Regex pattern `JJ\d+JJ` or `\|\|\d+\|\|` identifies verse boundaries in the text |
| **Chapter heading detection** | Matches `અધ્યાય` (Gujarati), `आध्याय` (Hindi/Sanskrit), or `chapter` (English) to track current chapter number |

### 3. Chunking Strategy

| Strategy | Tried? | Outcome |
|---|---|---|
| **Structure-based chunking** (split on verse markers) | ✅ Used (primary) | Best results — each chunk = one shloka + its commentary, preserving semantic integrity |
| **Sliding-window fallback** (800 chars, 150-char overlap) | ✅ Used (fallback) | Applied only when a single block exceeds `max_chars`; ensures no context is lost at boundaries |
| Fixed-size character chunking | ❌ Rejected | Breaks mid-verse or mid-sentence, destroying semantic coherence |
| Semantic chunking (embedding-based) | ❌ Rejected | Requires clean monolingual text; breaks down on mixed-script (Sanskrit + Gujarati) content |
| Sentence-level splitting (NLTK/spaCy) | ❌ Rejected | Gujarati/Sanskrit sentence tokenizers are unreliable; spaCy has no Gujarati model |

### 4. Embedding Models

| Model | Dimensions | Size | Languages | Tried? | Outcome |
|---|---|---|---|---|---|
| **`paraphrase-multilingual-MiniLM-L12-v2`** | 384 | ~118 MB | 50+ (incl. Gujarati, Hindi, Sanskrit) | ✅ Used | Strong cross-lingual alignment; English queries successfully retrieve Gujarati passages |
| `LaBSE` (Language-Agnostic BERT) | 768 | ~470 MB | 109 languages | ❌ Considered | More accurate for low-resource scripts but 4× slower on CPU and 4× larger; MiniLM-L12 was sufficient |
| `IndicBERT` / `MuRIL` | 768 | ~900 MB | 16 Indian languages | ❌ Considered | Better for Indic-specific tasks but much heavier; not justified for a 10-chapter demo |
| OpenAI `text-embedding-3-small` | 1536 | API-based | Multilingual | ❌ Rejected | Requires paid API; adds network dependency and cost |

### 5. Vector Store

| Option | Tried? | Outcome |
|---|---|---|
| **FAISS `IndexFlatIP`** (exact inner-product search) | ✅ Used | Simple, fast for small corpora (<1K chunks), no approximation error. Vectors are L2-normalised so inner product = cosine similarity |
| FAISS HNSW | ❌ Considered | Approximate but faster for 100K+ vectors; overkill for ~200 chunks |
| ChromaDB / Weaviate | ❌ Considered | Full vector DBs with persistence, but add infrastructure complexity for a demo |
| Pinecone | ❌ Rejected | Cloud-hosted, requires API key, adds latency |

### 6. LLM for Generation

| Model | Provider | Tried? | Outcome |
|---|---|---|---|
| **`llama-3.3-70b-versatile`** | Groq | ✅ Used | Excellent multilingual capability; correctly handles Gujarati context and responds in the user's language; fast inference via Groq's LPU |
| `llama-3.1-8b-instant` | Groq | ❌ Considered | Faster but weaker multilingual comprehension; struggles with Gujarati → English translation |
| GPT-4 / GPT-4o | OpenAI | ❌ Considered | Superior quality but paid API; Groq's free tier was sufficient |
| Gemma 2 | Google | ❌ Considered | Good multilingual but not available on Groq's free tier at the time |

### 7. UI Frameworks

| Framework | Tried? | Outcome |
|---|---|---|
| **Rich (CLI)** | ✅ Used | Beautiful terminal UI with panels, tables, and progress spinners |
| **Streamlit (Web UI)** | ✅ Used | Full web interface with chat bubbles, source cards, sidebar, and custom CSS theming |
| Gradio | ❌ Considered | Simpler but less customisable than Streamlit for chat-style interfaces |

---

## What Worked Well

### ✅ PyMuPDF for Multilingual Text Extraction
The reference PDF uses **embedded Unicode** for Gujarati and Devanagari scripts. PyMuPDF's `page.get_text("text")` extracted all characters perfectly — no OCR pipeline was needed, which saved significant complexity and processing time.

### ✅ Structure-Based Chunking
Splitting on verse markers (`JJ1JJ`, `JJ2JJ`, …) kept each **shloka paired with its Gujarati commentary** as a single semantic unit. This dramatically improved retrieval quality compared to naive fixed-size chunking, which would frequently split a verse from its explanation.

### ✅ Cross-Lingual Retrieval
The `paraphrase-multilingual-MiniLM-L12-v2` model proved remarkably effective at **cross-lingual alignment**. An English query like *"What does Arjuna say about his chariot?"* correctly retrieved the relevant Gujarati commentary passage, even though the query and document are in completely different scripts.

### ✅ Language-Matched Responses
The system prompt instructs the LLM to respond in the **same language as the user's question**. Combined with `llama-3.3-70b-versatile`'s strong multilingual capabilities, this works reliably — asking in Gujarati produces a Gujarati answer, asking in English produces English, etc.

### ✅ Groq's Inference Speed
Using Groq's LPU-accelerated API, generation latency is typically **1–3 seconds** for a full RAG answer, making the conversational experience feel responsive even with a 70B-parameter model.

### ✅ Dual Interface (CLI + Streamlit)
Having both a rich CLI (for quick testing and demos) and a polished Streamlit web UI (for presentation and user-facing deployment) provided flexibility across use cases.

---

## What Didn't Work / Challenges

### ❌ Windows Console Encoding
Python on Windows defaults to `cp1252` encoding, which cannot render Gujarati/Devanagari Unicode characters. The `rich` library would crash with `UnicodeEncodeError` when trying to print Gujarati text. **Fix:** Wrapped `sys.stdout` in a `TextIOWrapper` with `encoding="utf-8"` and `errors="replace"`.

### ❌ Sentence-Level Splitting for Gujarati
Standard NLP sentence tokenizers (NLTK, spaCy) have no Gujarati model. Attempting to split Gujarati text at sentence boundaries produced garbage results. **Fix:** Switched to structure-based chunking using the PDF's own verse markers.

### ❌ Semantic Chunking on Mixed-Script Text
Embedding-based semantic chunking (grouping sentences by similarity) assumes clean monolingual text. On mixed Sanskrit + Gujarati content, the similarity scores were noisy and unreliable. **Fix:** Abandoned semantic chunking in favour of deterministic structure-based splitting.

### ❌ No Persistent FAISS Index
The current system **rebuilds the entire index on every startup** — PDF extraction, chunking, embedding, and indexing all run from scratch. For a 10-chapter demo this takes ~30–60 seconds, but it would be impractical for larger corpora.

### ❌ Free-Tier Rate Limits
Groq's free tier has rate limits that can throttle responses during rapid sequential queries (e.g., running the 5-question demo). A 0.5-second delay was added between demo questions as a workaround.

---

## Handling Multilingual & Handwritten Content

### Multilingual Content Strategy

The Yatharthgeeta PDF contains three scripts: **Sanskrit (Devanagari)**, **Gujarati**, and **English**. Here's how each component of the pipeline handles this:

| Component | Multilingual Approach |
|---|---|
| **Text extraction** | PyMuPDF reads the PDF's embedded Unicode text layer directly — no script-specific handling needed since all three scripts are in Unicode |
| **Text cleaning** | Regex-based cleanup only removes ASCII control characters (`\x00–\x1f`), explicitly preserving all Unicode codepoints above `\x7f` (Gujarati range: `U+0A80–U+0AFF`, Devanagari: `U+0900–U+097F`) |
| **Structure detection** | Regex patterns match markers in **all three scripts**: `અધ્યાય` (Gujarati), `आध्याय` (Hindi), `chapter` (English) for chapter headings |
| **Chunking** | Script-agnostic — splits on structural markers (`JJ\d+JJ`), not on language boundaries |
| **Embedding** | `paraphrase-multilingual-MiniLM-L12-v2` natively supports all three languages and maps them into a **shared 384-dimensional vector space**, enabling cross-lingual retrieval |
| **LLM prompting** | System prompt explicitly instructs the model to detect the user's query language and respond in the same language, with original-script quotations preserved |

### Handwritten Content

The reference PDF is a **digitally typeset document** with an embedded text layer — it does not contain handwritten content. However, the architecture was designed with extensibility in mind:

- **If handwritten pages were present**, the pipeline would need an OCR stage between PDF loading and text extraction
- **Recommended OCR tools for Indic scripts:**
  - `surya-ocr` — open-source, supports Devanagari and Gujarati, runs locally
  - Google Cloud Vision API — best accuracy for handwritten Indic scripts, but requires API costs
  - `EasyOCR` — supports Hindi/Devanagari but Gujarati accuracy is limited
- **Integration point:** Replace `fitz.page.get_text("text")` with an OCR call for pages where the text layer is empty or unreliable

---

## Trade-offs & Design Decisions

### 1. MiniLM-L12 vs. LaBSE for Embeddings

| Factor | MiniLM-L12 (chosen) | LaBSE |
|---|---|---|
| Dimensions | 384 | 768 |
| Model size | ~118 MB | ~470 MB |
| CPU speed | ~3.5 sec/batch | ~14 sec/batch |
| Gujarati accuracy | Good | Better |
| **Decision** | Chosen for speed and size — adequate for a 10-chapter demo | Would be preferred for production with larger Indic corpora |

### 2. Exact vs. Approximate Vector Search

**Chose FAISS `IndexFlatIP` (exact search)** over approximate methods (HNSW, IVF). With only ~200 chunks, exact search completes in <1 ms. Approximate search adds complexity (tuning parameters, index training) with no measurable benefit at this scale.

### 3. Structure-Based vs. Semantic Chunking

**Chose structure-based chunking** because the PDF has clear structural markers (verse numbers). This is deterministic, language-agnostic, and preserves the natural verse ↔ commentary pairing. Semantic chunking was rejected because it performs poorly on mixed-script text.

### 4. Groq (Free Tier) vs. Paid APIs

**Chose Groq's free tier** with `llama-3.3-70b-versatile` for zero-cost deployment. Trade-off: rate limits during heavy usage. For production, a paid Groq plan or OpenAI GPT-4o would be more reliable.

### 5. In-Memory vs. Persistent Index

**Chose in-memory FAISS** for simplicity — no database setup, no file management. Trade-off: ~30–60 second startup time to rebuild the index. For production, `faiss.write_index()` / `faiss.read_index()` would eliminate this.

### 6. Top-5 Retrieval with No Re-ranking

**Chose a simple top-5 nearest-neighbour retrieval** without a cross-encoder re-ranker. For a small corpus, the bi-encoder's ranking is usually sufficient. A cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) would improve precision but add ~2 seconds of latency per query.

### 7. Conversation History Window (4 turns)

**Limited to last 4 turns** in the LLM context to stay within token limits on Groq's free tier while still enabling follow-up questions. Longer history would consume more tokens and risk hitting the model's context window.

---

## Improvements & Extensions with More Time

### Short-Term Improvements

| Improvement | Impact | Effort |
|---|---|---|
| **Persistent FAISS index** (`faiss.write_index` / `read_index`) | Eliminates 30–60s startup time | Low |
| **Environment-based API key management** | Already implemented (`os.environ.get`) — add `.env` file support with `python-dotenv` | Low |
| **Chapter-aware filtering** | Let users query within a specific chapter (pre-filter FAISS results by metadata) | Low |
| **Streaming responses** | Use Groq's streaming API for token-by-token display in the UI | Medium |

### Medium-Term Extensions

| Extension | Impact | Effort |
|---|---|---|
| **Hybrid search (BM25 + dense retrieval)** | BM25 catches exact term matches (e.g., specific Sanskrit words) that dense retrieval may miss; fuse scores with Reciprocal Rank Fusion | Medium |
| **Cross-encoder re-ranking** | Apply `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-20 → top-5 for improved precision | Medium |
| **OCR pipeline for scanned pages** | Integrate `surya-ocr` for handwritten/scanned Gujarati manuscripts | Medium |
| **Upgrade to LaBSE or MuRIL** | Better embedding quality for low-resource Indic scripts | Low |

### Long-Term Vision

| Extension | Impact | Effort |
|---|---|---|
| **Full Gita coverage (18 chapters)** | Extend from Ch. 1–10 to all 18 chapters + appendices | Medium |
| **Multi-document RAG** | Index multiple commentaries (Shankaracharya, Ramanujacharya, etc.) and compare interpretations | High |
| **Fine-tuned Indic embeddings** | Train a domain-specific embedding model on Gita corpus for higher retrieval accuracy | High |
| **Voice input/output** | Add speech-to-text (Whisper) and text-to-speech for an accessible, oral tradition-inspired experience | High |
| **Verse-level knowledge graph** | Build a graph connecting verses by theme, cross-references, and commentary relationships | High |

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
pip install -r requirements.txt
```

### Option 1: Streamlit Web UI (Recommended)

```bash
# Set your API key
export GROQ_API_KEY="gsk_YOUR_KEY_HERE"      # Linux/Mac
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"      # PowerShell

# Launch
streamlit run app.py
```

### Option 2: Rich CLI

```bash
# Interactive mode
python rag_geeta.py \
    --pdf "GEN AI TASK REF FILE Geeta-demo-1-10.pdf" \
    --api-key gsk_YOUR_KEY_HERE

# 5 demo questions (non-interactive)
python rag_geeta.py --pdf "..." --api-key "..." --demo
```

---

## CLI Commands

| Command | Description |
|---|---|
| Any question | LLM-based RAG answer with source citations |
| `/keyword <word>` | Exact substring search across all chunks |
| `/sources` | Toggle display of retrieved source passages |
| `/quit` | Exit the assistant |

---

## File Structure

```
rag_geeta_submission/
├── rag_geeta.py         # Core RAG pipeline (extraction, chunking, embedding, QA, CLI)
├── app.py               # Streamlit web UI with custom CSS theming
├── requirements.txt     # Python dependencies
├── README.md            # This report
└── GEN AI TASK REF FILE Geeta-demo-1-10.pdf   # Reference PDF (Chapters 1–10)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` (fitz) | PDF text extraction with Unicode support |
| `sentence-transformers` | Multilingual embedding model |
| `faiss-cpu` | Vector similarity search |
| `groq` | LLM API client (Groq Cloud) |
| `numpy` | Numerical operations |
| `rich` | Terminal UI (panels, tables, progress) |
| `streamlit` | Web UI framework |

---

<p align="center">
  <em>Built with ❤️ for multilingual scripture exploration</em>
</p>
]]>
