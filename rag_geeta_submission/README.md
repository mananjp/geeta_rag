# Multilingual Knowledge Extraction & Exploration Assistant
### RAG over Yatharthgeeta (Gujarati/Sanskrit Bhagavad Gita, Chapters 1-10)

---

## Quick Start (< 5 min)

```bash
pip install -r requirements.txt

# Interactive mode
python rag_geeta.py \
    --pdf GEN-AI-TASK-REF-FILE-Geeta-demo-1-10.pdf \
    --api-key gsk_YOUR_GROQ_KEY

# 5 demo questions (non-interactive)
python rag_geeta.py --pdf ... --api-key ... --demo
```

---

## Architecture

```
PDF ──► PyMuPDF Extractor ──► Structure Detector
                                      │
                              Chunking Strategy
                           (Structure-based + Sliding Window)
                                      │
                          Multilingual Embedder
                     (paraphrase-multilingual-MiniLM-L12-v2)
                                      │
                           FAISS IndexFlatIP (cosine)
                                      │
                    ┌────────────────┴─────────────────┐
                    │    Groq LLM (llama-3.3-70b)      │
                    │    Retrieve → Augment → Generate  │
                    └───────────────────────────────────┘
```

---

## Module Breakdown

### Module 1 — Document Understanding + Preprocessing
| Step | Tool | Notes |
|---|---|---|
| Text extraction | `PyMuPDF (fitz)` | Handles embedded Unicode (Gujarati/Devanagari) |
| Cleaning | Regex | Remove control chars, normalise whitespace |
| Structure detection | Pattern matching | Finds Sanskrit shloka markers `JJ<n>JJ` + chapter headings |

> **Multilingual note**: The PDF uses Unicode encoding for Gujarati & Devanagari,
> so standard PDF text extraction works without a heavy OCR model.
> For truly scanned images, `google/cloud-vision` or `surya-ocr` would be needed.

---

### Module 2 — Chunking + Embedding Strategy

**Chunking method: Structure-based + Sliding-Window Fallback**

- **Primary**: Split on verse markers (`JJ1JJ`, `JJ2JJ` …) → each chunk = one shloka + its Gujarati commentary (semantic integrity preserved)
- **Fallback**: Blocks > 800 chars → sliding window (800 chars, 150-char overlap) to avoid boundary context loss
- Each chunk carries metadata: `page`, `chapter`, `verse_id`, `section_type`

**Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`**

| Property | Value |
|---|---|
| Languages | 50+ including Gujarati, Hindi, Sanskrit |
| Vector dim | 384 |
| Size | ~118 MB |
| Cross-lingual? | Yes — English query retrieves Gujarati passages |
| Why not LaBSE? | LaBSE (768-dim, 470 MB) is more accurate for low-resource scripts but 4× slower on CPU; MiniLM-L12 sufficient for demo |

**Vector store: FAISS IndexFlatIP (exact cosine search)**

- In-memory, no server needed
- Vectors are L2-normalised → inner product = cosine similarity

---

### Module 3 — QA / Exploration Tool

**Retrieval-Augmented Generation (RAG)**

1. Embed query with same multilingual model
2. Top-5 FAISS nearest-neighbour retrieval
3. Inject context into Groq prompt (llama-3.3-70b-versatile)
4. Conversation history (last 4 turns) for follow-up questions

**Bonus: keyword search** (`/keyword <term>`) — simple substring match across all chunks

---

## CLI Commands

| Command | Description |
|---|---|
| Any question | LLM-based RAG answer |
| `/keyword <word>` | Keyword search across all chunks |
| `/sources` | Toggle source passage display |
| `/quit` | Exit |

---

## What Worked Well
- PyMuPDF extracted Gujarati Unicode cleanly — no OCR needed for this PDF
- Structure-based chunking preserved shloka ↔ commentary pairs
- Multilingual MiniLM retrieved correct Gujarati passages from English queries
- Groq's llama-3.3-70b correctly translated Gujarati context to English answers

## What Didn't / Trade-offs
- For truly scanned/handwritten pages, `surya-ocr` or Google Vision would be needed
- FAISS flat index → exact but O(n) at query time; for 1M+ chunks use HNSW
- MiniLM-L12 may miss nuanced Sanskrit poetry; LaBSE or IndicBERT would be better
- No persistent index — rebuilds every run; add `faiss.write_index()` for production

## Improvements with More Time
1. **Scanned page support**: Integrate `surya-ocr` for handwritten Gujarati
2. **Persistent index**: Save/load FAISS + chunk metadata to disk
3. **Hybrid search**: BM25 + dense retrieval (BM25 for exact term matching, dense for semantics)
4. **Re-ranking**: Cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top-20 → top-5
5. **Streamlit UI**: Replace CLI with a web interface
6. **Chapter-aware filtering**: Let user query within a specific chapter
