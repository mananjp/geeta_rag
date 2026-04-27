#!/usr/bin/env python3
"""
Regenerate the JSON cache for Yatharthgeeta PDF.

The PDF uses legacy non-Unicode fonts (GopikaTwo for Gujarati, Kautilya for
Devanagari). Direct text extraction gives garbled Latin-looking characters.

Strategy: Send each page's raw text to Groq LLM for decoding, with a very
strict prompt that forbids hallucination and demands 1:1 line correspondence.
"""

import os
import sys
import json
import re
import time
import fitz
from groq import Groq
from pathlib import Path

# Ensure standard output is utf-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

api_key = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key)

# Number of retries per page
MAX_RETRIES = 3


def clean_text_with_llm(text: str, page_num: int) -> str:
    """
    Send raw garbled text to LLM with a strict prompt that prevents
    hallucination and ensures faithful character-level decoding.
    """
    prompt = f"""You are a font-encoding expert. The text below was extracted from a Gujarati PDF that uses legacy non-Unicode fonts (GopikaTwo for Gujarati, Kautilya for Devanagari).

The extracted text looks like garbled Latin characters, but each character maps to a specific Gujarati or Devanagari Unicode character.

YOUR TASK:
- Convert EVERY character from the legacy encoding to proper Unicode Gujarati/Devanagari.
- Preserve ALL line breaks exactly as they appear.
- Preserve ALL numbers, verse markers (JJ1JJ, ||1||, etc.) exactly as-is.
- Preserve ALL Sanskrit shloka text that is already in Devanagari Unicode — do NOT change it.
- DO NOT add any text that is not in the original.
- DO NOT summarize or paraphrase.
- DO NOT repeat any sentence more than once unless it is repeated in the original.
- If you are uncertain about a character mapping, keep the original character.
- Output ONLY the decoded text. No explanations, no preamble, no "Here is the decoded text:" prefix.

CRITICAL RULES:
1. The output must have approximately the SAME number of lines as the input.
2. The output must have approximately the SAME length as the input.
3. NEVER generate repetitive text. If your output has the same sentence repeated more than twice, you are hallucinating — STOP and output the original text instead.
4. Each line of output must correspond to a line of input.

INPUT TEXT (Page {page_num}):
---
{text}
---

OUTPUT (decoded Unicode text only):"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise font-encoding decoder. You convert "
                            "legacy Indian font encodings to proper Unicode. You NEVER "
                            "hallucinate, summarize, or add extra text. If unsure, you "
                            "keep the original characters unchanged."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            decoded = response.choices[0].message.content.strip()

            # ── Anti-hallucination guard ──────────────────────────────
            # Check for excessive repetition (hallucination sign)
            lines = decoded.split('\n')
            if len(lines) > 4:
                # Count unique non-empty lines
                non_empty = [l.strip() for l in lines if l.strip()]
                if non_empty:
                    unique_ratio = len(set(non_empty)) / len(non_empty)
                    if unique_ratio < 0.3:
                        # More than 70% of lines are duplicates → hallucination
                        print(f"  ⚠ Page {page_num}: Detected hallucination "
                              f"(unique ratio={unique_ratio:.2f}), keeping raw text")
                        return text

            return decoded

        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1} failed for page {page_num}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)

    print(f"  ✗ All attempts failed for page {page_num}, keeping raw text")
    return text


def main():
    pdf_path = "GEN AI TASK REF FILE Geeta-demo-1-10.pdf"
    json_path = "GEN AI TASK REF FILE Geeta-demo-1-10.json"

    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    pages = []

    print(f"Processing {len(doc)} pages...")
    for i, page in enumerate(doc):
        raw = page.get_text("text")

        # Light cleanup
        raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        lines = [l.rstrip() for l in raw.splitlines()]
        clean = "\n".join(lines).strip()

        if clean.strip():
            print(f"\nDecoding page {i+1}/{len(doc)} with Groq LLM...")
            decoded_text = clean_text_with_llm(clean, i + 1)

            # Print a snippet to verify
            snippet = decoded_text[:120].replace('\n', ' ')
            print(f"  ✓ Snippet: {snippet}")

            pages.append({
                "page": i + 1,
                "raw_text": raw,
                "clean_text": decoded_text,
            })

            # Respect rate limits on free Groq tier
            time.sleep(1)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved cleaned text for {len(pages)} pages to {json_path}")


if __name__ == "__main__":
    main()
