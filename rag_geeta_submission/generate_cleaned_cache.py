import os
import sys
import json
import fitz
from groq import Groq
from pathlib import Path

# Ensure standard output is utf-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

api_key = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key)

def clean_text_with_llm(text, page_num):
    prompt = f"""
    You are an expert at decoding legacy Indian fonts. The following text was extracted from a PDF of the Bhagavad Gita (Yatharthgeeta) and uses legacy non-Unicode fonts (GopikaTwo for Gujarati, Kautilya for Devanagari). 
    
    Please convert the gibberish text back to standard Unicode Gujarati and Sanskrit/Devanagari.
    DO NOT translate the meaning, DO NOT summarize, and DO NOT add any extra conversational text. 
    Just output the EXACT same text but in proper Unicode Gujarati/Sanskrit characters.
    Preserve all line breaks, numbers, and verse markers like "JJ1JJ" or "||1||" exactly as they appear.
    
    Here is the gibberish text from Page {page_num}:
    {text}
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error on page {page_num}: {e}")
        return text

def main():
    pdf_path = "GEN AI TASK REF FILE Geeta-demo-1-10.pdf"
    json_path = "GEN AI TASK REF FILE Geeta-demo-1-10.json"
    
    doc = fitz.open(pdf_path)
    pages = []
    
    print(f"Processing {len(doc)} pages...")
    for i, page in enumerate(doc):
        raw = page.get_text("text")
        
        # light cleanup
        import re
        raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        lines = [l.rstrip() for l in raw.splitlines()]
        clean = "\n".join(lines).strip()
        
        if clean.strip():
            print(f"Decoding page {i+1}/{len(doc)} with Groq LLM...")
            decoded_text = clean_text_with_llm(clean, i+1)
            
            # Print a snippet to verify
            print(f"  Snippet: {decoded_text[:100].replace(chr(10), ' ')}")
            
            pages.append({
                "page": i + 1,
                "raw_text": raw,
                "clean_text": decoded_text
            })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
        
    print(f"Saved cleaned text to {json_path}")

if __name__ == "__main__":
    main()
