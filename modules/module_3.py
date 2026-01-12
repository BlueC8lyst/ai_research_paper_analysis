# ============================================
# MODULE 3: PDF TEXT EXTRACTION (FIXED)
# ============================================

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
    pymupdf = fitz
except Exception as exc:
    raise ImportError("PyMuPDF (fitz) is required. Install with `pip install pymupdf`.") from exc

try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except Exception:
    pymupdf4llm = None
    HAS_PYMUPDF4LLM = False


# -------------------------
# 1. TEXT EXTRACTION
# -------------------------
def extract_text_improved(pdf_path: Path) -> Optional[str]:
    """
    Improved text extraction using markdown layout analysis if available.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"  File not found: {pdf_path}")
        return None

    try:
        doc = pymupdf.open(str(pdf_path))

        # Basic validation
        if len(doc) == 0:
            doc.close()
            return None

        # Check for copyright placeholders
        try:
            first_page_text = doc[0].get_text().strip().lower()
            copyright_indicators = ["removed", "deleted", "takedown", "not available"]
            if any(k in first_page_text for k in copyright_indicators) and len(first_page_text) < 500:
                print(f"  PDF appears to contain takedown notice: {pdf_path.name}")
                doc.close()
                return None
        except:
            pass

        extracted_candidates = []

        # Strategy 1: Markdown (Better structure)
        if HAS_PYMUPDF4LLM:
            try:
                md_text = pymupdf4llm.to_markdown(str(pdf_path))
                if md_text and len(md_text) > 500:
                    extracted_candidates.append(("markdown", md_text))
            except Exception:
                pass

        # Strategy 2: Plain Text (Fallback)
        plain_text = []
        pages_to_process = min(50, len(doc))
        for page_no in range(pages_to_process):
            try:
                p_text = doc[page_no].get_text()
                if p_text:
                    plain_text.append(p_text)
            except Exception:
                continue
        
        full_text = "\n".join(plain_text).strip()
        if full_text and len(full_text) > 500:
            extracted_candidates.append(("regular", full_text))

        doc.close()

        if not extracted_candidates:
            return None

        # Prefer markdown if available
        for method, text in extracted_candidates:
            if method == "markdown" and len(text) > 1000:
                return text

        # Otherwise longest text
        return max(extracted_candidates, key=lambda x: len(x[1]))[1]

    except Exception as exc:
        print(f"  Extraction error for {pdf_path.name}: {exc}")
        return None


# -------------------------
# 2. SECTION EXTRACTION
# -------------------------
def extract_sections_improved(text: str) -> Dict[str, Any]:
    """
    Extract standard sections using Regex headers.
    """
    sections = {
        "title": "", "abstract": "", "introduction": "",
        "methods": "", "results": "", "conclusion": "",
        "references": "", "extracted_text": text[:20000] if text else ""
    }

    if not text or len(text) < 500:
        return sections

    lines = text.splitlines()
    header_patterns = {
        "abstract": [r'\babstract\b', r'\bsummary\b'],
        "introduction": [r'^\d+\.\s*introduction\b', r'\bintroduction\b', r'\bbackground\b'],
        "methods": [r'^\d+\.\s*methods?\b', r'\bmethods?\b', r'\bmethodology\b', r'\bexperimental\b'],
        "results": [r'^\d+\.\s*results?\b', r'\bresults?\b', r'\bfindings?\b'],
        "conclusion": [r'^\d+\.\s*conclusions?\b', r'\bconclusions?\b', r'\bdiscussion\b'],
        "references": [r'^\s*references\s*$', r'\bbibliography\b']
    }

    boundaries = {}
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip().lower()
        if len(line) > 100: continue
        
        for section_key, patterns in header_patterns.items():
            for pat in patterns:
                if re.search(pat, line):
                    if section_key not in boundaries:
                        boundaries[section_key] = idx
                        break

    if boundaries:
        sorted_sections = sorted(boundaries.items(), key=lambda x: x[1])
        for i, (section_name, start_idx) in enumerate(sorted_sections):
            start = start_idx + 1
            end = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(lines)
            content = "\n".join(lines[start:end]).strip()
            if len(content) > 100:
                sections[section_name] = content[:5000] 

    # Title Heuristic
    for line in lines[:10]:
        clean = line.strip()
        if 20 < len(clean) < 200 and not clean.lower().startswith("http"):
            sections["title"] = clean
            break

    return sections


# -------------------------
# 3. MAIN PIPELINE (RENAMED TO MATCH APP.PY)
# -------------------------

def process_paper_smart(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """Process a single paper."""
    print(f"\nProcessing: {pdf_path.name}")
    
    try:
        raw_text = extract_text_improved(pdf_path)
    except Exception:
        return None

    if not raw_text: 
        return None

    sections = extract_sections_improved(raw_text)
    
    meaningful = [k for k, v in sections.items() if k != "extracted_text" and len(v) > 200]
    
    return {
        "paper_id": pdf_path.stem,
        "filename": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "total_characters": len(raw_text),
        "meaningful_sections": meaningful,
        "sections": sections,
        "status": "success"
    }

def get_downloaded_papers(download_dir: str = "downloads") -> List[Path]:
    """Find PDFs in downloads directory."""
    path = Path(download_dir)
    return sorted(path.glob("*.pdf")) if path.exists() else []

def save_results_final(results: List[Dict[str, Any]], output_dir: str = "data/extracted") -> None:
    """Save extracted data to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for res in results:
        paper_id = res.get("paper_id", "unknown")
        # Truncate huge text fields for saving
        if "extracted_text" in res["sections"] and len(res["sections"]["extracted_text"]) > 15000:
             res["sections"]["extracted_text"] = res["sections"]["extracted_text"][:15000] + "...[truncated]"
             
        with (out_path / f"{paper_id}_extracted.json").open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

def run_extraction_pipeline(download_dir: str = "downloads", max_papers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Main entry point used by Streamlit (app.py).
    Renamed from 'extract_all_papers' or 'run_complete_extraction' to match app.py call.
    """
    print("\n" + "=" * 72)
    print("MODULE 3: PDF TEXT EXTRACTION")
    print("=" * 72)

    pdf_files = get_downloaded_papers(download_dir)
    if not pdf_files:
        print("No PDFs found. Please run Module 2 first.")
        return []

    if max_papers:
        pdf_files = pdf_files[:max_papers]

    print(f"\nProcessing {len(pdf_files)} PDF file(s)...")
    results = []

    for pdf_file in tqdm(pdf_files, desc="Extracting Text"):
        res = process_paper_smart(pdf_file)
        if res:
            results.append(res)

    if results:
        save_results_final(results)
    
    print(f"\nExtraction complete! Processed {len(results)} papers.")
    return results

if __name__ == "__main__":
    run_extraction_pipeline(max_papers=3)
