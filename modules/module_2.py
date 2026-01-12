# ============================================
# MODULE 2: PAPER SELECTION & PDF DOWNLOAD (FIXED)
# ============================================

import json
import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests
import fitz  # PyMuPDF

def load_search_results(filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load previously saved search results JSON."""
    if filepath:
        results_path = Path(filepath)
    else:
        results_dir = Path("data/search_results")
        if not results_dir.exists():
            print("Search results directory not found. Run Module 1 first.")
            return None

        json_files = sorted(
            (f for f in results_dir.iterdir() if f.suffix == ".json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not json_files:
            print("No search results found. Run Module 1 first.")
            return None

        results_path = json_files[0]
        print(f"Loading most recent search results: {results_path.name}")

    try:
        with results_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        papers = data.get("papers", [])
        print(f"Loaded {len(papers)} papers on '{data.get('topic', 'Unknown')}'")
        return data
    except Exception as exc:
        print(f"Error loading file {results_path}: {exc}")
        return None

def filter_papers_with_pdfs(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return subset of papers that contain a plausible PDF URL."""
    papers_with_pdf: List[Dict[str, Any]] = []
    for paper in papers:
        pdf_url = (paper.get("pdf_url") or "").strip()
        if not pdf_url:
            continue

        lower = pdf_url.lower()
        if lower.endswith(".pdf") or ".pdf?" in lower or "pdf" in lower:
            papers_with_pdf.append(paper)

    print(f"\nPDF Availability: {len(papers_with_pdf)} / {len(papers)} papers have PDFs.")
    return papers_with_pdf

def rank_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank papers by citation count (desc) then year (desc)."""
    valid = [p for p in papers if p.get("citationCount") is not None]
    ranked = sorted(
        valid,
        key=lambda x: (int(x.get("citationCount", 0)), int(x.get("year", 0) or 0)),
        reverse=True
    )
    return ranked

def select_top_papers(papers: List[Dict[str, Any]], count: int = 3) -> List[Dict[str, Any]]:
    """Select top N papers (by ranking) that have PDFs."""
    papers_with_pdf = filter_papers_with_pdfs(papers)
    ranked = rank_papers(papers_with_pdf)
    selected = ranked[:count]

    print(f"\nSelected top {len(selected)} papers for download:")
    for i, p in enumerate(selected, 1):
        print(f"{i}. {p.get('title', '')[:60]}... (Citations: {p.get('citationCount', 0)})")

    return selected

def download_pdf_with_verification(url: str, filename: str, max_retries: int = 2, chunk_size: int = 1024*32) -> bool:
    """Download a PDF URL with verification and retries."""
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    }

    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}: {url}")
            with session.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True) as resp:
                if resp.status_code >= 400:
                    print(f"    HTTP {resp.status_code} - skipping")
                    continue

                with open(target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)

                # Verification
                try:
                    with fitz.open(str(target)) as doc:
                        if len(doc) > 0:
                            return True
                except Exception:
                    pass
                
                # If verification failed
                target.unlink(missing_ok=True)
                
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(1)

    return False

def get_pdf_info(filepath: str) -> Dict[str, Any]:
    """Return basic metadata about a PDF file."""
    try:
        p = Path(filepath)
        if not p.exists():
            return {"is_valid": False}

        size_bytes = p.stat().st_size
        with fitz.open(str(p)) as doc:
            pages = len(doc)
        return {
            "pages": pages,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "is_valid": True
        }
    except Exception:
        return {"is_valid": False}

def download_selected_papers(selected_papers: List[Dict[str, Any]], output_dir: str = "downloads") -> List[Dict[str, Any]]:
    """Download selected papers."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for idx, paper in enumerate(selected_papers, start=1):
        title = paper.get("title", "untitled")
        safe_title = "".join(c for c in title if c.isalnum()).strip()[:20]
        short_hash = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
        filename = out_path / f"paper_{idx}_{safe_title}_{short_hash}.pdf"

        pdf_url = paper.get("pdf_url")
        success = False
        if pdf_url:
            success = download_pdf_with_verification(str(pdf_url), str(filename))

        paper["downloaded"] = success
        if success:
            paper["local_path"] = str(filename)
            paper["pdf_info"] = get_pdf_info(str(filename))
            downloaded.append(paper)
            print(f"    Success! Saved to {filename.name}")
        else:
            print("    Failed to download.")

    return downloaded

def save_download_report(downloaded_papers: List[Dict[str, Any]], topic: str) -> str:
    """Save download report."""
    report = {
        "topic": topic,
        "download_timestamp": datetime.now().isoformat(),
        "papers": downloaded_papers
    }
    
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path("data/reports") / f"download_report_{timestamp}.json"

    with report_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)
        
    return str(report_file)

# ====================
# MAIN EXECUTION (RENAMED TO MATCH APP.PY)
# ====================

def main_download_process(filepath: Optional[str] = None, count: int = 3) -> Optional[List[Dict[str, Any]]]:
    """
    Main entry for Module 2.
    Renamed from 'main_download' to 'main_download_process' to match app.py calls.
    """
    print("\n" + "=" * 72)
    print("MODULE 2: PAPER SELECTION & PDF DOWNLOAD")
    print("=" * 72)

    data = load_search_results(filepath)
    if not data:
        return None

    selected_papers = select_top_papers(data.get("papers", []), count=count)
    if not selected_papers:
        print("No papers with PDFs available for download.")
        return None

    downloaded = download_selected_papers(selected_papers)
    report_file = save_download_report(downloaded, data.get("topic", "unknown"))

    print(f"\nModule 2 complete! Downloaded {len(downloaded)} papers.")
    return downloaded

if __name__ == "__main__":
    main_download_process(count=3)
