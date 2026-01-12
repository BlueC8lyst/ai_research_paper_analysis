# !pip install PyMuPDF requests -q

import json
import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests
import fitz 

def load_search_results(filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load previously saved search results JSON.

    If filepath is None, find the newest JSON in data/search_results.

    Args:
        filepath: Optional path to a specific results file.

    Returns:
        Parsed JSON dict or None if loading fails.
    """
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
    """
    Return subset of papers that contain a plausible PDF URL.

    Args:
        papers: list of paper dicts (from Module 1 save format).

    Returns:
        List of papers that likely have PDF links.
    """
    papers_with_pdf: List[Dict[str, Any]] = []
    for paper in papers:
        pdf_url = (paper.get("pdf_url") or "").strip()
        if not pdf_url:
            continue

        lower = pdf_url.lower()
        if lower.endswith(".pdf") or ".pdf?" in lower or "pdf" in lower:
            papers_with_pdf.append(paper)

    print(f"\nPDF Availability:")
    print(f"  • Total papers checked: {len(papers)}")
    print(f"  • Papers with PDF URLs: {len(papers_with_pdf)}")

    return papers_with_pdf

def rank_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank papers by citation count (desc) then year (desc).

    Args:
        papers: list of paper dicts.

    Returns:
        Sorted list (highest ranked first).
    """
    valid = [p for p in papers if p.get("citationCount") is not None]
    ranked = sorted(
        valid,
        key=lambda x: (int(x.get("citationCount", 0)), int(x.get("year", 0) or 0)),
        reverse=True
    )
    return ranked


def select_top_papers(papers: List[Dict[str, Any]], count: int = 3) -> List[Dict[str, Any]]:
    """
    Select top N papers (by ranking) that have PDFs.

    Args:
        papers: All papers (from search results).
        count: Number of papers to select.

    Returns:
        Selected papers list.
    """
    papers_with_pdf = filter_papers_with_pdfs(papers)
    ranked = rank_papers(papers_with_pdf)
    selected = ranked[:count]

    print(f"\nSelected top {len(selected)} papers for download:")
    for i, p in enumerate(selected, 1):
        title = p.get("title", "")[:70]
        print(f"\n{i}. {title}{'...' if len(p.get('title', '')) > 70 else ''}")
        print(f"   Citations: {p.get('citationCount', 0)}")
        print(f"   Year: {p.get('year', 'N/A')}")
        authors = ", ".join(p.get("authors", [])[:2])
        print(f"   Authors: {authors}")

    return selected


def _is_response_pdf(response: requests.Response) -> bool:
    """
    Heuristic check if a requests response looks like a PDF.

    Args:
        response: requests.Response object.

    Returns:
        True if response content-type or initial bytes indicate PDF.
    """
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type:
        return True
    # Check first bytes (PDF files start with '%PDF')
    start = response.content[:4]
    return start == b"%PDF"



def download_pdf_with_verification(url: str, filename: str, max_retries: int = 2, chunk_size: int = 1024*32) -> bool:
    """
    Download a PDF URL to `filename` with verification.

    - Streams to disk using iter_content (no resp.raw.seek usage).
    - Checks initial bytes for '%PDF' and content-type hints.
    - Retries on transient errors with simple backoff.
    - Returns True on success (valid PDF), False otherwise.
    """
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
                status = resp.status_code

                if status == 403:
                    print(f"    HTTP 403 - Forbidden (site blocked this request).")
                    return False

                if status >= 400:
                    print(f"    HTTP {status} - skipping this attempt")
                    time.sleep(1 * attempt)
                    continue

                # Basic content-type check
                content_type = resp.headers.get("content-type", "").lower()
                looks_like_pdf_by_ct = "pdf" in content_type

                first_bytes = b""
                bytes_written = 0

                with open(target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue

                        if bytes_written < 8:
                            need = 8 - len(first_bytes)
                            first_bytes += chunk[:need]
                        fh.write(chunk)
                        bytes_written += len(chunk)

                if first_bytes.startswith(b"%PDF") or looks_like_pdf_by_ct:
                    # Final verification using PyMuPDF (fitz)
                    try:
                        import fitz
                        with fitz.open(str(target)) as doc:
                            if len(doc) > 0:
                                size = target.stat().st_size
                                print(f"    Downloaded: {size:,} bytes -> {target.name}")
                                return True
                            else:
                                print("    Downloaded file opened but has zero pages.")
                                target.unlink(missing_ok=True)
                                continue
                    except Exception as e:
                        print(f"    PDF verification failed (PyMuPDF): {e}")
                        target.unlink(missing_ok=True)
                        continue
                else:
                    print("    File does not look like a PDF (missing %PDF signature and content-type not PDF).")
                    target.unlink(missing_ok=True)
                    continue

        except requests.exceptions.Timeout:
            print("    Timeout during download attempt.")
            time.sleep(1 * attempt)
        except requests.exceptions.RequestException as e:
            # handle other network errors
            print(f"    Network error during download: {e}")
            time.sleep(1 * attempt)
        except Exception as e:
            print(f"    Unexpected error during download: {e}")
            time.sleep(1 * attempt)

    # All attempts failed
    return False

def get_pdf_info(filepath: str) -> Dict[str, Any]:
    """
    Return basic metadata about a PDF file.

    Args:
        filepath: path to PDF.

    Returns:
        Dict with pages, size_bytes, size_mb, is_valid.
    """
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
    """
    Download selected papers to output directory and collect metadata.

    Args:
        selected_papers: list of paper dicts with keys 'title' and 'pdf_url'.
        output_dir: directory to place downloaded PDFs.

    Returns:
        List of paper dicts augmented with download metadata.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    downloaded: List[Dict[str, Any]] = []
    print(f"\nStarting PDF downloads to: {out_path}/")
    print("-" * 60)

    for idx, paper in enumerate(selected_papers, start=1):
        title = paper.get("title", "untitled")
        print(f"\n[{idx}/{len(selected_papers)}] Downloading: {title[:60]}{'...' if len(title) > 60 else ''}")

        # Create safe filename derived from title and a short hash to avoid collisions
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_title = safe_title[:50] if len(safe_title) > 50 else safe_title
        short_hash = hashlib.md5(safe_title.encode("utf-8")).hexdigest()[:8]
        filename = out_path / f"paper_{idx}_{short_hash}.pdf"

        pdf_url = paper.get("pdf_url")
        success = False
        if pdf_url:
            success = download_pdf_with_verification(str(pdf_url), str(filename))

        if success:
            pdf_info = get_pdf_info(str(filename))
            paper["downloaded"] = True
            paper["local_path"] = str(filename)
            paper["download_time"] = datetime.now().isoformat()
            paper["pdf_info"] = pdf_info
            downloaded.append(paper)
            print(f"    Success! {pdf_info.get('pages', 'N/A')} pages, {pdf_info.get('size_mb', 'N/A')} MB")
        else:
            paper["downloaded"] = False
            print("    Failed to download.")

    return downloaded


def save_download_report(downloaded_papers: List[Dict[str, Any]], topic: str, output_dir: str = "downloads") -> str:
    """
    Save a detailed download report and a simple list of downloaded files.

    Args:
        downloaded_papers: list returned from download_selected_papers.
        topic: research topic string for report context.
        output_dir: directory where downloaded files are located.

    Returns:
        Path to the saved JSON report.
    """
    report = {
        "topic": topic,
        "download_timestamp": datetime.now().isoformat(),
        "total_selected": len(downloaded_papers),
        "successful_downloads": sum(1 for p in downloaded_papers if p.get("downloaded")),
        "failed_downloads": sum(1 for p in downloaded_papers if not p.get("downloaded")),
        "papers": downloaded_papers
    }

    Path("data/reports").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path("data/reports") / f"download_report_{timestamp}.json"

    with report_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, ensure_ascii=False)

    print(f"\nDownload report saved to: {report_file}")

    download_list = []
    for p in downloaded_papers:
        if p.get("downloaded"):
            pdf_info = p.get("pdf_info", {})
            download_list.append({
                "title": p.get("title"),
                "local_file": p.get("local_path"),
                "size_mb": pdf_info.get("size_mb"),
                "pages": pdf_info.get("pages")
            })

    list_file = Path(output_dir) / "downloaded_papers_list.json"
    with list_file.open("w", encoding="utf-8") as fh:
        json.dump(download_list, fh, indent=4, ensure_ascii=False)

    return str(report_file)


def verify_downloads(output_dir: str = "downloads") -> int:
    """
    Verify all PDFs in the output directory and print a summary.

    Args:
        output_dir: directory containing downloaded PDFs.

    Returns:
        Number of valid PDF files.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        print(f"Directory '{output_dir}' does not exist!")
        return 0

    pdf_files = sorted(out_path.glob("*.pdf"))
    print("\n" + "=" * 60)
    print("VERIFICATION OF DOWNLOADS")
    print("=" * 60)
    print(f"\nDirectory: {out_path.resolve()}")
    print(f"PDF files found: {len(pdf_files)}")

    total_size = 0
    valid_count = 0

    if pdf_files:
        print("\nFile Details:")
        print("-" * 60)

        for pdf in pdf_files:
            size = pdf.stat().st_size
            total_size += size
            info = get_pdf_info(str(pdf))
            is_valid = info.get("is_valid", False)
            if is_valid:
                valid_count += 1
                try:
                    with fitz.open(str(pdf)) as doc:
                        pages = len(doc)
                except Exception:
                    pages = "N/A"
                print(f" {pdf.name}")
                print(f"   Size: {size:,} bytes ({size / (1024 * 1024):.2f} MB)")
                print(f"   Pages: {pages}")
            else:
                print(f" {pdf.name} - INVALID PDF")
                print(f"   Size: {size:,} bytes")

    print(f"\nSummary:")
    print(f"  • Total PDF files: {len(pdf_files)}")
    print(f"  • Valid PDFs: {valid_count}")
    print(f"  • Total size: {total_size / (1024 * 1024):.2f} MB")

    return valid_count



def main_download(filepath: Optional[str] = None, download_count: int = 3) -> Optional[List[Dict[str, Any]]]:
    """
    Main entry for Module 2: load search results, select top papers, download them, save a report.

    Args:lepath: optional specific search results JSON file to load.
        download_count: number of top papers to download.

    Returns:
        List of downloaded paper metadata, or None if pipeline could not proceed.
    """
    print("\n" + "=" * 72)
    print("MODULE 2: PAPER SELECTION & PDF DOWNLOAD")
    print("=" * 72)

    data = load_search_results(filepath)
    if not data:
        return None

    selected_papers = select_top_papers(data.get("papers", []), count=download_count)
    if not selected_papers:
        print("No papers with PDFs available for download.")
        return None

    downloaded = download_selected_papers(selected_papers)
    report_file = save_download_report(downloaded, data.get("topic", "unknown"))
    verify_downloads()

    print(f"\nModule 2 complete!")
    print(f"  Downloaded papers are in: downloads/")
    print(f"  Report saved to: {report_file}")

    return downloaded


if __name__ == "__main__":
    main_download(download_count=3)


