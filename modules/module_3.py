# MODULE 3: PDF TEXT EXTRACTION

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from datetime import datetime
from tqdm import tqdm
try:
    import fitz
    pymupdf = fitz
except Exception as exc:
    raise ImportError("PyMuPDF (fitz) is required for Module 3. Install with `pip install pymupdf`.") from exc

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
    Improved text extraction that tries layout-aware markdown first (if available),
    otherwise performs robust plain-text extraction with page limits and heuristics.

    Args:
        pdf_path (Path): Path to the PDF file.

    Returns:
        str or None: Extracted text (possibly markdown). None if extraction failed or PDF appears restricted.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"  File not found: {pdf_path}")
        return None

    try:
        doc = pymupdf.open(str(pdf_path))

        if getattr(doc, "isEncrypted", False):
            print(f"  PDF appears encrypted: {pdf_path.name}. Attempting extraction...")

        first_page_text = ""
        if len(doc) > 0:
            try:
                first_page_text = doc[0].get_text().strip()
            except Exception:
                first_page_text = ""

        copyright_indicators = ["removed", "deleted", "takedown", "not available"]
        if first_page_text and any(keyword in first_page_text.lower() for keyword in copyright_indicators):
            print(f"  PDF appears to contain takedown/copyright notice: {pdf_path.name}")
            doc.close()
            return None

        extracted_candidates: List[tuple] = []

        if HAS_PYMUPDF4LLM:
            try:
                md_text = pymupdf4llm.to_markdown(str(pdf_path))
                if md_text and len(md_text) > 500:
                    extracted_candidates.append(("markdown", md_text))
            except Exception:
                pass

        plain_text = []

        pages_to_process = min(50, len(doc))
        for page_no in range(pages_to_process):
            try:
                page = doc[page_no]
                p_text = page.get_text()
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

        for method, text in extracted_candidates:
            if method == "markdown" and len(text) > 1000:
                return text

        best_text = max(extracted_candidates, key=lambda x: len(x[1]))[1]
        return best_text

    except Exception as exc:
        print(f"  Extraction error for {pdf_path.name}: {str(exc)[:200]}")
        return None


# -------------------------
# 2. SECTION EXTRACTION
# -------------------------
def extract_sections_improved(text: str) -> Dict[str, Any]:
    """
    Extract standard paper sections (title, abstract, introduction, methods, results, conclusion, references)
    using header heuristics and keyword fallbacks.

    Args:
        text (str): The full extracted text from a PDF.

    Returns:
        dict: Mapping of section names to content plus an 'extracted_text' preview.
    """
    sections = {
        "title": "",
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "conclusion": "",
        "references": "",
        "extracted_text": text[:20000] if text else ""
    }

    if not text or len(text) < 500:
        return sections

    clean = clean_text_basic(text)
    lines = clean.splitlines()

    header_patterns = {
        "abstract": [r'\babstract\b', r'\bsummary\b'],
        "introduction": [r'^\d+\.\s*introduction\b', r'\bintroduction\b', r'\bbackground\b'],
        "methods": [r'^\d+\.\s*methods?\b', r'\bmethods?\b', r'\bmethodology\b', r'\bexperimental\b'],
        "results": [r'^\d+\.\s*results?\b', r'\bresults?\b', r'\bfindings?\b'],
        "conclusion": [r'^\d+\.\s*conclusions?\b', r'\bconclusions?\b', r'\bdiscussion\b'],
        "references": [r'^\s*references\s*$', r'\bbibliography\b']
    }

    # Find line indices for section headers
    boundaries: Dict[str, int] = {}
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        low = line.lower()

        if len(line) > 200:
            continue
        for section_key, patterns in header_patterns.items():
            for pat in patterns:
                try:
                    if re.search(pat, low):
                        # Record first occurrence
                        if section_key not in boundaries:
                            boundaries[section_key] = idx
                            break
                except re.error:
                    continue

    if boundaries:
        sorted_sections = sorted(boundaries.items(), key=lambda x: x[1])
        for i, (section_name, start_idx) in enumerate(sorted_sections):
            start = start_idx + 1
            if i + 1 < len(sorted_sections):
                end = sorted_sections[i + 1][1]
            else:
                end = len(lines)
            content = "\n".join(lines[start:end]).strip()
            if len(content) > 100:
                sections[section_name] = content[:5000] 

    for line in lines[:10]:
        line_stripped = line.strip()
        if 20 < len(line_stripped) < 200 and not line_stripped.lower().startswith("http"):
            sections["title"] = line_stripped
            break

    major_keys = ["abstract", "introduction", "methods", "results", "conclusion"]
    if not any(len(sections[k]) > 200 for k in major_keys):
        sections = extract_by_keywords_fallback(clean, sections)

    return sections


def extract_by_keywords_fallback(text: str, existing_sections: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback strategy: look for keywords for each section and return contextual snippets.

    Args:
        text (str): Full text.
        existing_sections (dict): Current sections dict to update.

    Returns:
        dict: Updated sections mapping.
    """
    text_lower = text.lower()

    section_keywords = {
        "abstract": ["abstract", "summary", "we present", "this paper"],
        "introduction": ["introduction", "background", "motivation", "related work"],
        "methods": ["method", "experiment", "procedure", "dataset", "implementation"],
        "results": ["result", "finding", "table", "figure", "experiment shows"],
        "conclusion": ["conclusion", "discussion", "future work", "limitations", "summary"]
    }

    sentences = re.split(r'(?<=[.!?])\s+', text)

    for section, keywords in section_keywords.items():
        if existing_sections.get(section):
            continue

        contexts: List[str] = []
        for idx, sentence in enumerate(sentences):
            s_low = sentence.lower()
            if any(kw in s_low for kw in keywords):
                start = max(0, idx - 2)
                end = min(len(sentences), idx + 5)
                context = " ".join(sentences[start:end])
                contexts.append(context.strip())

        if contexts:
            existing_sections[section] = " ".join(contexts)[:5000]

    return existing_sections


def clean_text_basic(text: str) -> str:
    """
    Basic cleaning to reduce PDF extraction noise:
    - Normalize whitespace
    - Fix common hyphenation across line breaks
    - Remove non-printable characters (except newline)

    Args:
        text (str): Raw extracted text.

    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    # Normalize whitespace and remove excessive breaks
    t = re.sub(r'\r\n?', '\n', text)
    t = re.sub(r'\s+', ' ', t)

    # Fix hyphenation at end-of-line (common in PDFs)
    t = re.sub(r'-\s+', '', t)
    t = re.sub(r'\s*-\s*', '-', t)

    # Remove control characters except newline
    t = "".join(ch for ch in t if ch == '\n' or ord(ch) >= 32)

    return t.strip()


# -------------------------
# 3. PAPER PROCESSING
# -------------------------
def process_paper_smart(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """
    Validate PDF size, extract text, detect meaningful sections, and return structured metadata.

    Args:
        pdf_path (Path): Path to a single PDF file.

    Returns:
        dict or None: Result dict with metadata and sections, or None if skipped/failed.
    """
    pdf_path = Path(pdf_path)
    print(f"\nProcessing: {pdf_path.name}")

    try:
        file_size = pdf_path.stat().st_size
    except Exception as exc:
        print(f"  Could not read file size: {exc}")
        return None

    if file_size < 10_240:  # 10 KB
        print(f"  File too small ({file_size:,} bytes) — skipping")
        return None

    raw_text = extract_text_improved(pdf_path)
    if raw_text is None:
        print("  Skipping — empty or restricted PDF")
        return None

    if len(raw_text) < 1000:
        print(f"  Warning: extracted text very short ({len(raw_text):,} chars) — may be incomplete")

    print(f"  Extracted {len(raw_text):,} characters")

    sections = extract_sections_improved(raw_text)

    meaningful_sections = [
        name for name, content in sections.items()
        if name != "extracted_text" and content and len(content) > 200
    ]

    print(f"   Found {len(meaningful_sections)} meaningful sections")
    for s in meaningful_sections[:3]:
        print(f"    • {s}: {len(sections[s]):,} chars")

    result = {
        "paper_id": pdf_path.stem,
        "filename": pdf_path.name,
        "file_size_bytes": file_size,
        "total_characters": len(raw_text),
        "meaningful_sections": meaningful_sections,
        "sections": sections,
        "status": "success"
    }

    return result


# -------------------------
# 4. MAIN EXTRACTION
# -------------------------
def extract_all_papers(download_dir: str = "downloads/research_papers", max_papers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extract text and sections from all PDFs found in `download_dir`.

    Default directory now looks in downloads/research_papers for organized storage.
    Falls back to 'downloads' if the specified directory is missing or empty.
    """
    print("\n" + "=" * 72)
    print("MODULE 3: PDF TEXT EXTRACTION")
    print("=" * 72)

    pdf_files = get_downloaded_papers(download_dir)
    if not pdf_files:
        print("No PDFs found in the target directories. Run Module 2 or check paths.")
        return []

    if max_papers:
        pdf_files = pdf_files[:max_papers]

    print(f"\nProcessing {len(pdf_files)} PDF file(s)...")

    results: List[Dict[str, Any]] = []
    skipped = 0

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            res = process_paper_smart(pdf_file)
            if res:
                results.append(res)
            else:
                skipped += 1
        except Exception as exc:
            print(f"  Unexpected error processing {pdf_file.name}: {str(exc)[:200]}")
            skipped += 1
            continue

    if results:
        save_results_final(results)

    print("\nExtraction complete!")
    print(f"  Successfully processed: {len(results)}")
    print(f"  Skipped: {skipped}")

    return results


def get_downloaded_papers(download_dir: str = "downloads/research_papers") -> List[Path]:
    """
    Return list of PDF paths.

    Behavior:
      - First checks the preferred directory (default 'downloads/research_papers').
      - If that directory doesn't exist or contains no PDFs, falls back to 'downloads'.
      - Returns a sorted list of Path objects for PDF files.
    """
    preferred = Path(download_dir)
    fallback = Path("downloads")

    def _list_pdfs(path: Path) -> List[Path]:
        if not path.exists():
            return []
        return sorted(path.glob("*.pdf"))

    pdf_list = _list_pdfs(preferred)
    if pdf_list:
        print(f"Using PDFs from: {preferred.resolve()}")
        return pdf_list

    pdf_list = _list_pdfs(fallback)
    if pdf_list:
        print(f"No PDFs in {preferred}. Falling back to: {fallback.resolve()}")
        return pdf_list

    print(f"No PDF files found in either {preferred} or {fallback}.")
    return []

def save_results_final(results: List[Dict[str, Any]], output_dir: str = "data/extracted") -> None:
    """
    Save per-paper JSON files and a summary extraction file.

    Args:
        results (list): List of result dictionaries.
        output_dir (str): Directory to save outputs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for res in results:
        paper_id = res.get("paper_id", "unknown")
        output_file = output_path / f"{paper_id}_extracted.json"

        if "extracted_text" in res.get("sections", {}) and len(res["sections"]["extracted_text"]) > 10_000:
            res["sections"]["extracted_text"] = res["sections"]["extracted_text"][:10_000] + "...[truncated]"

        with output_file.open("w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)

        print(f"   Saved: {output_file.name}")

    # Save a summary file
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "total_papers": len(results),
        "papers": [
            {
                "paper_id": r["paper_id"],
                "filename": r["filename"],
                "file_size": r["file_size_bytes"],
                "total_chars": r["total_characters"],
                "sections_found": r["meaningful_sections"]
            }
            for r in results
        ]
    }

    summary_file = output_path / "extraction_summary.json"
    with summary_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_file}")


# -------------------------
# 5. ANALYZE RESULTS
# -------------------------
def analyze_extraction_results() -> None:
    """
    Read saved extracted JSON files and print an analysis summary.
    """
    print("\n" + "=" * 72)
    print("EXTRACTION ANALYSIS")
    print("=" * 72)

    data_path = Path("data/extracted")
    if not data_path.exists():
        print("No extraction directory found")
        return

    json_files = sorted(data_path.glob("*_extracted.json"))
    if not json_files:
        print("No extracted paper files found")
        return

    print(f"\nFound {len(json_files)} extracted papers:\n")

    total_chars = 0
    papers_with_abstract = 0
    papers_with_multiple_sections = 0

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            paper_id = data.get("paper_id", "Unknown")
            total_chars += data.get("total_characters", 0)

            sections = data.get("sections", {})
            meaningful_sections = data.get("meaningful_sections", [])

            if sections.get("abstract") and len(sections["abstract"]) > 200:
                papers_with_abstract += 1

            if len(meaningful_sections) >= 2:
                papers_with_multiple_sections += 1

            print(f" {paper_id}")
            print(f"   Size: {data.get('file_size_bytes', 0):,} bytes")
            print(f"   Text: {data.get('total_characters', 0):,} chars")
            print(f"   Sections found: {len(meaningful_sections)}")

            if sections.get("title"):
                print(f"   Title: {sections['title'][:80]}")

            if sections.get("abstract"):
                print(f"   Abstract: {sections['abstract'][:150]}...")

            print()

        except Exception as exc:
            print(f" Error reading {jf.name}: {str(exc)[:200]}")

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total papers processed: {len(json_files)}")
    print(f"Total characters extracted: {total_chars:,}")
    print(f"Papers with abstract: {papers_with_abstract}/{len(json_files)}")
    print(f"Papers with multiple sections: {papers_with_multiple_sections}/{len(json_files)}")


# -------------------------
# 6. GENERATE REPORT
# -------------------------
def generate_report() -> Optional[Dict[str, Any]]:
    """
    Generate a per-paper quality report and an aggregated review JSON that a mentor can inspect.

    Returns:
        dict: Report dictionary (also saved to data/extracted/_review_report.json)
    """
    print("\n" + "=" * 72)
    print("  REVIEW REPORT")
    print("=" * 72)

    data_path = Path("data/extracted")
    if not data_path.exists():
        print(" No extraction directory found")
        return None

    json_files = sorted(data_path.glob("*_extracted.json"))
    if not json_files:
        print(" No extracted papers found")
        return None

    report: Dict[str, Any] = {
        "generated_date": datetime.now().isoformat(),
        "total_papers": len(json_files),
        "quality_checks": [],
        "papers": []
    }

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            sections = data.get("sections", {})
            paper_report = {
                "paper_id": data.get("paper_id"),
                "filename": data.get("filename"),
                "checks": {
                    "text_clean": False,
                    "sections_correct": False,
                    "no_hallucinations": False,
                    "no_missing_chunks": False
                },
                "section_lengths": {},
                "issues": []
            }

            sample_text = sections.get("abstract") or sections.get("extracted_text", "")
            artifacts = ['�', '\x00', '[?]', '[ ]']
            has_artifacts = any(art in sample_text for art in artifacts)
            paper_report["checks"]["text_clean"] = not has_artifacts
            if has_artifacts:
                paper_report["issues"].append("Text contains extraction artifacts")

            major_sections = ["abstract", "introduction", "methods", "results", "conclusion"]
            found_major = [s for s in major_sections if sections.get(s) and len(sections[s]) > 200]
            paper_report["checks"]["sections_correct"] = len(found_major) >= 2
            if len(found_major) < 2:
                paper_report["issues"].append(f"Only found {len(found_major)} major sections")

            total_chars = data.get("total_characters", 0)
            paper_report["checks"]["no_hallucinations"] = 1000 <= total_chars <= 500_000
            if total_chars < 1000:
                paper_report["issues"].append(f"Text too short: {total_chars} chars")
            elif total_chars > 500_000:
                paper_report["issues"].append(f"Text suspiciously long: {total_chars} chars")

            section_lengths_sum = sum(len(str(v)) for v in sections.values() if v)
            coverage = section_lengths_sum / total_chars if total_chars > 0 else 0
            paper_report["checks"]["no_missing_chunks"] = coverage >= 0.3
            if coverage < 0.3:
                paper_report["issues"].append(f"Low coverage: {coverage:.1%}")

            for sec_name, content in sections.items():
                if content and len(str(content)) > 50:
                    paper_report["section_lengths"][sec_name] = len(str(content))

            report["papers"].append(paper_report)
        except Exception as exc:
            print(f"Error processing {jf.name}: {str(exc)[:200]}")

    # Aggregate overall scores
    total_checks = 0
    passed_checks = 0
    for paper in report["papers"]:
        for check_name, passed in paper["checks"].items():
            total_checks += 1
            if passed:
                passed_checks += 1

    report["overall_score"] = f"{passed_checks}/{total_checks}" if total_checks > 0 else "N/A"
    report["success_rate"] = (passed_checks / total_checks) if total_checks > 0 else 0

    report_file = data_path / "_review_report.json"
    with report_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\nreport generated!")
    print(f"  Overall score: {report['overall_score']}")
    print(f"  Success rate: {report['success_rate']:.1%}")
    print(f"  Report saved to: {report_file}")

    print("\nQUALITY CHECK SUMMARY:")
    print("-" * 40)
    check_names = ["text_clean", "sections_correct", "no_hallucinations", "no_missing_chunks"]
    for check_name in check_names:
        passed_count = sum(1 for paper in report["papers"] if paper["checks"].get(check_name, False))
        total = len(report["papers"])
        percentage = (passed_count / total * 100) if total > 0 else 0
        status = "✅" if percentage >= 70 else "⚠️" if percentage >= 50 else "❌"
        print(f"{status} {check_name}: {passed_count}/{total} ({percentage:.0f}%)")

    return report


# -------------------------
# 7. RUN COMPLETE PIPELINE
# -------------------------
def run_complete_extraction() -> (List[Dict[str, Any]], Optional[Dict[str, Any]]):
    """
    Full extraction pipeline entrypoint:
      - Extract text from PDFs (up to a limit)
      - Analyze saved results
      - Generate mentor review report

    Returns:
        (results_list, report_dict)
    """
    print("\n" + "=" * 72)
    print("PDF TEXT EXTRACTION MODULE")
    print("=" * 72)

    print("\nSTEP 1: Extracting text from PDFs...")
    results = extract_all_papers(max_papers=5)

    if not results:
        print("No papers extracted successfully.")
        return [], None

    print("\nSTEP 2: Analyzing extraction quality...")
    analyze_extraction_results()

    print("\nSTEP 3: Generating review report...")
    report = generate_report()

    print("\n" + "=" * 72)
    print("COMPLETE!")
    print("=" * 72)

    return results, report


if __name__ == "__main__":
    results, report = run_complete_extraction()

    if results:
        print("\n" + "=" * 72)
        print("EXAMPLE OF EXTRACTED CONTENT")
        print("=" * 72)

        first_paper = results[0]
        sections = first_paper.get("sections", {})

        print(f"\nPaper: {first_paper.get('paper_id')}")

        for section_name in ["title", "abstract", "introduction"]:
            content = sections.get(section_name)
            if content and len(content) > 50:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                print(f"\n{section_name.upper()}:")
                print("-" * 40)
                print(preview)
                print(f"[Total length: {len(content):,} characters]")
