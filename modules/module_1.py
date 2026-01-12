# ============================================
# MODULE 1: TOPIC INPUT & PAPER SEARCH (FIXED)
# ============================================

# !pip install semanticscholar python-dotenv requests pandas pymupdf-layout -q

import json
import os
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from semanticscholar import SemanticScholar

def setup_api_key() -> SemanticScholar:
    """
    Initialize SemanticScholar with a longer timeout to prevent errors.
    """
    load_dotenv()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    # Set timeout to 60 seconds (default is usually 10)
    if not api_key:
        print(" [Config] No API Key found. Using limited rate (Timeout=60s).")
        scholar_client = SemanticScholar(timeout=60)
    else:
        print(" [Config] API Key loaded. (Timeout=60s).")
        scholar_client = SemanticScholar(api_key=api_key, timeout=60)

    return scholar_client


def search_papers(topic: str, limit: int = 20) -> Optional[Dict[str, Any]]:
    """
    Search with automatic retries for timeouts.
    """
    if not topic or not topic.strip():
        raise ValueError("search_papers requires a non-empty topic string.")

    print(f"\nSearching for papers on: '{topic}' (limit={limit})")
    
    scholar_client = setup_api_key()
    
    # Retry configuration
    max_retries = 3
    retry_delay = 5  # seconds

    results = None
    last_error = None

    # RETRY LOOP
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f" ... Attempt {attempt}/{max_retries}: Retrying search...")
            
            results = scholar_client.search_paper(
                query=topic,
                limit=limit,
                fields=[
                    "paperId", "title", "abstract", "year", "authors",
                    "citationCount", "openAccessPdf", "url", "venue"
                ]
            )
            # If successful, break the loop
            break
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # If it's a timeout or connection error, wait and retry
            if "time" in error_msg or "connect" in error_msg or "500" in error_msg:
                print(f" [!] Connection issue: {e}")
                time.sleep(retry_delay)
            else:
                # If it's a logic error (like invalid query), stop immediately
                break

    # If we failed after all retries
    if not results:
        print(f"\n [Error] Search failed after {max_retries} attempts.")
        print(f" Last error: {last_error}")
        return None

    # Process Results
    try:
        papers: List[Dict[str, Any]] = []

        for paper in results:
            raw_authors = getattr(paper, "authors", []) or []
            authors: List[str] = []
            for a in raw_authors:
                if hasattr(a, "name"):
                    authors.append(getattr(a, "name"))
                elif isinstance(a, dict) and "name" in a:
                    authors.append(a["name"])
                else:
                    authors.append(str(a))

            open_access_pdf = getattr(paper, "openAccessPdf", None)
            pdf_url = None
            has_pdf = False
            if open_access_pdf:
                if isinstance(open_access_pdf, dict):
                    pdf_url = open_access_pdf.get("url")
                else:
                    pdf_url = getattr(open_access_pdf, "get", lambda x: None)("url")
                has_pdf = bool(pdf_url)

            paper_entry = {
                "title": getattr(paper, "title", "") or "No title",
                "authors": authors,
                "year": getattr(paper, "year", None),
                "paperId": getattr(paper, "paperId", None),
                "abstract": (getattr(paper, "abstract", "") or "")[:300] + "...",
                "citationCount": getattr(paper, "citationCount", 0),
                "venue": getattr(paper, "venue", None),
                "url": getattr(paper, "url", None),
                "pdf_url": pdf_url,
                "has_pdf": has_pdf
            }
            papers.append(paper_entry)

        papers_with_pdf = sum(1 for p in papers if p["has_pdf"])

        print("Search complete!")
        print(f"  Total papers returned: {len(papers)}")
        print(f"  Papers with PDF available: {papers_with_pdf}")

        return {
            "topic": topic,
            "search_timestamp": datetime.now().isoformat(),
            "total_results": len(papers),
            "papers_with_pdf": papers_with_pdf,
            "papers": papers
        }

    except Exception as exc:
        print(f"Error processing search results: {exc}")
        return None


def save_search_results(data: Dict[str, Any], filename: Optional[str] = None) -> str:
    """Save results to JSON."""
    if not data or "topic" not in data:
        raise ValueError("save_search_results requires data dict.")

    if not filename:
        safe_topic = "".join(c for c in data["topic"] if c.isalnum() or c == " ").strip()
        safe_topic = safe_topic.replace(" ", "_") or "search"
        filename = f"paper_search_results_{safe_topic}.json"

    os.makedirs("data/search_results", exist_ok=True)
    filepath = os.path.join("data/search_results", filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)

    print(f"Search results saved to: {filepath}")
    return filepath


def display_search_results(data: Dict[str, Any], max_display: int = 10) -> None:
    """Display results table."""
    if not data or "papers" not in data:
        print("No data to display.")
        return

    papers = data["papers"]
    
    # Simple console display to avoid Pandas dependencies if needed
    print("\n" + "=" * 60)
    print(f" RESULTS: {data.get('topic', 'Unknown')}")
    print("=" * 60)
    
    for i, p in enumerate(papers[:max_display]):
        pdf_status = "✅ PDF" if p['has_pdf'] else "❌ No PDF"
        print(f"{i+1}. [{pdf_status}] {p['title'][:80]}...")
        print(f"   Year: {p['year']} | Citations: {p['citationCount']}")
        print(f"   Authors: {', '.join(p['authors'][:2])}")
        print("-" * 60)


# ====================
# MAIN EXECUTION
# ====================

def main_search(topic: Optional[str] = None):
    """Main execution flow."""
    print("\nRESEARCH PAPER AUTOMATION TOOL")
    print("Module 1: Discovery Phase")
    
    if topic is None:
        topic_input = input("\n >>> Enter research topic (default: 'machine learning'): ").strip()
        if topic_input:
            topic = topic_input
    
    if not topic:
        topic = "machine learning"

    results = search_papers(topic, limit=20)

    if results:
        save_path = save_search_results(results)
        display_search_results(results)
        return results, save_path
    else:
        print("\n [!] No results found (Connection timed out or 0 papers).")
        return None, None

if __name__ == "__main__":
    main_search()
