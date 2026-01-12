# ============================================
# MODULE 1: TOPIC INPUT & PAPER SEARCH (UI COMPATIBLE)
# ============================================

import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from semanticscholar import SemanticScholar

# 

def setup_api_key() -> SemanticScholar:
    """
    Initialize SemanticScholar.
    Looks for .env in the current directory AND the parent directory.
    """
    # 1. Try loading from current directory
    load_dotenv()
    
    # 2. If not found, look in parent directory (common issue in modular projects)
    if not os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        parent_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(parent_env)

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if not api_key:
        print(" [Config] No API Key found. Using limited rate (Timeout=60s).")
        # Increase timeout to avoid 'Endpoint request timed out' errors
        return SemanticScholar(timeout=60)
    else:
        print(" [Config] API Key loaded.")
        return SemanticScholar(api_key=api_key, timeout=60)


def search_papers(topic: str, limit: int = 20) -> Optional[Dict[str, Any]]:
    """
    Search Semantic Scholar for papers on a given topic.
    """
    if not topic or not topic.strip():
        print(" [Error] Search requires a non-empty topic.")
        return None

    print(f"\nSearching for papers on: '{topic}' (limit={limit})")

    scholar_client = setup_api_key()

    try:
        # Request fields that are useful downstream
        results = scholar_client.search_paper(
            query=topic,
            limit=limit,
            fields=[
                "paperId", "title", "abstract", "year", "authors",
                "citationCount", "openAccessPdf", "url", "venue"
            ]
        )

        papers: List[Dict[str, Any]] = []

        # Iterate safely through results
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

            # Handle abstract truncation safely
            abstract_text = getattr(paper, "abstract", "") or ""
            if abstract_text and len(abstract_text) > 300:
                abstract_text = abstract_text[:300] + "..."

            paper_entry = {
                "title": getattr(paper, "title", "") or "No title",
                "authors": authors,
                "year": getattr(paper, "year", None),
                "paperId": getattr(paper, "paperId", None),
                "abstract": abstract_text,
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
        print(f"Error searching papers: {exc}")
        return None


def save_search_results(data: Dict[str, Any], filename: Optional[str] = None) -> str:
    """Save search results to JSON."""
    if not data or "topic" not in data:
        return ""

    if not filename:
        safe_topic = "".join(c for c in data["topic"] if c.isalnum() or c == " ").strip()
        safe_topic = safe_topic.replace(" ", "_") or "search"
        filename = f"paper_search_results_{safe_topic}.json"

    # Use os.path.join for cross-platform compatibility
    save_dir = os.path.join("data", "search_results")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)

    print(f"Search results saved to: {filepath}")
    return filepath


def main_search(topic: Optional[str] = None):
    """
    Main entry point.
    CRITICAL FIX: Accepts 'topic' as an argument so Streamlit can pass data to it.
    """
    print("\n" + "=" * 72)
    print("MODULE 1: TOPIC INPUT & PAPER SEARCH")
    print("=" * 72)

    # If no topic passed (e.g. running from command line), ask for input
    if topic is None:
        try:
            topic_input = input("\nEnter research topic: ").strip()
            if topic_input:
                topic = topic_input
        except Exception:
            pass

    # Fallback default
    if not topic:
        topic = "machine learning"

    # Execute Search
    results = search_papers(topic, limit=20)
    
    if not results:
        print(" [!] No results found.")
        return None, None

    save_path = save_search_results(results)
    
    # We skip the heavy pandas display here to keep the UI logs clean
    print(f"\nModule 1 complete. Found {len(results['papers'])} papers.")
    return results, save_path


if __name__ == "__main__":
    main_search()
