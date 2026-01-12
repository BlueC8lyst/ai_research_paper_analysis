# !pip install semanticscholar python-dotenv requests -q

import json
import os
import textwrap
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from semanticscholar import SemanticScholar


def setup_api_key() -> SemanticScholar:
    """
    Initialize and return a SemanticScholar client.

    Behavior:
    - Attempts to load SEMANTIC_SCHOLAR_API_KEY from a .env file.
    - If not found, does NOT write a real API key to disk (hard-coded keys removed).
      Instead, continues without a key (limited rate) and prints clear instructions.
    - Returns an initialized SemanticScholar client (with or without api_key).

    Returns:
        SemanticScholar: Initialized client object.
    """
    load_dotenv()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if not api_key:
        print(
            "SEMANTIC_SCHOLAR_API_KEY not found in environment. "
            "Proceeding without API key (limited rate)."
        )
        print(
            "To use a key: create a .env file with a line like:\n"
            "SEMANTIC_SCHOLAR_API_KEY=wFKolR3bfa5XUZaFntmdo5AXd7kL506y1klYRd3y\n"
            "Then re-run this script."
        )
        scholar_client = SemanticScholar()
    else:
        scholar_client = SemanticScholar(api_key=api_key)
        print("Semantic Scholar initialized with API key.")

    return scholar_client



def search_papers(topic: str, limit: int = 20) -> Optional[Dict[str, Any]]:
    """
    Search Semantic Scholar for papers on a given topic.

    Args:
        topic (str): Topic/query string for searching papers.
        limit (int): Maximum number of papers to request.

    Returns:
        dict or None: Dictionary containing search metadata and papers list
                      or None if an error occurred.
    """
    if not topic or not topic.strip():
        raise ValueError("search_papers requires a non-empty topic string.")

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
                "abstract": (getattr(paper, "abstract", "") or "")[:300] + ("..." if getattr(paper, "abstract", None) and len(getattr(paper, "abstract", "")) > 300 else ""),
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
    """
    Save search results dict to a JSON file under data/search_results.

    Args:
        data (dict): Data returned by `search_papers`.
        filename (str, optional): Custom filename. If None, generate from topic.

    Returns:
        str: Full path of the saved JSON file.
    """
    if not data or "topic" not in data:
        raise ValueError("save_search_results requires data dictionary with a 'topic' key.")

    # Create a filesystem-safe filename if not provided
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
    """
    Display search results as a pandas DataFrame (table).

    If running in a Jupyter / notebook environment the DataFrame will render
    as a nice HTML table. In a plain console, the DataFrame will be printed
    as text. Shows top `max_display` papers.
    """
    if not data or "papers" not in data:
        print("No data to display.")
        return

    papers = data["papers"]
    total = len(papers)
    pdf_count = sum(1 for p in papers if p.get("has_pdf"))
    no_pdf_count = total - pdf_count

    print("\n" + "=" * 72)
    print(f"SEARCH RESULTS: {data.get('topic', 'Unknown topic')}")
    print("=" * 72)
    print("\nStatistics:")
    print(f"  • Total papers: {total}")
    print(f"  • Papers with PDF: {pdf_count}")
    print(f"  • Papers without PDF: {no_pdf_count}")

    to_show = min(max_display, total)
    if to_show == 0:
        print("\nNo papers to display.")
        return

    # Build rows for DataFrame
    rows = []
    for idx, paper in enumerate(papers[:to_show], start=1):
        title = paper.get("title", "") or ""
        authors = paper.get("authors", []) or []
        authors_display = ", ".join(authors)
        year = paper.get("year", "")
        citations = paper.get("citationCount", 0)
        has_pdf = paper.get("has_pdf", False)
        pdf_url = paper.get("pdf_url", "") or ""
        url = paper.get("url", "") or ""
        abstract = (paper.get("abstract") or "")
        if len(abstract) > 300:
            abstract = abstract[:297] + "..."

        rows.append({
            "#": idx,
            "Title": title,
            "Authors": authors_display,
            "Year": year,
            "Citations": citations,
            "Has PDF": has_pdf,
            "PDF URL": pdf_url,
            "URL": url,
            "Abstract": abstract
        })

    df = pd.DataFrame(rows)

    col_order = ["#", "Title", "Authors", "Year", "Citations", "Has PDF", "PDF URL", "URL", "Abstract"]
    df = df[col_order]

    try:
        from IPython.display import display as _display, HTML
        _display(df)
    except Exception:
        pd.set_option("display.max_colwidth", 120)
        print("\nTop results (DataFrame):\n")
        print(df.to_string(index=False))

    print(f"\nShowing top {to_show} of {total} papers. Use `max_display` to change the table size.")



def main_search() -> (Optional[Dict[str, Any]], Optional[str]):
    """
    Interactive main entry for Module 1.

    Returns:
        Tuple of (results dict or None, path to saved file or None).
    """
    print("\n" + "=" * 72)
    print("MODULE 1: TOPIC INPUT & PAPER SEARCH")
    print("=" * 72)

    try:
        topic = input("\nEnter research topic: ").strip()
    except Exception:
        topic = ""

    if not topic:
        topic = "artificial intelligence"

    results = search_papers(topic, limit=20)
    if not results:
        print("No results found or an error occurred during search.")
        return None, None

    save_path = save_search_results(results)
    display_search_results(results)

    print("\nModule 1 complete. Results saved to:", save_path)
    print("Proceed to Module 2 for paper selection and PDF download.")
    return results, save_path


if __name__ == "__main__":
    main_search()

