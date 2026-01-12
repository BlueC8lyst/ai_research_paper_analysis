# ============================================
# MODULE 4: CROSS-PAPER ANALYSIS (FIXED)
# ============================================

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====================
# 1. LOAD EXTRACTED PAPERS
# ====================

def load_extracted_papers(data_dir: str = "data/extracted") -> List[Dict[str, Any]]:
    """Load all extracted paper JSON files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print("Data directory not found.")
        return []

    json_files = sorted(data_path.glob("*_extracted.json"))
    if not json_files:
        print("No extracted papers found. Run Module 3 first.")
        return []

    papers = []
    print(f"Loading {len(json_files)} extracted papers from {data_path}...")

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                papers.append(data)
        except Exception as exc:
            print(f"  Error loading {jf.name}: {exc}")

    return papers

# ====================
# 2. KEY INFORMATION EXTRACTION
# ====================

def extract_key_information(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured key information from a paper."""
    sections = paper.get("sections", {})
    full_text = sections.get("extracted_text", "")[:10000]
    
    # 1. Year
    year = paper.get("year", "Unknown")
    
    # 2. Methods
    methods_text = sections.get("methods", full_text)
    methods = _extract_keywords(methods_text, [
        "deep learning", "machine learning", "neural network", "transformer",
        "cnn", "rnn", "lstm", "bert", "gpt", "reinforcement learning",
        "statistical", "regression", "classification", "clustering",
        "svm", "random forest", "xgboost", "bayesian", "monte carlo"
    ])

    # 3. Datasets
    datasets = _extract_keywords(full_text, [
        "imagenet", "cifar", "mnist", "coco", "pascal", "wikitext",
        "bookcorpus", "squad", "glue", "kaggle", "uci", "pubmed", "arxiv"
    ])

    # 4. Findings
    findings_text = sections.get("results", "") + " " + sections.get("conclusion", "")
    if len(findings_text) < 100: findings_text = full_text
    findings = _extract_sentences(findings_text, [
        "result shows", "findings show", "we found", "we demonstrate",
        "achieves", "outperforms", "improves", "increases", "reduces"
    ])

    return {
        "paper_id": paper.get("paper_id", "unknown"),
        "title": sections.get("title", "Unknown"),
        "year": year,
        "methods": methods[:5],
        "datasets": datasets[:5],
        "key_findings": findings[:5],
        "metrics": [] # Simplified for robustness
    }

def _extract_keywords(text: str, keywords: List[str]) -> List[str]:
    """Helper to find keywords in text."""
    found = set()
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            found.add(kw)
    return list(found)

def _extract_sentences(text: str, triggers: List[str]) -> List[str]:
    """Helper to find sentences containing trigger phrases."""
    sentences = re.split(r'[.!?]+', text)
    found = []
    for sent in sentences:
        if any(t in sent.lower() for t in triggers):
            clean = re.sub(r'\s+', ' ', sent).strip()
            if len(clean) > 20 and clean not in found:
                found.append(clean[:200])
    return found

# ====================
# 3. COMPARISON LOGIC
# ====================

def compare_papers(papers_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple papers."""
    
    # Common Methods
    all_methods = [m for p in papers_info for m in p.get("methods", [])]
    method_counts = defaultdict(int)
    for m in all_methods: method_counts[m] += 1
    common_methods = [m for m, c in method_counts.items() if c > 1]

    # Similarities (TF-IDF)
    similarity_scores = _calculate_similarity(papers_info)

    return {
        "total_papers": len(papers_info),
        "papers": papers_info,
        "common_methods": common_methods,
        "similarity_scores": similarity_scores
    }

def _calculate_similarity(papers_info: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute TF-IDF cosine similarity."""
    if len(papers_info) < 2: return {}
    
    texts = []
    ids = []
    for p in papers_info:
        # Combine title + findings for semantic comparison
        content = f"{p['title']} {' '.join(p['key_findings'])}"
        texts.append(content)
        ids.append(p['paper_id'])
        
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(texts)
        sim_matrix = cosine_similarity(matrix)
        
        scores = {}
        for i in range(len(ids)):
            scores[ids[i]] = {}
            for j in range(len(ids)):
                if i != j:
                    scores[ids[i]][ids[j]] = round(float(sim_matrix[i][j]), 3)
        return scores
    except:
        return {}

# ====================
# 4. SAVE RESULTS
# ====================

def save_results(data: Dict[str, Any], output_dir: str = "data/analysis") -> None:
    """Save results to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save Main Comparison
    if "comparison" in data:
        with (out_path / "comparison.json").open("w", encoding="utf-8") as f:
            json.dump(data["comparison"], f, indent=2)
            
    # Save Single Analysis (if applicable)
    if "analysis" in data:
        with (out_path / "single_paper_analysis.json").open("w", encoding="utf-8") as f:
            json.dump(data["analysis"], f, indent=2)

# ====================
# 5. MAIN EXECUTION (RENAMED TO MATCH APP.PY)
# ====================

def main_analysis() -> Optional[Dict[str, Any]]:
    """
    Main entry point for Module 4.
    Renamed from 'run_analysis' to 'main_analysis' to match app.py.
    """
    print("\n" + "=" * 72)
    print("MODULE 4: INTELLIGENT ANALYSIS")
    print("=" * 72)

    raw_papers = load_extracted_papers()
    if not raw_papers:
        return None

    # Extract info for all papers
    papers_info = [extract_key_information(p) for p in raw_papers]

    if len(papers_info) == 1:
        print("\n[Mode] Single Paper Analysis")
        analysis = papers_info[0] # simplified for robustness
        result = {"type": "single", "analysis": analysis}
        save_results(result)
        return result
    else:
        print(f"\n[Mode] Comparative Analysis ({len(papers_info)} papers)")
        comparison = compare_papers(papers_info)
        result = {"type": "comparison", "data": {"comparison": comparison}}
        save_results(result)
        return result

if __name__ == "__main__":
    main_analysis()
