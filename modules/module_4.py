# pip install scikit-learn numpy

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# 1. LOAD EXTRACTED PAPERS
# -------------------------
def load_extracted_papers(data_dir: str = "data/extracted") -> List[Dict[str, Any]]:
    """
    Load all extracted paper JSON files from the given directory.

    Args:
        data_dir: Directory containing `*_extracted.json` files.

    Returns:
        List of paper dictionaries (parsed JSON).
    """
    data_path = Path(data_dir)
    papers: List[Dict[str, Any]] = []

    json_files = sorted(data_path.glob("*_extracted.json"))
    if not json_files:
        print("No extracted papers found. Run Module 3 first.")
        return []

    print(f"Loading {len(json_files)} extracted papers from {data_path}...")

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                papers.append(data)
                print(f"  ✓ {data.get('paper_id', jf.stem)}: {data.get('total_characters', 0):,} chars")
        except Exception as exc:
            print(f"  Error loading {jf.name}: {exc}")

    return papers


# -------------------------
# 2. SINGLE PAPER ANALYSIS
# -------------------------
def analyze_single_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform an in-depth analysis of a single extracted paper.

    Args:
        paper: A single paper dict (as produced by Module 3).

    Returns:
        A structured analysis dictionary with extracted insights.
    """
    print("\nPerforming deep analysis of a single paper...")

    info = extract_key_information(paper)

    analysis = {
        "paper_id": info.get("paper_id"),
        "title": info.get("title"),
        "year": info.get("year"),
        "methods_used": info.get("methods", []),
        "datasets_mentioned": info.get("datasets", []),
        "key_findings": info.get("key_findings", []),
        "limitations": info.get("limitations", []),
        "contributions": info.get("contributions", []),
        "metrics_reported": info.get("metrics", []),
        "paper_structure": analyze_paper_structure(paper),
        "research_quality_indicators": assess_research_quality(info),
        "recommendations_for_future_research": generate_recommendations(info)
    }

    return analysis


def analyze_paper_structure(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze which standard sections are present and their lengths.

    Args:
        paper: extracted paper dict.

    Returns:
        Dict describing present/missing sections and lengths.
    """
    sections = paper.get("sections", {})
    structure: Dict[str, Any] = {
        "sections_present": [],
        "sections_missing": [],
        "section_lengths": {}
    }

    expected_sections = ["title", "abstract", "introduction", "methods", "results", "conclusion", "references"]

    for sec in expected_sections:
        content = sections.get(sec, "")
        if content and len(content.split()) > 10:
            structure["sections_present"].append(sec)
            structure["section_lengths"][sec] = len(content.split())
        else:
            structure["sections_missing"].append(sec)

    return structure


def assess_research_quality(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess research quality using simple heuristic indicators.

    Args:
        info: result of extract_key_information.

    Returns:
        Dict with quality indicators and an overall score.
    """
    quality_indicators: Dict[str, Any] = {
        "has_methods": bool(info.get("methods")),
        "has_datasets": bool(info.get("datasets")),
        "has_findings": bool(info.get("key_findings")),
        "has_limitations": bool(info.get("limitations")),
        "has_metrics": bool(info.get("metrics")),
        "method_diversity": len(info.get("methods", [])),
        "finding_clarity": len(info.get("key_findings", []))
    }

    score = 0
    max_score = 7
    if quality_indicators["has_methods"]:
        score += 1
    if quality_indicators["has_datasets"]:
        score += 1
    if quality_indicators["has_findings"]:
        score += 1
    if quality_indicators["has_limitations"]:
        score += 1
    if quality_indicators["has_metrics"]:
        score += 1
    if quality_indicators["method_diversity"] >= 2:
        score += 1
    if quality_indicators["finding_clarity"] >= 2:
        score += 1

    quality_indicators["overall_score"] = f"{score}/{max_score}"
    quality_indicators["percentage"] = (score / max_score) * 100

    return quality_indicators


def generate_recommendations(info: Dict[str, Any]) -> List[str]:
    """
    Produce short recommendations for future research based on extracted info.

    Args:
        info: key information extracted from a paper.

    Returns:
        List of recommendation strings.
    """
    recommendations: List[str] = []

    methods = info.get("methods", [])
    if methods:
        recommendations.append(f"Compare with other studies using: {methods[0]}")

    limitations = info.get("limitations", [])
    if limitations:
        recommendations.append(f"Address limitations such as: {limitations[0][:120]}...")

    datasets = info.get("datasets", [])
    if datasets:
        recommendations.append("Explore additional datasets to validate findings")

    recommendations.append("Compare with recent papers in the same domain")
    recommendations.append("Consider using alternative methodologies referenced in related work")

    return recommendations[:3]


# -------------------------
# 3. KEY INFORMATION EXTRACTION
# -------------------------
def extract_key_information(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured key information from a paper.

    Args:
        paper: paper dict with 'sections' etc.

    Returns:
        Dict with fields: paper_id, title, year, methods, datasets, key_findings, limitations, contributions, metrics.
    """
    sections = paper.get("sections", {})
    info = {
        "paper_id": paper.get("paper_id", "unknown"),
        "title": sections.get("title", "Unknown"),
        "year": extract_year(paper),
        "methods": extract_methods(paper),
        "datasets": extract_datasets(paper),
        "key_findings": extract_key_findings(paper),
        "limitations": extract_limitations(paper),
        "contributions": extract_contributions(paper),
        "metrics": extract_metrics(paper)
    }
    return info


def extract_year(paper: Dict[str, Any]) -> str:
    """
    Try to extract a 4-digit year from the title or extracted_text.

    Returns year as string or 'Unknown'.
    """
    title = paper.get("sections", {}).get("title", "") or ""
    match = re.search(r"\b(19|20)\d{2}\b", title)
    if match:
        return match.group(0)

    text = paper.get("sections", {}).get("extracted_text", "") or ""
    match = re.search(r"\b(19|20)\d{2}\b", text[:5000])
    if match:
        return match.group(0)

    if paper.get("year"):
        return str(paper.get("year"))

    return "Unknown"


def extract_methods(paper: Dict[str, Any]) -> List[str]:
    """
    Extract likely methods mentioned in methods/results/conclusion sections using keywords.

    Returns up to 5 method snippets.
    """
    methods_text = paper.get("sections", {}).get("methods", "") or ""
    if not methods_text:
        methods_text = (paper.get("sections", {}).get("extracted_text", "") or "")[:5000]

    method_keywords = [
        "deep learning", "machine learning", "neural network", "transformer",
        "cnn", "rnn", "lstm", "bert", "gpt", "reinforcement learning",
        "statistical", "regression", "classification", "clustering",
        "svm", "random forest", "xgboost", "bayesian", "monte carlo",
        "simulation", "experiment", "analysis", "framework", "model",
        "algorithm", "approach", "technique", "methodology"
    ]

    found: List[str] = []
    sentences = re.split(r"[.!?]+", methods_text.lower())

    for sent in sentences:
        for kw in method_keywords:
            if kw in sent and len(sent.strip()) > 20:
                clean = re.sub(r"\s+", " ", sent).strip()
                if clean not in found:
                    found.append(clean[:200])
                    break
        if len(found) >= 5:
            break

    # Fallback: look in results/conclusion
    if not found:
        combined = (paper.get("sections", {}).get("results", "") or "") + " " + (paper.get("sections", {}).get("conclusion", "") or "")
        for sent in re.split(r"[.!?]+", combined.lower()):
            for kw in method_keywords[:10]:
                if kw in sent and len(sent.strip()) > 20:
                    clean = re.sub(r"\s+", " ", sent).strip()
                    if clean not in found:
                        found.append(clean[:200])
                        break
            if len(found) >= 5:
                break

    return found[:5]


def extract_datasets(paper: Dict[str, Any]) -> List[str]:
    """
    Find mentions of common datasets or dataset-like phrases within the extracted_text.
    """
    text = (paper.get("sections", {}).get("extracted_text", "") or "").lower()[:10000]

    dataset_patterns = [
        r"imagenet", r"cifar", r"mnist", r"coco", r"pascal", r"wikitext",
        r"bookcorpus", r"squad", r"glue", r"superglue", r"kaggle", r"uci",
        r"pubmed", r"arxiv", r"dataset", r"corpus", r"benchmark", r"repository"
    ]
    found: List[str] = []

    for pat in dataset_patterns:
        if re.search(pat, text):
            found.append(pat)

    sentences = re.split(r"[.!?]+", text)
    for sent in sentences:
        if any(k in sent for k in ["dataset", "corpus", "benchmark", "collection"]):
            clean = re.sub(r"\s+", " ", sent).strip()[:150]
            if clean and clean not in found:
                found.append(clean)

    unique = []
    for x in found:
        if x not in unique:
            unique.append(x)
        if len(unique) >= 5:
            break

    return unique


def extract_key_findings(paper: Dict[str, Any]) -> List[str]:
    """
    Heuristically extract sentences that appear to describe findings/results.
    """
    text = (paper.get("sections", {}).get("results", "") or "") or (paper.get("sections", {}).get("conclusion", "") or "")
    if not text:
        text = (paper.get("sections", {}).get("extracted_text", "") or "")[:3000]

    result_keywords = [
        "result shows", "findings show", "we found", "we demonstrate",
        "achieves", "outperforms", "improves", "increases", "reduces",
        "accuracy", "precision", "recall", "f1", "score", "performance",
        "significant", "better than", "compared to", "surpasses"
    ]

    findings: List[str] = []
    sentences = re.split(r"[.!?]+", text.lower())

    for sent in sentences:
        if any(kw in sent for kw in result_keywords) and len(sent.strip()) > 30:
            clean = re.sub(r"\s+", " ", sent).strip()
            if clean not in findings:
                findings.append(clean[:300])
        if len(findings) >= 5:
            break

    if len(findings) < 2:
        conclusion = (paper.get("sections", {}).get("conclusion", "") or "")
        for sent in re.split(r"[.!?]+", conclusion.lower())[:5]:
            if len(sent.strip()) > 50:
                findings.append(re.sub(r"\s+", " ", sent).strip()[:300])
            if len(findings) >= 5:
                break

    return findings[:5]


def extract_limitations(paper: Dict[str, Any]) -> List[str]:
    """
    Extract sentences indicating limitations or future work.
    """
    text = (paper.get("sections", {}).get("conclusion", "") or "") or (paper.get("sections", {}).get("extracted_text", "") or "")[:5000]

    limitation_keywords = [
        "limitation", "drawback", "shortcoming", "weakness",
        "future work", "further research", "need to", "could be improved",
        "challenge", "difficulty", "issue", "problem", "not consider",
        "assumption", "restriction", "constraint", "only work"
    ]

    limitations: List[str] = []
    sentences = re.split(r"[.!?]+", text.lower())

    for sent in sentences:
        if any(kw in sent for kw in limitation_keywords) and len(sent.strip()) > 30:
            clean = re.sub(r"\s+", " ", sent).strip()
            if clean not in limitations:
                limitations.append(clean[:300])
        if len(limitations) >= 3:
            break

    return limitations[:3]


def extract_contributions(paper: Dict[str, Any]) -> List[str]:
    """
    Extract statements of contribution from abstract/introduction.
    """
    abstract = (paper.get("sections", {}).get("abstract", "") or "")[:1000]
    intro = (paper.get("sections", {}).get("introduction", "") or "")[:1000]
    text = (abstract + " " + intro).lower()

    contribution_keywords = [
        "contribution", "contribute", "propose", "introduce",
        "novel", "new method", "new approach", "we present",
        "this paper", "our work", "main contribution", "key contribution"
    ]

    contributions: List[str] = []
    for sent in re.split(r"[.!?]+", text):
        if any(kw in sent for kw in contribution_keywords) and len(sent.strip()) > 30:
            clean = re.sub(r"\s+", " ", sent).strip()
            if clean not in contributions:
                contributions.append(clean[:300])
        if len(contributions) >= 3:
            break

    return contributions[:3]


def extract_metrics(paper: Dict[str, Any]) -> List[str]:
    """
    Extract numeric metric mentions from the results section.
    """
    results_text = paper.get("sections", {}).get("results", "") or ""
    if not results_text:
        return []

    metric_patterns = [
        r"accuracy\s*[:=]\s*\d+\.?\d*%?", r"precision\s*[:=]\s*\d+\.?\d*%?",
        r"recall\s*[:=]\s*\d+\.?\d*%?", r"f1[\s\-]?score\s*[:=]\s*\d+\.?\d*%?",
        r"auc\s*[:=]\s*\d+\.?\d*", r"mae\s*[:=]\s*\d+\.?\d*", r"rmse\s*[:=]\s*\d+\.?\d*",
        r"\d+\.?\d*\s*%"
    ]

    metrics: List[str] = []
    for pat in metric_patterns:
        matches = re.findall(pat, results_text.lower())
        for m in matches:
            if m not in metrics:
                metrics.append(m)

    return metrics[:5]


# -------------------------
# 4. COMPARISON FUNCTIONS
# -------------------------
def compare_papers(papers_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple paper summaries to find similarities, differences, and gaps.

    Args:
        papers_info: list of paper summary dicts (from extract_key_information).

    Returns:
        Dict containing comparison results.
    """
    print(f"\nComparing {len(papers_info)} papers...")

    comparison = {
        "total_papers": len(papers_info),
        "papers": papers_info,
        "similarities": find_similarities(papers_info),
        "differences": find_differences(papers_info),
        "common_methods": find_common_elements(papers_info, "methods"),
        "common_datasets": find_common_elements(papers_info, "datasets"),
        "timeline_analysis": analyze_timeline(papers_info),
        "research_gaps": identify_research_gaps(papers_info)
    }

    return comparison


def find_similarities(papers_info: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Identify methods/datasets/findings that appear in more than one paper.
    """
    methods_count: defaultdict = defaultdict(int)
    datasets_count: defaultdict = defaultdict(int)
    findings_count: defaultdict = defaultdict(int)

    for paper in papers_info:
        for m in paper.get("methods", []):
            key = m[:50].lower()
            methods_count[key] += 1
        for d in paper.get("datasets", []):
            key = d[:50].lower()
            datasets_count[key] += 1
        for f in paper.get("key_findings", []):
            key = f[:50].lower()
            findings_count[key] += 1

    similar_items = {
        "methods": [item for item, cnt in methods_count.items() if cnt > 1 and len(item) > 10],
        "datasets": [item for item, cnt in datasets_count.items() if cnt > 1 and len(item) > 10],
        "findings": [item for item, cnt in findings_count.items() if cnt > 1 and len(item) > 10]
    }

    return similar_items


def find_differences(papers_info: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    For each paper, find unique methods/datasets/findings compared to other papers.
    """
    differences = {
        "unique_methods": defaultdict(list),
        "unique_datasets": defaultdict(list),
        "unique_findings": defaultdict(list)
    }

    paper_methods: Dict[str, Set[str]] = defaultdict(set)
    paper_datasets: Dict[str, Set[str]] = defaultdict(set)
    paper_findings: Dict[str, Set[str]] = defaultdict(set)
    all_methods: Set[str] = set()
    all_datasets: Set[str] = set()
    all_findings: Set[str] = set()

    for paper in papers_info:
        pid = paper.get("paper_id", "")
        for m in paper.get("methods", []):
            key = m[:50].lower()
            paper_methods[pid].add(key)
            all_methods.add(key)
        for d in paper.get("datasets", []):
            key = d[:50].lower()
            paper_datasets[pid].add(key)
            all_datasets.add(key)
        for f in paper.get("key_findings", []):
            key = f[:50].lower()
            paper_findings[pid].add(key)
            all_findings.add(key)

    for pid in paper_methods:
        unique_methods = paper_methods[pid] - set().union(*(paper_methods[qid] for qid in paper_methods if qid != pid))
        if unique_methods:
            differences["unique_methods"][pid] = list(unique_methods)[:3]

        unique_datasets = paper_datasets[pid] - set().union(*(paper_datasets[qid] for qid in paper_datasets if qid != pid))
        if unique_datasets:
            differences["unique_datasets"][pid] = list(unique_datasets)[:3]

        unique_findings = paper_findings[pid] - set().union(*(paper_findings[qid] for qid in paper_findings if qid != pid))
        if unique_findings:
            differences["unique_findings"][pid] = list(unique_findings)[:3]

    return {
        "unique_methods": dict(differences["unique_methods"]),
        "unique_datasets": dict(differences["unique_datasets"]),
        "unique_findings": dict(differences["unique_findings"])
    }


def find_common_elements(papers_info: List[Dict[str, Any]], element_type: str) -> List[str]:
    """
    Find elements (methods/datasets) that appear in every paper.

    Args:
        papers_info: list of paper info dicts.
        element_type: 'methods' or 'datasets' or other list-key.

    Returns:
        List of common elements (first 5).
    """
    element_sets: List[Set[str]] = []
    for paper in papers_info:
        elements = paper.get(element_type, [])
        element_sets.append({e[:50].lower() for e in elements if len(e) > 10})

    if not element_sets:
        return []

    common = set.intersection(*element_sets)
    return list(common)[:5]


def analyze_timeline(papers_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze temporal distribution of papers (earliest, latest, range, counts).

    Returns:
        timeline dict or note if insufficient data.
    """
    years: List[int] = []
    for p in papers_info:
        y = p.get("year", "Unknown")
        try:
            if isinstance(y, str) and y.isdigit():
                yi = int(y)
            elif isinstance(y, int):
                yi = y
            else:
                continue
            if 1900 <= yi <= 2100:
                years.append(yi)
        except Exception:
            continue

    if len(years) >= 2:
        timeline = {
            "earliest": min(years),
            "latest": max(years),
            "range": max(years) - min(years),
            "count_by_year": {str(year): years.count(year) for year in sorted(set(years))}
        }
    else:
        timeline = {"note": "Insufficient year data"}

    return timeline


def identify_research_gaps(papers_info: List[Dict[str, Any]]) -> List[str]:
    """
    Heuristically identify research gaps from aggregated limitations and un-used popular methods.
    """
    gaps: List[str] = []

    all_limitations: List[str] = []
    for p in papers_info:
        all_limitations.extend(p.get("limitations", []))

    limitation_counts: defaultdict = defaultdict(int)
    for lim in all_limitations:
        key = lim[:100].lower()
        limitation_counts[key] += 1

    frequent_limitations = [lim for lim, cnt in limitation_counts.items() if cnt > 1 and len(lim) > 20]
    if frequent_limitations:
        gaps.append("Common limitations across papers:")
        gaps.extend(frequent_limitations[:3])

    methods_used = {m.lower() for p in papers_info for m in p.get("methods", [])}
    datasets_used = {d.lower() for p in papers_info for d in p.get("datasets", [])}

    common_methods_in_field = [
        "deep learning", "transfer learning", "reinforcement learning",
        "explainable ai", "few-shot learning", "meta learning"
    ]

    missing_methods = [m for m in common_methods_in_field if m not in methods_used]
    if missing_methods:
        gaps.append("Potentially unexplored methods in this set of papers:")
        gaps.extend(missing_methods[:3])

    return gaps[:5]


def calculate_similarity_scores(papers_info: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise cosine similarity between papers using TF-IDF on title+abstract+findings.

    Returns:
        Nested dict mapping paper_id -> other_paper_id -> similarity score (0..1).
    """
    texts: List[str] = []
    paper_ids: List[str] = []

    for p in papers_info:
        pid = p.get("paper_id", "")
        paper_ids.append(pid)
        title = p.get("title", "")
        abstract = p.get("sections", {}).get("abstract", "")[:1000]
        findings = " ".join(p.get("key_findings", []))[:1000]
        combined = " ".join([title, abstract, findings])
        texts.append(combined)

    if not texts:
        return {}

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    try:
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except Exception as exc:
        print(f"Error computing similarity matrix: {exc}")
        # Return empty similarity if failure
        return {}

    similarity_scores: Dict[str, Dict[str, float]] = {}
    n = len(paper_ids)
    for i in range(n):
        similarity_scores[paper_ids[i]] = {}
        for j in range(n):
            if i == j:
                continue
            similarity_scores[paper_ids[i]][paper_ids[j]] = float(f"{sim_matrix[i, j]:.3f}")

    return similarity_scores


# -------------------------
# 5. SAVE RESULTS
# -------------------------
def save_results(analysis_type: str, data: Dict[str, Any], output_dir: str = "data/analysis") -> str:
    """
    Save analysis results to JSON and generate human-readable reports.

    Args:
        analysis_type: "single" or "comparison"
        data: analysis data dict
        output_dir: directory to save results

    Returns:
        Path to output directory as string.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if analysis_type == "single":
        output_file = output_path / "single_paper_analysis.json"
        with output_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"   Single paper analysis saved to: {output_file}")
        generate_single_paper_report(data, output_path)

    elif analysis_type == "comparison":
        comparison_file = output_path / "comparison.json"
        with comparison_file.open("w", encoding="utf-8") as fh:
            json.dump(data.get("comparison", {}), fh, indent=2, ensure_ascii=False)
        print(f"  Comparison saved to: {comparison_file}")

        similarity_file = output_path / "similarity_scores.json"
        with similarity_file.open("w", encoding="utf-8") as fh:
            json.dump(data.get("similarity_scores", {}), fh, indent=2, ensure_ascii=False)
        print(f"   Similarity scores saved to: {similarity_file}")

        generate_comparison_report(data, output_path)

    return str(output_path)


def generate_single_paper_report(analysis: Dict[str, Any], output_path: Path) -> None:
    """
    Create a human-readable text report for a single-paper analysis.
    """
    report_lines: List[str] = []
    report_lines.append("=" * 80)
    report_lines.append("SINGLE PAPER IN-DEPTH ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\n PAPER: {analysis.get('paper_id')}")
    report_lines.append(f" Title: {analysis.get('title')}")
    report_lines.append(f" Year: {analysis.get('year')}\n")

    report_lines.append("METHODS IDENTIFIED:")
    report_lines.append("-" * 40)
    if analysis.get("methods_used"):
        for m in analysis["methods_used"]:
            report_lines.append(f"• {m}")
    else:
        report_lines.append("No specific methods identified")

    report_lines.append("\nKEY FINDINGS:")
    report_lines.append("-" * 40)
    if analysis.get("key_findings"):
        for f in analysis["key_findings"]:
            report_lines.append(f"• {f}")
    else:
        report_lines.append("No key findings extracted")

    report_lines.append("\nLIMITATIONS:")
    report_lines.append("-" * 40)
    if analysis.get("limitations"):
        for lim in analysis["limitations"]:
            report_lines.append(f"• {lim}")
    else:
        report_lines.append("No limitations mentioned")

    report_lines.append("\nRESEARCH QUALITY:")
    report_lines.append("-" * 40)
    quality = analysis.get("research_quality_indicators", {})
    report_lines.append(f"Overall Score: {quality.get('overall_score', 'N/A')} ({quality.get('percentage', 0):.1f}%)")
    report_lines.append(f"Has Methods: {'✅' if quality.get('has_methods') else '❌'}")
    report_lines.append(f"Has Datasets: {'✅' if quality.get('has_datasets') else '❌'}")
    report_lines.append(f"Has Findings: {'✅' if quality.get('has_findings') else '❌'}")
    report_lines.append(f"Has Limitations: {'✅' if quality.get('has_limitations') else '❌'}")

    report_lines.append("\nRECOMMENDATIONS FOR FUTURE RESEARCH:")
    report_lines.append("-" * 40)
    for r in analysis.get("recommendations_for_future_research", []):
        report_lines.append(f"• {r}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("ANALYSIS COMPLETE")
    report_lines.append("=" * 80)

    report_file = output_path / "single_paper_report.txt"
    with report_file.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    print(f"   Summary report saved to: {report_file}")


def generate_comparison_report(data: Dict[str, Any], output_path: Path) -> None:
    """
    Create a human-readable comparison report for multiple papers.
    """
    comparison = data.get("comparison", {})
    similarity_scores = data.get("similarity_scores", {})

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("CROSS-PAPER COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTotal papers analyzed: {comparison.get('total_papers', 0)}\n")

    lines.append("PAPERS ANALYZED:")
    lines.append("-" * 40)
    for paper in comparison.get("papers", []):
        lines.append(f"\n• {paper.get('paper_id', 'unknown')}")
        lines.append(f"  Title: {paper.get('title', 'Unknown')}")
        lines.append(f"  Year: {paper.get('year', 'Unknown')}")
        lines.append(f"  Methods: {len(paper.get('methods', []))} found")
        lines.append(f"  Datasets: {len(paper.get('datasets', []))} found")

    lines.append("\nKEY SIMILARITIES:")
    lines.append("-" * 40)
    sim = comparison.get("similarities", {})
    if sim.get("methods"):
        lines.append("\nCommon Methods:")
        for m in sim["methods"]:
            lines.append(f"  • {m}")
    if sim.get("datasets"):
        lines.append("\nCommon Datasets:")
        for d in sim["datasets"]:
            lines.append(f"  • {d}")

    lines.append("\nPAPER SIMILARITY SCORES:")
    lines.append("-" * 40)
    for pid, scores in similarity_scores.items():
        lines.append(f"\n{pid}:")
        for other, score in scores.items():
            lines.append(f"  vs {other}: {score:.3f}")

    if comparison.get("research_gaps"):
        lines.append("\nIDENTIFIED RESEARCH GAPS:")
        lines.append("-" * 40)
        for gap in comparison["research_gaps"]:
            lines.append(f"• {gap}")

    lines.append("\n" + "=" * 80)
    lines.append("COMPARISON COMPLETE")
    lines.append("=" * 80)

    report_file = output_path / "comparison_report.txt"
    with report_file.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"  Comparison report saved to: {report_file}")


# -------------------------
# 6. MAIN ANALYSIS PIPELINE
# -------------------------
def run_analysis() -> Optional[Dict[str, Any]]:
    """
    Main pipeline entrypoint for Module 4.

    - Loads extracted papers
    - If a single paper: run deep single-paper analysis
    - If multiple papers: extract key info, compare, compute similarities, and save
    """
    print("\n" + "=" * 80)
    print("PAPER ANALYSIS MODULE")
    print("=" * 80)

    print("\nSTEP 1: Loading extracted papers...")
    papers = load_extracted_papers()
    if not papers:
        print(" No papers to analyze")
        return None

    if len(papers) == 1:
        print("\nℹ Only 1 paper found. Performing in-depth single paper analysis...")
        paper = papers[0]
        analysis = analyze_single_paper(paper)
        info = extract_key_information(paper)

        print("\nSTEP 2: Saving analysis results...")
        save_path = save_results("single", analysis)

        print("\nSINGLE PAPER ANALYSIS COMPLETE!")
        print(f"Files saved to: {save_path}")
        return {"type": "single", "analysis": analysis, "paper_info": info}

    else:
        print(f"\nSTEP 2: Analyzing {len(papers)} papers for comparison...")
        papers_info: List[Dict[str, Any]] = []
        for p in papers:
            info = extract_key_information(p)
            papers_info.append(info)
            print(f"  ✓ {info.get('paper_id')}: {len(info.get('methods', []))} methods, {len(info.get('key_findings', []))} findings")

        print("\nSTEP 3: Comparing papers...")
        comparison = compare_papers(papers_info)

        print("\nSTEP 4: Calculating similarity scores...")
        similarity_scores = calculate_similarity_scores(papers_info)

        print("\nSTEP 5: Saving comparison results...")
        data = {
            "comparison": comparison,
            "similarity_scores": similarity_scores
        }
        save_path = save_results("comparison", data)

        print("\nCROSS-PAPER ANALYSIS COMPLETE!")
        print(f"Files saved to: {save_path}")

        return {"type": "comparison", "data": data, "papers_info": papers_info}


# -------------------------
# 7. DEMO / TEST HELPERS
# -------------------------
def create_demo_paper_for_testing() -> Dict[str, Any]:
    """
    Create a small demo paper info dict for testing comparison features.
    """
    demo_paper = {
        "paper_id": "demo_paper_ai_ethics",
        "title": "Ethical Considerations in Artificial Intelligence Systems",
        "year": "2023",
        "methods": ["machine learning", "ethical framework analysis", "case studies"],
        "datasets": ["AI ethics guidelines corpus", "public opinion surveys"],
        "key_findings": [
            "AI systems show bias in 78% of tested scenarios",
            "Current ethical frameworks lack enforcement mechanisms",
            "Transparency is the most cited ethical concern"
        ],
        "limitations": [
            "Study limited to Western ethical frameworks",
            "Small sample size for public opinion data"
        ],
        "contributions": [
            "Proposes new AI ethics assessment framework",
            "Identifies key gaps in current regulations"
        ],
        "metrics": ["accuracy: 85%", "f1-score: 0.82"]
    }
    return demo_paper


def run_with_demo_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Example function showing how to run comparison with a real paper + a demo paper.
    """
    real_papers = load_extracted_papers()
    if not real_papers:
        raise RuntimeError("No real papers found to demo with.")

    demo = create_demo_paper_for_testing()
    real_info = extract_key_information(real_papers[0])

    papers_info = [real_info, demo]
    comparison = compare_papers(papers_info)
    similarity_scores = calculate_similarity_scores(papers_info)

    print("\nDemo comparison complete.")
    return comparison, similarity_scores


# -------------
# Entry point
# -------------
if __name__ == "__main__":
    result = run_analysis()
    if result:
        if result.get("type") == "single":
            print("\nSingle paper analysis completed.")
        else:
            print("\nComparison analysis completed.")
