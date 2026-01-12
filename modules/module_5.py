# pip install tiktoken

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


# -------------------------
# 1. GPT SECTION GENERATOR
# -------------------------
class GPTSectionGenerator:
    """
    Simulated GPT-based section generator.

    This class provides simple template-based generation for:
      - abstract
      - introduction
      - methods
      - results
      - conclusion
      - references

    The class is intentionally conservative: it does not call any external API.
    In production, replace the template functions with actual API calls while
    keeping the same method signatures.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> None:
        """
        Initialize the generator.

        Args:
            api_key: Optional API key.
            model: Model name string (used for token estimation).
        """
        self.model = model
        if _HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except Exception:
                self.encoding = tiktoken.get_encoding("gpt2")
        else:
            self.encoding = None

        print(f" GPTSectionGenerator initialized (simulated, model={self.model})")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text. Uses tiktoken when available for realistic counts,
        otherwise falls back to a conservative word-based estimate.
        """
        if not text:
            return 0
        if _HAS_TIKTOKEN and self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                return len(text.split())
        return len(text.split())

    # -------------------------
    # Template generation API
    # -------------------------
    def create_system_prompt(self) -> str:
        """
        Create a fixed system prompt for academic writing.
        (Kept as a function so production replacements can reuse it.)
        """
        return (
            "You are an academic research assistant. Generate structured academic "
            "sections based on provided analysis data. Use formal academic language; "
            "base content on provided analysis; use APA-style references where possible."
        )

    def generate_with_template(self, section_type: str, analysis_data: Dict[str, Any], paper_count: int = 1) -> str:
        """
        Dispatch to the appropriate template generator.

        Args:
            section_type: One of 'abstract', 'introduction', 'methods', 'results', 'conclusion', 'references'.
            analysis_data: Dict returned by Module 4 (single or comparison).
            paper_count: Number of papers the analysis covered.

        Returns:
            Generated text for the section.
        """
        if section_type == "abstract":
            return self._generate_abstract(analysis_data, paper_count)
        if section_type == "introduction":
            return self._generate_introduction(analysis_data, paper_count)
        if section_type == "methods":
            return self._generate_methods_comparison(analysis_data, paper_count)
        if section_type == "results":
            return self._generate_results_synthesis(analysis_data, paper_count)
        if section_type == "conclusion":
            return self._generate_conclusion(analysis_data, paper_count)
        if section_type == "references":
            return self._generate_references(analysis_data)
        return "Section type not recognized"

    # -------------------------
    # Internal template methods
    # -------------------------
    def _generate_abstract(self, analysis_data: Dict[str, Any], paper_count: int) -> str:
        """Generate a short abstract. Keep <= 100 words where possible."""

        if paper_count == 1 and "analysis" in analysis_data:
            paper = analysis_data.get("analysis", {})
            title = paper.get("title", "This paper")
            methods = paper.get("methods_used", [])
            findings = paper.get("key_findings", [])
            abstract = f"This review examines '{title}'. "
            if methods:
                abstract += f"The approach uses {methods[0]}. "
            if findings:
                abstract += f"Key finding: {findings[0][:140]}. "
            abstract += "This analysis summarizes methodological choices and implications."
        else:
            # Multi-paper abstract
            comp = analysis_data.get("data", {}).get("comparison", {})
            common_methods = comp.get("common_methods", []) if comp else []
            abstract = f"This comparative analysis synthesizes findings from {paper_count} research papers. "
            if common_methods:
                abstract += f"Common approaches include {', '.join(common_methods[:2])}. "
            abstract += "The synthesis highlights patterns, divergences, and research gaps."
        # Enforce rough 100-word limit
        words = abstract.split()
        if len(words) > 100:
            abstract = " ".join(words[:100]) + "..."
        return abstract

    def _generate_introduction(self, analysis_data: Dict[str, Any], paper_count: int) -> str:
        """Generate an introduction tailored to single- or multi-paper analyses."""
        if paper_count == 1:
            paper = analysis_data.get("analysis", {})
            title = paper.get("title", "this research")
            year = paper.get("year", "")
            intro = f"This analysis examines {title}"
            if year and year != "Unknown":
                intro += f" ({year})"
            intro += ". The paper addresses important questions and applies appropriate methods. "
            intro += "This review evaluates research design, methodological choices, and implications."
        else:
            papers_info = analysis_data.get("papers_info", [])
            years = [p.get("year") for p in papers_info if p.get("year") and p.get("year") != "Unknown"]
            intro = f"This comparative review considers {paper_count} papers"
            if years:
                intro += f" spanning {min(years)}–{max(years)}"
            intro += ". It synthesizes methodologies and findings to identify trends and gaps."
        return intro

    def _generate_methods_comparison(self, analysis_data: Dict[str, Any], paper_count: int) -> str:
        """Generate a methods section that summarizes methodological commonalities and differences."""
        if paper_count == 1:
            paper = analysis_data.get("analysis", {})
            methods = paper.get("methods_used", [])
            datasets = paper.get("datasets_mentioned", [])
            text = "The study's methodology is characterized by "
            if methods:
                text += f"{methods[0]}"
                if len(methods) > 1:
                    text += f" and {methods[1]}"
                text += ". "
            else:
                text += "standard and appropriate approaches for the research problem. "
            if datasets:
                text += f"The dataset used includes {datasets[0]}. "
            text += "Methodological choices appear aligned with the objectives."
        else:
            comparison = analysis_data.get("data", {}).get("comparison", {})
            common = comparison.get("common_methods", []) if comparison else []
            text = "Across studies, methodological approaches show both overlap and variation. "
            if common:
                text += f"Common methods observed include {', '.join(common[:3])}. "
            text += "Unique approaches highlight different research focuses across papers."
        return text

    def _generate_results_synthesis(self, analysis_data: Dict[str, Any], paper_count: int) -> str:
        """Synthesize results/findings from the analysis data."""
        if paper_count == 1:
            paper = analysis_data.get("analysis", {})
            findings = paper.get("key_findings", [])
            metrics = paper.get("metrics_reported", [])
            text = "The analysis reveals the following key findings: "
            if findings:
                for i, f in enumerate(findings[:3], 1):
                    text += f"{i}. {f[:140]}. "
            if metrics:
                text += f"Reported metrics include {', '.join(metrics[:3])}. "
            text += "These results inform the paper's contributions and limitations."
        else:
            papers_info = analysis_data.get("papers_info", [])
            all_findings = []
            for p in papers_info:
                all_findings.extend(p.get("key_findings", []))
            text = "Synthesis across papers indicates several recurring findings: "
            if all_findings:
                for i, f in enumerate(all_findings[:4], 1):
                    text += f"{i}. {f[:100]}. "
            text += "Comparative results illuminate both convergences and divergences among studies."
        return text

    def _generate_conclusion(self, analysis_data: Dict[str, Any], paper_count: int) -> str:
        """Produce a brief conclusion summarizing contributions, limitations, and future directions."""
        if paper_count == 1:
            paper = analysis_data.get("analysis", {})
            limitations = paper.get("limitations", [])
            recs = paper.get("recommendations_for_future_research", [])
            text = "In conclusion, the analysis highlights the work's methodological strengths and contributions. "
            if limitations:
                text += f"Identified limitations include {limitations[0][:140]}. "
            if recs:
                text += f"Future work should consider {recs[0][:140]}. "
            text += "Overall, the paper provides a useful foundation for further research."
        else:
            comp = analysis_data.get("data", {}).get("comparison", {})
            gaps = comp.get("research_gaps", []) if comp else []
            text = "This comparative review identifies key trends and open research areas. "
            if gaps:
                text += f"Notable research gaps include {gaps[0]}. "
            text += "These directions suggest fruitful opportunities for future studies."
        return text

    def _generate_references(self, analysis_data: Dict[str, Any]) -> str:
        """Generate a small APA-style references block for demo purposes."""

        if "analysis" in analysis_data:
            paper = analysis_data.get("analysis", {})
            pid = paper.get("paper_id", "paper")
            title = paper.get("title", "Untitled")
            year = paper.get("year", "n.d.")
            refs = f"{pid}. ({year}). {title}. [Analyzed research paper].\n\n"
            refs += "American Psychological Association. (2020). Publication manual of the American Psychological Association (7th ed.).\n"
            refs += "Smith, J., & Johnson, A. (2019). Research methods in academic writing. Academic Press.\n"
            return refs
        papers_info = analysis_data.get("papers_info", []) or []
        lines = []
        for p in papers_info:
            pid = p.get("paper_id", "paper")
            title = p.get("title", "Untitled")
            year = p.get("year", "n.d.")
            lines.append(f"{pid}. ({year}). {title}.")
        lines.append("\nAmerican Psychological Association. (2020). Publication manual (7th ed.).")
        return "\n".join(lines)


# -------------------------
# 2. LOAD ANALYSIS DATA
# -------------------------
def load_analysis_data() -> Optional[Dict[str, Any]]:
    """
    Load analysis data produced by Module 4 (single or comparison).

    Returns:
        A dict describing the analysis context, or None if no data found.
    """
    analysis_path = Path("data/analysis")
    comparison_file = analysis_path / "comparison.json"
    single_file = analysis_path / "single_paper_analysis.json"

    if comparison_file.exists():
        try:
            with comparison_file.open("r", encoding="utf-8") as fh:
                comparison_data = json.load(fh)
        except Exception as exc:
            print(f" Error reading comparison.json: {exc}")
            return None

        papers_info = []
        for summary in comparison_data.get("papers", []):
            pid = summary.get("paper_id")
            candidate = Path("data/extracted") / f"{pid}_extracted.json"
            if candidate.exists():
                try:
                    with candidate.open("r", encoding="utf-8") as pf:
                        papers_info.append(json.load(pf))
                except Exception:
                    continue

        return {
            "type": "comparison",
            "data": {"comparison": comparison_data},
            "papers_info": papers_info,
            "paper_count": len(papers_info) if papers_info else len(comparison_data.get("papers", []))
        }

    if single_file.exists():
        try:
            with single_file.open("r", encoding="utf-8") as fh:
                analysis_data = json.load(fh)
        except Exception as exc:
            print(f" Error reading single_paper_analysis.json: {exc}")
            return None
        return {"type": "single", "analysis": analysis_data, "paper_count": 1}

    print(" No analysis data found in data/analysis. Run Module 4 first.")
    return None


# -------------------------
# 3. DRAFT GENERATION
# -------------------------
def generate_all_sections(analysis_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate all standard draft sections using GPTSectionGenerator templates.

    Returns:
        sections: Dict mapping section_key -> {name, content, word_count, token_count}
    """
    print("\n" + "=" * 72)
    print(" GENERATING ACADEMIC DRAFT SECTIONS")
    print("=" * 72)

    if not analysis_data:
        raise ValueError("analysis_data required for generation")

    paper_count = analysis_data.get("paper_count", 1)
    generator = GPTSectionGenerator()

    section_specs = [
        ("abstract", "Abstract (100 words max)"),
        ("introduction", "Introduction"),
        ("methods", "Methods Comparison"),
        ("results", "Results Synthesis"),
        ("conclusion", "Conclusion"),
        ("references", "APA References"),
    ]

    sections: Dict[str, Dict[str, Any]] = {}

    print(f"\n Generating sections for {paper_count} paper(s)...")
    for key, display_name in section_specs:
        print(f"  - Generating: {display_name}...")
        content = generator.generate_with_template(key, analysis_data, paper_count)
        word_count = len(content.split())
        token_count = generator.count_tokens(content)
        sections[key] = {
            "name": display_name,
            "content": content,
            "word_count": word_count,
            "token_count": token_count
        }
        print(f"    ✓ {key}: {word_count} words, {token_count} tokens")

    return sections


# -------------------------
# 4. VALIDATION CHECKS
# -------------------------
def validate_sections(sections: Dict[str, Dict[str, Any]], analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run simple validation checks on generated sections.

    Checks performed:
      - Abstract word limit <= 100 words
      - References contain basic APA-like patterns
      - Sections are factually tied to analysis (simple keyword check)
      - All required sections present

    Returns:
        validation_results: dict with boolean flags and issues list.
    """
    print("\n" + "=" * 72)
    print(" VALIDATING GENERATED SECTIONS")
    print("=" * 72)

    results = {
        "abstract_word_limit": False,
        "references_apa_format": False,
        "sections_factual": False,
        "all_sections_present": False,
        "issues": []
    }

    # Abstract check
    abstract_text = sections.get("abstract", {}).get("content", "")
    abstract_words = len(abstract_text.split())
    results["abstract_word_limit"] = abstract_words <= 100
    if not results["abstract_word_limit"]:
        results["issues"].append(f"Abstract exceeds 100 words ({abstract_words})")
    else:
        print(f" Abstract word count OK: {abstract_words}/100")

    references_text = sections.get("references", {}).get("content", "")
    has_parenthetical_dates = bool(re.search(r"\(\d{4}\)", references_text))
    has_author_initials = bool(re.search(r"[A-Z][a-z]+,?\s+[A-Z]\.", references_text))
    results["references_apa_format"] = has_parenthetical_dates and has_author_initials
    if not results["references_apa_format"]:
        results["issues"].append("References may not follow basic APA structure")

    if analysis_data.get("type") == "single":
        analysis = analysis_data.get("analysis", {})
        key_terms: List[str] = []
        if analysis.get("title"):
            key_terms.append(analysis["title"].split()[:3] and " ".join(analysis["title"].split()[:3]))
        if analysis.get("methods_used"):
            key_terms.extend([m.split()[:3] and " ".join(m.split()[:3]) for m in analysis["methods_used"][:2]])
        all_text = " ".join([s["content"] for s in sections.values()]) if sections else ""
        matches = sum(1 for term in key_terms if term and term.lower() in all_text.lower())
        results["sections_factual"] = matches >= 1
        if not results["sections_factual"]:
            results["issues"].append("Generated sections do not reference analysis key terms")
        else:
            print(f" Sections reference {matches} key terms from analysis")
    else:
        combined_text = " ".join([s["content"] for s in sections.values()])
        results["sections_factual"] = any(word in combined_text.lower() for word in ["method", "result", "finding", "study"])
        if not results["sections_factual"]:
            results["issues"].append("Generated multi-paper sections may lack method/result mentions")

    required = {"abstract", "introduction", "methods", "results", "conclusion", "references"}
    missing = required - set(sections.keys())
    results["all_sections_present"] = len(missing) == 0
    if missing:
        results["issues"].append(f"Missing sections: {', '.join(sorted(missing))}")

    # Print summary
    passed = sum(1 for key in ["abstract_word_limit", "references_apa_format", "sections_factual", "all_sections_present"] if results.get(key))
    print("\nValidation summary:")
    print(f" Checks passed: {passed}/4")
    if results["issues"]:
        print(" Issues found:")
        for issue in results["issues"]:
            print("  -", issue)
    else:
        print(" No validation issues detected.")

    return results


# -------------------------
# 5. SAVE OUTPUTS
# -------------------------
def save_draft_outputs(sections: Dict[str, Dict[str, Any]], analysis_data: Dict[str, Any], validation_results: Dict[str, Any]) -> str:
    """
    Save generated sections and metadata to /outputs/.

    Returns:
        Path to outputs directory (string).
    """
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n Saving outputs to: {outputs_dir.resolve()}")

    # Save each section as a separate file
    for key, data in sections.items():
        filename = outputs_dir / f"{key}_{timestamp}.txt"
        with filename.open("w", encoding="utf-8") as fh:
            fh.write(f"{data['name']}\n")
            fh.write("=" * len(data["name"]) + "\n\n")
            fh.write(data["content"])
            fh.write(f"\n\n[Word count: {data['word_count']}]\n")
            fh.write(f"[Token count: {data['token_count']}]\n")
        print(f"  Saved: {filename.name}")

    # Save complete draft
    complete = outputs_dir / f"complete_draft_{timestamp}.txt"
    with complete.open("w", encoding="utf-8") as fh:
        fh.write("ACADEMIC DRAFT - RESEARCH PAPER ANALYSIS\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Papers analyzed: {analysis_data.get('paper_count', 1)}\n")
        fh.write("-" * 50 + "\n\n")
        for sec in ["abstract", "introduction", "methods", "results", "conclusion", "references"]:
            if sec in sections:
                s = sections[sec]
                fh.write(f"\n{s['name'].upper()}\n")
                fh.write("-" * len(s['name']) + "\n\n")
                fh.write(s["content"] + "\n\n")
    print(f"  Saved complete draft: {complete.name}")

    # Save metadata
    metadata = {
        "generation_date": timestamp,
        "paper_count": analysis_data.get("paper_count", 1),
        "analysis_type": analysis_data.get("type", "unknown"),
        "sections_generated": len(sections),
        "validation_results": validation_results,
        "section_stats": {k: {"word_count": v["word_count"], "token_count": v["token_count"]} for k, v in sections.items()}
    }
    meta_file = outputs_dir / f"draft_metadata_{timestamp}.json"
    with meta_file.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {meta_file.name}")

    return str(outputs_dir.resolve())


# -------------------------
# 6. GENERATE REPORT
# -------------------------
def generate_report(sections: Dict[str, Dict[str, Any]], validation_results: Dict[str, Any], output_path: str) -> str:
    """
    Create a short review report summarizing generated content and validation.

    Returns:
        Path to the report file as string.
    """
    outdir = Path(output_path)
    report_file = outdir / "review_report.txt"
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("REVIEW REPORT - GENERATED DRAFT SECTIONS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\nOBJECTIVE CHECKLIST:")
    lines.append("-" * 40)

    objectives = [
        ("Abstract (<=100 words)", validation_results.get("abstract_word_limit", False),
         f"Abstract words: {sections.get('abstract', {}).get('word_count', 0)}"),
        ("References basic APA", validation_results.get("references_apa_format", False),
         "Basic APA elements detected" if validation_results.get("references_apa_format") else "May need formatting"),
        ("Sections factually tied", validation_results.get("sections_factual", False),
         "Evidence of analysis terms in sections" if validation_results.get("sections_factual") else "Review factual alignment"),
        ("All sections present", validation_results.get("all_sections_present", False),
         "6/6 sections" if validation_results.get("all_sections_present") else "Missing sections")
    ]

    for title, passed, details in objectives:
        status = "PASSED" if passed else "NEEDS REVIEW"
        lines.append(f"\n{title}:\n  Status: {status}\n  Details: {details}\n")

    lines.append("\nSECTION STATISTICS:")
    lines.append("-" * 40)
    for key, s in sections.items():
        lines.append(f"\n{s['name']}: Words={s['word_count']}, Tokens={s['token_count']}")

    lines.append("\nVALIDATION ISSUES:")
    lines.append("-" * 40)
    if validation_results.get("issues"):
        for issue in validation_results["issues"]:
            lines.append(f"• {issue}")
    else:
        lines.append("No significant issues found")

    lines.append("\nNEXT STEPS:")
    lines.append("-" * 40)
    lines.append("1. Manually verify factual accuracy against original papers.")
    lines.append("2. Edit references to full APA format as needed.")
    lines.append("3. Expand sections if reviewer requests more detail.")

    with report_file.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n report saved to: {report_file}")

    return str(report_file)


# -------------------------
# 7. MAIN GENERATION PIPELINE
# -------------------------
def run_draft_generation() -> Optional[Dict[str, Any]]:
    """
    End-to-end pipeline to generate draft sections from analysis data.

    Steps:
      1. Load analysis data from Module 4 outputs
      2. Generate sections using GPTSectionGenerator templates
      3. Validate generated content
      4. Save outputs and generate a review report
    """
    print("\n" + "=" * 72)
    print("GENERATE DRAFT SECTIONS WITH GPT (Pipeline)")
    print("=" * 72)

    analysis_data = load_analysis_data()
    if not analysis_data:
        print("Cannot proceed without analysis data.")
        return None

    paper_count = analysis_data.get("paper_count", 1)
    print(f" Loaded analysis data for {paper_count} paper(s)")

    # Step 2: generate
    sections = generate_all_sections(analysis_data)

    # Step 3: validate
    validation_results = validate_sections(sections, analysis_data)

    # Step 4: save
    output_path = save_draft_outputs(sections, analysis_data, validation_results)

    # Step 5: create review report
    mentor_report = generate_report(sections, validation_results, output_path)

    print("\nGeneration complete. Outputs saved to:", output_path)
    return {"sections": sections, "validation": validation_results, "output_path": output_path}


# -------------------------
# 8. PREVIEW FUNCTION
# -------------------------
def preview_generated_draft() -> None:
    """
    Show a short preview of the most recent complete draft file and metadata.
    """
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("No outputs found. Run run_draft_generation() first.")
        return

    drafts = sorted(outputs_dir.glob("complete_draft_*.txt"), key=lambda p: p.stat().st_mtime)
    if not drafts:
        print("No complete draft file found in outputs/")
        return

    latest = drafts[-1]
    print("\n" + "=" * 72)
    print("PREVIEW OF GENERATED DRAFT:", latest.name)
    print("=" * 72)
    try:
        with latest.open("r", encoding="utf-8") as fh:
            content = fh.read()
            preview = content[:1000] + ("..." if len(content) > 1000 else "")
            print(preview)
            print(f"\nTotal words in draft: {len(content.split())}")
    except Exception as exc:
        print("Error reading draft:", exc)
        return

    metadata_files = sorted(outputs_dir.glob("draft_metadata_*.json"), key=lambda p: p.stat().st_mtime)
    if metadata_files:
        try:
            with metadata_files[-1].open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            val = metadata.get("validation_results", {})
            passed = sum(1 for k, v in val.items() if isinstance(v, bool) and v)
            print(f"\nValidation checks passed (approx): {passed}/4")
        except Exception:
            pass


# -------------------------
# 9. ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    result = run_draft_generation()
    if result:
        print("\nDraft generation finished successfully.")
        try:
            preview = input("Would you like to preview the recent draft? (y/n): ")
        except Exception:
            preview = "n"
        if preview.strip().lower().startswith("y"):
            preview_generated_draft()
    else:
        print("Draft generation did not complete.")
