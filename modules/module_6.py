# ============================================
# MODULE 6: DRAFT AGGREGATION & CRITIQUE (FIXED)
# ============================================

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple


# -------------------------
# 1. LOAD GENERATED DRAFT
# -------------------------
def load_latest_draft(outputs_dir: str = "outputs") -> Optional[str]:
    """Load the most recent aggregated complete draft text."""
    out_path = Path(outputs_dir)
    if not out_path.exists():
        print("No outputs found. Run Module 5 first.")
        return None

    draft_files = sorted(out_path.glob("complete_draft_*.txt"), key=lambda p: p.stat().st_mtime)
    if not draft_files:
        print("No complete draft found in outputs/")
        return None

    latest = draft_files[-1]
    try:
        with latest.open("r", encoding="utf-8") as fh:
            content = fh.read()
        print(f"Loaded draft: {latest.name}")
        return content
    except Exception as exc:
        print(f"Error reading draft {latest.name}: {exc}")
        return None


def load_individual_sections(outputs_dir: str = "outputs") -> Dict[str, str]:
    """Load the latest individual section files."""
    out_path = Path(outputs_dir)
    if not out_path.exists():
        return {}

    section_patterns = {
        "abstract": "abstract_*.txt",
        "introduction": "introduction_*.txt",
        "methods": "methods_*.txt",
        "results": "results_*.txt",
        "conclusion": "conclusion_*.txt",
        "references": "references_*.txt",
    }

    sections: Dict[str, str] = {}
    for key, pattern in section_patterns.items():
        files = list(out_path.glob(pattern))
        if not files:
            continue
        latest = max(files, key=lambda p: p.stat().st_mtime)
        try:
            with latest.open("r", encoding="utf-8") as fh:
                content = fh.read()
            lines = content.splitlines()
            section_content = "\n".join(lines[2:]) if len(lines) > 2 else content
            sections[key] = section_content.strip()
        except Exception as exc:
            print(f"Error reading section file {latest.name}: {exc}")
            continue

    return sections


# -------------------------
# 2. AGGREGATE FULL DRAFT
# -------------------------
def create_full_draft_markdown(
    sections: Dict[str, str],
    critique_feedback: Optional[Dict[str, Any]] = None,
    title: str = "Research Paper Analysis Review",
) -> str:
    """Combine individual sections into a polished markdown draft."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"*Generated: {now}*")
    lines.append(f"*Status: {'Revised' if critique_feedback else 'Initial'} Draft*")
    lines.append("\n---\n")

    section_order = ["abstract", "introduction", "methods", "results", "conclusion", "references"]
    for key in section_order:
        content = sections.get(key)
        if content:
            lines.append(f"\n## {key.upper()}\n")
            lines.append(content)
            lines.append("\n---\n")

    if critique_feedback:
        lines.append("\n## Critique & Revision Notes\n")
        lines.append("### Issues Identified:\n")
        issues_found = False
        checks = critique_feedback.get("checks", {})
        for check_type, check_data in checks.items():
            passed = bool(check_data.get("passed", False))
            suggestion = check_data.get("suggestion", "")
            if not passed:
                issues_found = True
                lines.append(f"- **{check_type.replace('_', ' ').title()}**: {suggestion}")

        if not issues_found:
            lines.append("No major issues identified. Draft is well-structured.")

        lines.append("\n### Suggested Revisions:\n")
        for suggestion in critique_feedback.get("suggestions", [])[:5]:
            lines.append(f"- {suggestion}")

    full_text = "\n".join(lines)
    word_count = len(re.findall(r"\b\w+\b", full_text))
    lines.append(f"\n\n*Word count: {word_count}*")

    return "\n".join(lines)


# -------------------------
# 3. CRITIQUE SYSTEM
# -------------------------
class DraftCritique:
    """Critique system that analyzes the draft."""

    def __init__(self) -> None:
        self.criteria = {
            "clarity": self.check_clarity,
            "flow": self.check_flow,
            "missing_references": self.check_missing_references,
            "repetition": self.check_repetition,
            "style": self.check_academic_style,
            "structure": self.check_structure,
        }

    def critique_draft(self, draft_text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Run the full critique over the draft."""
        print("Analyzing draft quality...")

        total_checks = len(self.criteria)
        critique_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "suggestions": [],
            "score": 0,
            "total_checks": total_checks,
        }

        passed_checks = 0
        for name, func in self.criteria.items():
            print(f"  • Checking {name}...", end=" ")
            try:
                passed, feedback = func(draft_text, sections)
            except Exception as exc:
                passed = False
                feedback = [f"Error running check: {exc}"]

            critique_results["checks"][name] = {
                "passed": passed,
                "feedback": feedback,
                "suggestion": self.generate_suggestion(name, passed, feedback),
            }

            if passed:
                passed_checks += 1
                print("✅")
            else:
                print("❌")

        critique_results["score"] = passed_checks
        critique_results["passed_checks"] = passed_checks
        critique_results["suggestions"] = self.generate_overall_suggestions(critique_results)

        return critique_results

    # ---------- individual checks ----------
    def check_clarity(self, draft_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        sentences = [s.strip() for s in re.split(r"[.!?]+", draft_text) if s.strip()]
        long_sentences = [s for s in sentences if len(s.split()) > 40]
        if long_sentences:
            issues.append(f"{len(long_sentences)} sentences are very long (>40 words)")

        passive_patterns = [r"\b(is|are|was|were)\s+\w+ed\b", r"\bbe\s+\w+ed\b"]
        passive_count = sum(len(re.findall(pat, draft_text.lower())) for pat in passive_patterns)
        if passive_count > 10:
            issues.append(f"High use of passive voice ({passive_count} instances)")

        passed = len(issues) == 0
        return passed, issues

    def check_flow(self, draft_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        required_order = ["abstract", "introduction", "methods", "results", "conclusion"]
        missing = [s for s in required_order if s not in sections]
        if missing:
            issues.append(f"Missing sections: {', '.join(missing)}")

        if "conclusion" in sections and "introduction" in sections:
            intro_keywords = ["paper", "study", "research", "analysis"]
            conclusion_text = sections.get("conclusion", "").lower()
            if not any(k in conclusion_text for k in intro_keywords):
                issues.append("Conclusion does not clearly reference introduction/key aims")

        passed = len(issues) == 0
        return passed, issues

    def check_missing_references(self, draft_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if "references" in sections:
            ref_text = sections["references"]
            has_years = bool(re.search(r"\(\d{4}\)", ref_text))
            has_authors = bool(re.search(r"[A-Z][a-z]+,\s+[A-Z]\.", ref_text))
            if not has_years:
                issues.append("References missing publication years")
            if not has_authors:
                issues.append("References may be missing author names or initials")

            ref_lines = [line for line in ref_text.splitlines() if line.strip()]
            if len(ref_lines) < 3:
                issues.append(f"Only {len(ref_lines)} references found (suggest 5+ for review)")
        else:
            issues.append("No references section found")

        passed = len(issues) == 0
        return passed, issues

    def check_repetition(self, draft_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        words = re.findall(r"\b\w+\b", draft_text.lower())
        freq: Dict[str, int] = defaultdict(int)
        for w in words:
            if len(w) > 4:
                freq[w] += 1

        common_exclude = {"paper", "study", "research", "analysis", "method", "result", "finding"}
        overused = sorted([(w, c) for w, c in freq.items() if c > 5 and w not in common_exclude], key=lambda x: -x[1])
        if overused:
            top = overused[:3]
            issues.append("Overused words: " + ", ".join([f"{w}({c})" for w, c in top]))

        section_texts = list(sections.values())
        for i in range(len(section_texts)):
            for j in range(i + 1, len(section_texts)):
                a_start = section_texts[i][:50].strip()
                b_start = section_texts[j][:50].strip()
                if a_start and a_start == b_start:
                    issues.append("Two sections share the same opening text — possible duplication")
                    break

        passed = len(issues) == 0
        return passed, issues

    def check_academic_style(self, draft_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        informal_words = ["really", "very", "a lot", "got", "stuff", "thing"]
        informal_count = sum(draft_text.lower().count(w) for w in informal_words)
        if informal_count > 3:
            issues.append(f"Informal language used ({informal_count} instances)")

        first_person_matches = re.findall(r"\b(I|we|our|us|my|mine)\b", draft_text, flags=re.I)
        if len(first_person_matches) > 5:
            issues.append(f"High use of first-person pronouns ({len(first_person_matches)})")

        paragraphs = [p for p in draft_text.split("\n\n") if p.strip()]
        short_paragraphs = [p for p in paragraphs if len(p.split()) < 50]
        if len(short_paragraphs) > 3:
            issues.append(f"{len(short_paragraphs)} very short paragraphs (<50 words)")

        passed = len(issues) == 0
        return passed, issues

    def check_structure(self, draft_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        required = {"abstract", "introduction", "methods", "results", "conclusion", "references"}
        present = set(sections.keys())
        missing = required - present
        if missing:
            issues.append(f"Missing required sections: {', '.join(sorted(missing))}")

        for sec, content in sections.items():
            words = len(re.findall(r"\b\w+\b", content))
            if sec == "abstract" and words > 150:
                issues.append(f"Abstract too long ({words} words; aim for <150)")
            if sec == "introduction" and words < 100:
                issues.append(f"Introduction too short ({words} words; aim for 100+)")

        if "methods" in sections and "results" in sections and "conclusion" in sections:
            m_len = len(re.findall(r"\b\w+\b", sections.get("methods", "")))
            r_len = len(re.findall(r"\b\w+\b", sections.get("results", "")))
            c_len = len(re.findall(r"\b\w+\b", sections.get("conclusion", "")))
            if c_len > (m_len + r_len):
                issues.append("Conclusion unusually long compared to Methods and Results")

        passed = len(issues) == 0
        return passed, issues

    # ---------- suggestion helpers ----------
    def generate_suggestion(self, criterion: str, passed: bool, feedback: List[str]) -> str:
        base_recs = {
            "clarity": "Use shorter sentences and prefer active voice.",
            "flow": "Add clear transitions and ensure the conclusion ties back to the introduction.",
            "missing_references": "Expand and format the references list (use APA).",
            "repetition": "Vary vocabulary and rephrase repeated phrases.",
            "style": "Avoid informal words and excessive first-person pronouns.",
            "structure": "Ensure all required sections are present and balanced in length.",
        }
        if passed:
            return f"{criterion.title()} is fine."
        suggestion = base_recs.get(criterion, "Review and improve this area.")
        if feedback:
            return f"{suggestion} Issues: {'; '.join(feedback[:3])}"
        return suggestion

    def generate_overall_suggestions(self, critique_results: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []
        failed = [name for name, c in critique_results.get("checks", {}).items() if not c.get("passed")]
        if not failed:
            suggestions.append("Draft is well-structured. Minor polishing only needed.")
            suggestions.append("Check references formatting before submission.")
            return suggestions

        if "clarity" in failed:
            suggestions.append("Revise long sentences; convert passive voice to active where appropriate.")
        if "flow" in failed:
            suggestions.append("Add transition sentences between sections; make conclusion reference introduction.")
        if "missing_references" in failed:
            suggestions.append("Add 2-3 more relevant citations and ensure APA format.")
        if "repetition" in failed:
            suggestions.append("Replace frequently repeated words with synonyms.")
        if "style" in failed:
            suggestions.append("Eliminate informal language; reduce first-person pronouns.")
        if "structure" in failed:
            suggestions.append("Ensure all sections are present and appropriately balanced by length.")

        suggestions.append("Read the draft aloud to catch awkward phrasing.")
        suggestions.append("Have a peer review for factual accuracy.")
        return suggestions[:7]


# -------------------------
# 4. REVISION CYCLE
# -------------------------
def run_revision_cycle(
    draft_text: str,
    sections: Dict[str, str],
    critique_results: Dict[str, Any],
    iteration: int = 1,
) -> Tuple[str, Dict[str, str]]:
    print(f"Running revision cycle {iteration}...")
    revised = dict(sections)

    checks = critique_results.get("checks", {})
    for criterion, info in checks.items():
        if not info.get("passed"):
            feedback = info.get("feedback", [])
            revised = apply_revisions(revised, criterion, feedback)

    revised_draft = create_full_draft_markdown(revised, critique_results)
    return revised_draft, revised


def apply_revisions(sections: Dict[str, str], criterion: str, feedback: List[str]) -> Dict[str, str]:
    revised = dict(sections) 

    if criterion == "clarity":
        for name, content in list(revised.items()):
            sentences = re.split(r'(?<=[.!?])\s+', content)
            new_sentences: List[str] = []
            for s in sentences:
                words = s.split()
                if len(words) > 60:
                    mid = len(words) // 2
                    new_sentences.append(" ".join(words[:mid]) + ".")
                    new_sentences.append(" ".join(words[mid:]) + ".")
                else:
                    new_sentences.append(s)
            revised[name] = " ".join(new_sentences).strip()

    elif criterion == "repetition":
        replacements = {
            "paper": "study", "research": "investigation", "analysis": "examination",
            "method": "approach", "result": "finding"
        }
        for name in ("abstract", "conclusion"):
            if name in revised:
                content = revised[name]
                for old, new in replacements.items():
                    if content.lower().count(old) > 2:
                        content = re.sub(rf'\b{old}\b', new, content, count=2, flags=re.I)
                revised[name] = content

    elif criterion == "structure":
        intro = revised.get("introduction", "")
        intro_words = len(re.findall(r"\b\w+\b", intro))
        if intro and intro_words < 100:
            addition = (" This analysis provides a more detailed examination of the methodological "
                        "approaches and findings. The review situates the work within the broader "
                        "research context and evaluates implications and potential improvements.")
            revised["introduction"] = (intro + " " + addition).strip()

    elif criterion == "missing_references":
        if "references" in revised:
            if "PLEASE_ADD_FULL_REFERENCES" not in revised["references"]:
                revised["references"] = revised["references"].strip() + "\n\n[PLEASE_ADD_FULL_REFERENCES]"
        else:
            revised["references"] = "[PLEASE_ADD_FULL_REFERENCES]"

    elif criterion == "flow":
        if "conclusion" in revised and "introduction" in revised:
            cons = revised["conclusion"]
            trans = "In line with the introduction, this conclusion revisits the main aims and synthesizes the outcomes."
            if trans not in cons:
                revised["conclusion"] = trans + "\n\n" + cons

    elif criterion == "style":
        for name, content in list(revised.items()):
            for informal in [" really ", " a lot ", " got ", " stuff ", " thing "]:
                if informal in content:
                    content = content.replace(informal, " ")
            revised[name] = content

    return revised


# -------------------------
# 5. SAVE OUTPUTS
# -------------------------
def save_critique_results(critique_results: Dict[str, Any], outputs_dir: str = "outputs", iteration: int = 1) -> str:
    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"critique_feedback_iteration_{iteration}.json"
    try:
        with filename.open("w", encoding="utf-8") as fh:
            json.dump(critique_results, fh, indent=2, ensure_ascii=False)
        print(f"Critique feedback saved: {filename.name}")
        return str(filename)
    except Exception as exc:
        print(f"Error saving critique feedback: {exc}")
        return ""


def save_revised_draft(revised_draft: str, outputs_dir: str = "outputs", iteration: int = 1) -> str:
    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / f"revised_draft_iteration_{iteration}.md"
    txt_file = out_path / f"revised_draft_iteration_{iteration}.txt"

    try:
        with md_file.open("w", encoding="utf-8") as fh:
            fh.write(revised_draft)
        with txt_file.open("w", encoding="utf-8") as fh:
            fh.write(revised_draft)
        print(f"Revised draft saved: {md_file.name}")
        return str(md_file)
    except Exception as exc:
        print(f"Error saving revised draft: {exc}")
        return ""


def save_revision_summary(original_critique: Dict[str, Any], revised_critique: Optional[Dict[str, Any]], iterations: int, outputs_dir: str = "outputs") -> str:
    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "revision_date": datetime.now().isoformat(),
        "total_iterations": iterations,
        "improvement_summary": {
            "original_score": original_critique.get("score", 0),
            "final_score": revised_critique.get("score", 0) if revised_critique else original_critique.get("score", 0),
            "improvement": (revised_critique.get("score", 0) - original_critique.get("score", 0)) if revised_critique else 0,
        },
        "issues_resolved": [],
        "remaining_issues": [],
    }

    if revised_critique:
        for crit in original_critique.get("checks", {}):
            orig_passed = bool(original_critique["checks"][crit].get("passed", False))
            new_passed = bool(revised_critique["checks"].get(crit, {}).get("passed", False))
            if not orig_passed and new_passed:
                summary["issues_resolved"].append(crit)
            elif not orig_passed and not new_passed:
                summary["remaining_issues"].append(crit)

    filename = out_path / "revision_summary.json"
    try:
        with filename.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"Revision summary saved: {filename.name}")
        return str(filename)
    except Exception as exc:
        print(f"Error saving revision summary: {exc}")
        return ""


# -------------------------
# 6. MAIN PIPELINE (RENAMED TO MATCH APP.PY)
# -------------------------
def run_refinement_cycle(iterations: int = 2) -> Optional[Dict[str, Any]]:
    """
    Main entry point for Module 6.
    Renamed to 'run_refinement_cycle' to match app.py.
    """
    print("\n" + "=" * 72)
    print("DRAFT AGGREGATION & CRITIQUE MODULE")
    print("=" * 72)

    outputs_dir = "outputs"
    
    print("STEP 1: Loading generated draft...")
    draft_text = load_latest_draft(outputs_dir)
    if not draft_text:
        return None

    print("Loading individual sections...")
    sections = load_individual_sections(outputs_dir)
    print(f"  Loaded {len(sections)} sections")

    print("STEP 2: Creating full markdown draft...")
    full_draft = create_full_draft_markdown(sections)

    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    initial_file = out_path / "full_draft_initial.md"
    try:
        with initial_file.open("w", encoding="utf-8") as fh:
            fh.write(full_draft)
        print(f"Full draft saved: {initial_file.name}")
    except Exception as exc:
        print(f"Error saving initial draft: {exc}")

    print("STEP 3: Running draft critique...")
    critic = DraftCritique()
    critique_results = critic.critique_draft(full_draft, sections)
    print(f"Critique Score: {critique_results['score']}/{critique_results['total_checks']}")

    save_critique_results(critique_results, outputs_dir, iteration=1)

    current_sections = sections
    current_critique = critique_results
    revised_files: List[str] = []
    revised_critique: Optional[Dict[str, Any]] = None

    for iteration in range(1, iterations + 1):
        print(f"\nIteration {iteration}/{iterations}")
        revised_draft, revised_sections = run_revision_cycle(full_draft, current_sections, current_critique, iteration)
        saved_path = save_revised_draft(revised_draft, outputs_dir, iteration)
        revised_files.append(saved_path)

        revised_critique = critic.critique_draft(revised_draft, revised_sections)
        save_critique_results(revised_critique, outputs_dir, iteration + 1)

        current_sections = revised_sections
        current_critique = revised_critique
        full_draft = revised_draft
        print(f"  Score after revision {iteration}: {revised_critique['score']}/{revised_critique['total_checks']}")

    print("\nSTEP 5: Creating revision summary...")
    summary_file = save_revision_summary(critique_results, revised_critique, iterations, outputs_dir)

    print("\n" + "=" * 72)
    print("COMPLETE!")
    print("=" * 72)

    return {
        "initial_draft_file": str(initial_file),
        "initial_critique": critique_results,
        "revised_files": revised_files,
        "final_critique": revised_critique,
        "revision_summary_file": summary_file,
    }


if __name__ == "__main__":
    run_refinement_cycle(iterations=2)
