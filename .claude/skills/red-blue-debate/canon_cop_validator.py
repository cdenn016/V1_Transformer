"""Canon-cop grep validator for the red-blue-debate skill.

Mechanical pass: scans a target file (opening, rebuttal, or sur-rebuttal) for
patterns that fire a source-of-truth strike per the debate methodology.

Outputs JSON to stdout with strike counts and concrete line references. The
calling agent (debate-canon-cop) reads the JSON, runs a follow-up LLM pass for
subtle phrasing the grep misses, and applies the soft-cap rule.

This is NOT the full validator — it only catches the mechanical patterns. The
LLM pass owned by the debate-canon-cop agent catches the subtle ones.

Usage:
    python canon_cop_validator.py \\
        --target docs/debates/<slug>/02_red_opening.md \\
        --bibliography .claude/agents/vfe-knowledge/external_bibliography.md \\
        --canon-dir .claude/agents/vfe-knowledge/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


MANUSCRIPT_AUTHORITY_PATTERNS = [
    (r"as\s+shown\s+in\s+`?Attention/", "Attention-as-authority"),
    (r"per\s+CLAUDE\.md", "CLAUDE.md-as-authority"),
    (r"as\s+shown\s+in\s+CLAUDE\.md", "CLAUDE.md-as-authority"),
    (r"the\s+manuscript\s+establishes", "manuscript-establishes"),
    (r"as\s+the\s+(project|manuscript|framework|paper)\s+(defines|establishes|derives|shows|states)", "framework-establishes"),
    (r"by\s+construction\s+in\s+(our|this)\s+(framework|work|paper|project)", "by-construction-circularity"),
    (r"as\s+established\s+by\s+our\s+(framework|construction|derivation)", "framework-as-authority"),
    (r"the\s+user'?s?\s+(framework|construction|derivation)\s+(shows|establishes|proves)", "user-framework-as-authority"),
    (r"per\s+user_theory_summary\.md", "user_theory_summary-as-authority"),
]

MANUSCRIPT_AUTHORITY_STRIKE = 1


CITATION_RE = re.compile(r"\[([A-Z][A-Za-z]+(?:&[A-Z][A-Za-z]+)?)\s+(\d{4})(?:\s+(§?[\w\.\-]+))?\]")


@dataclass
class Hit:
    line: int
    snippet: str
    pattern_id: str
    strikes: int


@dataclass
class CitationCheck:
    line: int
    citation: str
    author: str
    year: str
    section: str | None
    verified: bool
    note: str = ""

    @property
    def strikes(self) -> int:
        return 0 if self.verified else 2


@dataclass
class Report:
    target: str
    total_strikes: int
    action: str  # "RECORD" or "MANDATORY_REWRITE"
    manuscript_authority_hits: list[Hit] = field(default_factory=list)
    citation_checks: list[CitationCheck] = field(default_factory=list)
    attention_citation_count: int = 0
    claude_md_citation_count: int = 0
    external_citation_count: int = 0


def load_bibliography_keys(bib_path: Path) -> set[str]:
    """Load author-year keys from the external bibliography file.

    The bibliography is a markdown file with entries like
        - Nakahara, M. (2003). *Geometry, Topology and Physics* (2nd ed.).
        - Friston, K. (2010). The free-energy principle. ...

    We extract (LastName, Year) tuples and store them as canonical keys
    like "Nakahara2003", "Friston2010".
    """
    keys: set[str] = set()
    if not bib_path.exists():
        return keys

    text = bib_path.read_text(encoding="utf-8", errors="replace")

    # Match "LastName, Initial(s)." or "LastName et al." followed by "(YYYY)"
    entry_re = re.compile(r"\b([A-Z][A-Za-z]+)(?:\s*&\s*[A-Z][A-Za-z]+|\s+et\s+al\.)?\s*,?\s*[A-Z]\.?(?:\s*[A-Z]\.?)*\s*\(?(\d{4})\)?")

    for match in entry_re.finditer(text):
        last_name, year = match.group(1), match.group(2)
        keys.add(f"{last_name}{year}")

    return keys


def scan_manuscript_authority(target_text: str) -> list[Hit]:
    """Scan for manuscript-as-authority patterns."""
    hits: list[Hit] = []
    for idx, line in enumerate(target_text.splitlines(), start=1):
        for pattern, pattern_id in MANUSCRIPT_AUTHORITY_PATTERNS:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                hits.append(
                    Hit(
                        line=idx,
                        snippet=line.strip()[:200],
                        pattern_id=pattern_id,
                        strikes=MANUSCRIPT_AUTHORITY_STRIKE,
                    )
                )
                break  # one strike per pattern per line
    return hits


def scan_citations(target_text: str, bib_keys: set[str]) -> tuple[list[CitationCheck], dict[str, int]]:
    """Scan citation patterns and verify against bibliography.

    Returns (citation checks, citation-source counts).
    """
    checks: list[CitationCheck] = []
    counts = {"attention": 0, "claude_md": 0, "external": 0}

    for idx, line in enumerate(target_text.splitlines(), start=1):
        # Count attention/CLAUDE.md citations (raw — context-blind)
        counts["attention"] += len(re.findall(r"Attention/[A-Za-z_]+\.tex", line))
        counts["claude_md"] += len(re.findall(r"\bCLAUDE\.md\b", line))

        # External canon citations
        for match in CITATION_RE.finditer(line):
            author, year, section = match.group(1), match.group(2), match.group(3)
            key = f"{author}{year}"
            verified = key in bib_keys
            note = "" if verified else f"key '{key}' not found in external_bibliography.md"
            checks.append(
                CitationCheck(
                    line=idx,
                    citation=match.group(0),
                    author=author,
                    year=year,
                    section=section,
                    verified=verified,
                    note=note,
                )
            )
            counts["external"] += 1

    return checks, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Canon-cop grep validator")
    parser.add_argument("--target", type=Path, required=True, help="The opening/rebuttal/surrebuttal file to validate")
    parser.add_argument("--bibliography", type=Path, required=True, help="external_bibliography.md")
    parser.add_argument("--canon-dir", type=Path, required=True, help="Directory containing external_canon_*.md")
    args = parser.parse_args()

    if not args.target.exists():
        print(json.dumps({"error": f"target file not found: {args.target}"}), file=sys.stderr)
        return 2

    target_text = args.target.read_text(encoding="utf-8", errors="replace")
    bib_keys = load_bibliography_keys(args.bibliography)

    manuscript_hits = scan_manuscript_authority(target_text)
    citation_checks, counts = scan_citations(target_text, bib_keys)

    total_strikes = sum(h.strikes for h in manuscript_hits) + sum(c.strikes for c in citation_checks)
    action = "MANDATORY_REWRITE" if total_strikes >= 3 else "RECORD"

    report = Report(
        target=str(args.target),
        total_strikes=total_strikes,
        action=action,
        manuscript_authority_hits=manuscript_hits,
        citation_checks=citation_checks,
        attention_citation_count=counts["attention"],
        claude_md_citation_count=counts["claude_md"],
        external_citation_count=counts["external"],
    )

    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
