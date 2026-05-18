"""Strip banned LaTeX spacing macros \;  \,  \!  from both manuscripts.

Project policy bans these per CLAUDE.md. We replace them with a single space
inside math mode (LaTeX normalises whitespace in math) and remove them
outright outside math; the simplest robust pass is global removal, which
LaTeX handles fine because math-mode tokens are space-insensitive.

We do NOT touch:
  - bib file
  - figure captions inside \cite{...}, \ref{...} (these macros never
    contain banned spacing tokens)
  - the verifier scripts in this directory
"""
import re
from pathlib import Path

TARGETS = [
    Path(r"C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\Attention\GL(K)_attention.tex"),
    Path(r"C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\Attention\GL(K)_supplementary.tex"),
]

# Banned tokens: \;  \,  \!
# Use a single regex that matches any of the three when followed by a non-letter
# (so we don't break a longer command like \!sup or \,foo - actually these are
# all standard LaTeX spacing macros that take no argument)
PATTERN = re.compile(r"\\[;,!]")

def strip(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    before = len(PATTERN.findall(text))
    # Replace with a single space; LaTeX will collapse runs of whitespace in
    # math mode, and in text mode this preserves word boundaries.
    new = PATTERN.sub(" ", text)
    # Collapse runs of >=2 spaces that we may have introduced inline (but not
    # at line starts, which preserves indentation)
    new = re.sub(r"(?<!\n)[ \t]{2,}", " ", new)
    path.write_text(new, encoding="utf-8")
    return before

if __name__ == "__main__":
    total = 0
    for p in TARGETS:
        n = strip(p)
        print(f"{p.name}: removed {n} occurrences")
        total += n
    print(f"Total removed: {total}")
