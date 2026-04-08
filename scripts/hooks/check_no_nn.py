#!/usr/bin/env python3
"""Pre-tool hook: block edits that introduce neural network components.

Enforces the project's hard constraint: NO nn.Linear, nn.Module subclasses,
MLPs, or activation functions. The only allowed nn usage is the existing
linear output projection to vocabulary size.

Exit code 0 = allow, exit code 2 = block (PreToolUse convention).
Reads the Claude Code hook JSON payload from stdin.
"""

import json
import re
import sys

BANNED_PATTERNS = [
    r"\bnn\.Linear\b",
    r"\bnn\.Module\b",
    r"\bnn\.Sequential\b",
    r"\bnn\.ModuleList\b",
    r"\bnn\.GELU\b",
    r"\bnn\.ReLU\b",
    r"\bnn\.SiLU\b",
    r"\bnn\.Tanh\b",
    r"\bnn\.Sigmoid\b",
    r"\bnn\.LayerNorm\b",
    r"\bnn\.BatchNorm",
    r"\bnn\.Dropout\b",
    r"\bF\.gelu\b",
    r"\bF\.relu\b",
    r"\bF\.silu\b",
    r"\bclass\s+\w+\(nn\.Module\)",
]

# Files that legitimately reference nn for baseline comparisons or tests
EXEMPT_PATHS = [
    "baselines/",
    "tests/",
    "test_",
    "scripts/hooks/",
]


def check_content(content: str, file_path: str) -> list[str]:
    """Return list of violation descriptions found in content."""
    # Only gate Python source. Documentation, configs, notebooks, etc.
    # legitimately reference the banned symbols when describing the
    # constraint itself.
    if not file_path.endswith(".py"):
        return []
    if any(exempt in file_path for exempt in EXEMPT_PATHS):
        return []

    violations = []
    for pattern in BANNED_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            violations.append(f"  Banned pattern: {matches[0]}")
    return violations


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # can't parse, allow

    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # For Edit tool: check new_string
    # For Write tool: check content
    text_to_check = tool_input.get("new_string", "") or tool_input.get("content", "")
    if not text_to_check:
        return 0

    violations = check_content(text_to_check, file_path)
    if violations:
        print("BLOCKED: Neural network components detected!")
        print("This project prohibits nn.Linear, MLPs, and activation functions.")
        print("Violations found:")
        for v in violations:
            print(v)
        print(f"\nFile: {file_path}")
        print("See CLAUDE.md 'Hard Constraints' section.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
