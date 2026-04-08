#!/usr/bin/env python3
"""Pre-tool hook: run the test suite before a git commit lands.

Fires on PreToolUse for Bash. If the tool_input.command invokes
``git commit`` (as a real command, not embedded inside a quoted
argument), run pytest with -x. Non-zero pytest exit becomes exit 2,
which blocks the commit. Anything else exits 0 (allow).

Reads the Claude Code hook JSON payload from stdin.
"""

import json
import shlex
import subprocess
import sys


NEEDLE = ("git", "commit")


def invokes_git_commit(command: str) -> bool:
    """Return True iff ``command`` actually runs ``git commit``.

    Uses shlex so that quoted occurrences (e.g. inside a here-string or
    a Python one-liner) do not false-positive. Walks the token stream
    and treats shell separators (``;``, ``&&``, ``||``, ``|``) as
    command boundaries.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Unbalanced quotes — fall back to a conservative no-match.
        return False

    separators = {";", "&&", "||", "|", "&"}
    at_command_start = True
    prev = None
    for tok in tokens:
        if tok in separators:
            at_command_start = True
            prev = None
            continue
        if at_command_start and tok == NEEDLE[0]:
            prev = tok
            at_command_start = False
            continue
        if prev == NEEDLE[0] and tok == NEEDLE[1]:
            return True
        at_command_start = False
        prev = tok
    return False


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = hook_input.get("tool_input", {}).get("command", "")
    if not invokes_git_commit(command):
        return 0

    rc = subprocess.call(
        ["python", "-m", "pytest", "-x", "--tb=short", "-q"],
    )
    return 2 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
