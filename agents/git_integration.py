"""Conservative local Git integration for the engineer agent.

Autocommit is intentionally opt-in: a demo should never capture unrelated work
from a developer's checkout.  When enabled, only the explicitly listed files
are staged and committed with a local-only identity.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def propose_or_commit(case_id: str, changed_files: list[Path]) -> dict[str, Any]:
    message = f"fix(ml): recover fraud model after {case_id}"
    files = [path.relative_to(ROOT).as_posix() for path in changed_files]
    command = f'git add {" ".join(files)} && git commit -m "{message}"'
    if os.environ.get("SHERLOCKML_AUTOCOMMIT") != "1":
        return {
            "mode": "proposed",
            "message": message,
            "command": command,
            "sha": None,
            "note": "Set SHERLOCKML_AUTOCOMMIT=1 to create this local commit.",
        }
    if not (ROOT / ".git").exists():
        return {
            "mode": "unavailable",
            "message": message,
            "command": command,
            "sha": None,
            "note": "No local Git repository is available.",
        }

    _run(["git", "config", "user.name", "SherlockML Autonomous Engineer"])
    _run(["git", "config", "user.email", "sherlockml@local.invalid"])
    added = _run(["git", "add", "-f", "--", *files], allow_failure=True)
    if added.returncode != 0:
        return {
            "mode": "failed",
            "message": message,
            "command": command,
            "sha": None,
            "note": added.stdout.strip() or added.stderr.strip(),
        }
    committed = _run(["git", "commit", "--only", "-m", message, "--", *files], allow_failure=True)
    if committed.returncode != 0 and "nothing to commit" not in committed.stdout.lower():
        return {
            "mode": "failed",
            "message": message,
            "command": command,
            "sha": None,
            "note": committed.stdout.strip() or committed.stderr.strip(),
        }
    sha = _run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    return {
        "mode": "committed",
        "message": message,
        "command": command,
        "sha": sha,
        "note": "Committed only the controlled repair artifacts to this local repository.",
    }


def _run(command: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=not allow_failure,
        text=True,
    )
