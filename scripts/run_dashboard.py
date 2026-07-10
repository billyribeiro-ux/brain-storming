#!/usr/bin/env python3
"""Launch the Aether Streamlit cockpit.

Usage:
    python scripts/run_dashboard.py [--port 8501]

Thin wrapper around ``streamlit run aether/dashboard/app.py`` that pins the
working directory to the repo root and makes the ``aether`` package importable
regardless of where the script is invoked from.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # sys.path bootstrap
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Aether dashboard.")
    parser.add_argument("--port", type=int, default=8501,
                        help="Streamlit server port (default: 8501)")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        "streamlit", "run",
        str(REPO_ROOT / "aether" / "dashboard" / "app.py"),
        "--server.port", str(args.port),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
