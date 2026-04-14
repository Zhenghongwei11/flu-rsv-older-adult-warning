#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "run_all.sh"
    if not script.exists():
        raise SystemExit(f"Missing entrypoint: {script}")
    proc = subprocess.run([str(script), *argv], check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

