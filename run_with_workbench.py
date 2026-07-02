from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from workbench_bootstrap import setup_workbench_environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a workbench script after configuring local package paths."
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_workbench_environment()

    script_path = args.script.resolve()
    sys.argv = [str(script_path), *args.script_args]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

