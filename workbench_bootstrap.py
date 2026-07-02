from __future__ import annotations

import os
import sys
from pathlib import Path


WORKBENCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKBENCH_ROOT.parent
DEPENDENCY_ROOTS = (
    PROJECT_ROOT / "qratena",
    PROJECT_ROOT / "qigeon",
    PROJECT_ROOT / "qhipu-lab",
    PROJECT_ROOT / "q-b2c",
)

QRATENA_DATA_ROOT = PROJECT_ROOT / "qratena" / "qratena" / "data"
QRATENA_NINJA_PROFILE = Path("devices/ninja_chip/profile.json")


def setup_workbench_environment() -> None:
    """Make local package checkouts and qratena data visible to experiments."""
    for path in (*DEPENDENCY_ROOTS, PROJECT_ROOT, WORKBENCH_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    if os.environ.get("QRATENA_DATA_DIR") is None:
        os.environ["QRATENA_DATA_DIR"] = str(QRATENA_DATA_ROOT)
