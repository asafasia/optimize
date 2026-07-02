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
WORKBENCH_QRATENA_DATA_ROOT = WORKBENCH_ROOT / "data"
WORKBENCH_MPLCONFIG_ROOT = WORKBENCH_ROOT / ".cache" / "matplotlib"
WORKBENCH_KERNEL_TRACES_ROOT = WORKBENCH_ROOT / "outputs" / "qratena" / "qratena_kernel_traces"
QRATENA_NINJA_PROFILE = Path("devices/ninja_chip/profile.json")


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _parse_env_value(value: str) -> str:
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_workbench_dotenv(path: Path = WORKBENCH_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs without overriding the existing environment."""
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or key in os.environ:
            continue
        os.environ[key] = _parse_env_value(value)


def ensure_workbench_qratena_data_root() -> None:
    WORKBENCH_QRATENA_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    link_workbench_data_dir("devices", QRATENA_DATA_ROOT / "devices")
    link_workbench_data_dir("qratena_kernel_traces", WORKBENCH_KERNEL_TRACES_ROOT)


def link_workbench_data_dir(name: str, target: Path) -> None:
    link = WORKBENCH_QRATENA_DATA_ROOT / name
    if link.exists() or link.is_symlink():
        return
    if target.exists():
        link.symlink_to(target, target_is_directory=True)


def setup_workbench_environment() -> None:
    """Make local package checkouts and qratena data visible to experiments."""
    load_workbench_dotenv()
    ensure_workbench_qratena_data_root()

    for path in (*DEPENDENCY_ROOTS, PROJECT_ROOT, WORKBENCH_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    if os.environ.get("QRATENA_DATA_DIR") is None:
        os.environ["QRATENA_DATA_DIR"] = str(WORKBENCH_QRATENA_DATA_ROOT)
    if os.environ.get("MPLCONFIGDIR") is None:
        WORKBENCH_MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(WORKBENCH_MPLCONFIG_ROOT)
