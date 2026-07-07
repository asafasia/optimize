from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def python_files() -> list[Path]:
    excluded_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "outputs",
        "data",
        "laboneq_output",
    }
    return [
        path
        for path in ROOT.rglob("*.py")
        if not excluded_parts.intersection(path.relative_to(ROOT).parts)
    ]


def test_optimizer_package_uses_root_import_style():
    offenders = []
    forbidden = "workbench" + ".optimize"
    for path in (ROOT / "optimize").rglob("*.py"):
        text = path.read_text()
        if forbidden in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_no_wildcard_resource_imports():
    offenders = []
    forbidden = {
        "from resources import " + "*",
        "from workbench.resources import " + "*",
    }
    for path in python_files():
        text = path.read_text()
        if any(pattern in text for pattern in forbidden):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_readout_amplitude_optimizer_does_not_run_hardware_workflows_as_script():
    text = (ROOT / "optimize/readout/readout_amplitude_optimizer.py").read_text()
    main_block = text.split('if __name__ == "__main__":', maxsplit=1)[1]

    assert "raise SystemExit" in main_block
    assert "workflow.run()" not in main_block
    assert "optimizer.run()" not in main_block
