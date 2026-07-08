from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[4]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType

from optimize.readout.optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.optimizer.scan_types import ReadoutScanMethod
from optimize.readout.optimizer.artifacts import load_readout_task_manifest
from resources.load_profile import load_profile, load_task_manager


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_key, args.output_root)
    metadata = load_metadata(run_dir)
    manifest = load_readout_task_manifest(run_dir)
    qubit_names = list(manifest["qubits"])
    workflow_settings = workflow_settings_from_metadata(metadata)

    profile = load_profile(args.profile_name or workflow_settings.profile_name)
    task_manager = load_task_manager()
    optimizer = ReadoutAmplitudeSweepWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=ReadoutAmplitudeSweepSettings(
            amplitudes=[],
            method=ReadoutScanMethod.SWEEP,
            auto_save_results=False,
            use_live_html_plotter=False,
            workflow_settings=workflow_settings,
        ),
    )

    summary = optimizer.check_submitted_results(run_dir)
    print(summary["message"])
    print(f"Task counts: {summary['counts']}")
    print_task_keys("Pending task keys", summary["pending_task_keys"])
    print_task_keys("Failed/cancelled task keys", summary["failed_task_keys"])

    if summary["failed"]:
        return
    if not args.wait and not summary["ready_to_collect"]:
        return

    result = optimizer.collect_submitted_results(
        run_dir,
        save_results=True,
        wait=True,
    )

    if isinstance(result, dict) and "ready_to_collect" in result:
        print(result["message"])
    else:
        print(f"Collected and saved optimizer results to {run_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect saved task-manager results for a submitted optimizer run."
    )
    parser.add_argument("run_key", help="Run folder name or full run folder path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "readout_optimize",
        help="Root to search when run_key is not a full path.",
    )
    parser.add_argument(
        "--profile-name",
        help="Override metadata profile name when loading the profile.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for and collect results even if check_submitted_results is not ready.",
    )
    return parser.parse_args()


def resolve_run_dir(run_key: str, output_root: Path) -> Path:
    candidate = Path(run_key).expanduser()
    if candidate.exists():
        return candidate.resolve()

    matches = [
        manifest_path.parent
        for manifest_path in output_root.expanduser().rglob("task_manifest.json")
        if manifest_path.parent.name == run_key
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Could not find run folder {run_key!r} under {output_root}."
        )

    joined = "\n".join(str(path) for path in matches)
    raise RuntimeError(
        f"Run key {run_key!r} matched multiple folders:\n{joined}\n"
        "Use the full run folder path."
    )


def load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def workflow_settings_from_metadata(
    metadata: dict[str, Any],
) -> ReadoutFidelityWorkflowSettings:
    saved = metadata.get("workflow_settings") or {}
    return ReadoutFidelityWorkflowSettings(
        profile_name=saved.get("profile_name", "main"),
        do_emulation=False,
        run_resonator=bool(saved.get("run_resonator", False)),
        run_kernels=bool(saved.get("run_kernels", True)),
        run_iq_blobs=bool(saved.get("run_iq_blobs", True)),
        do_plotting=False,
        show_handler_output=False,
        report_timing=True,
        task_status_poll_interval=float(saved.get("task_status_poll_interval", 10.0)),
        task_execution_mode="wait",
        low_priority_tasks=bool(saved.get("low_priority_tasks", False)),
        reset=reset_from_metadata(saved),
        states=list(saved.get("states", ["g", "e"])),
    )


def reset_from_metadata(saved_settings: dict[str, Any]) -> ResetSettings:
    saved_reset = saved_settings.get("reset")
    if not isinstance(saved_reset, dict):
        return ResetSettings(reset_type=ResetType.ACTIVE, reset_num=5)

    reset_type = saved_reset.get("reset_type", ResetType.ACTIVE)
    return ResetSettings(
        reset_type=ResetType(reset_type),
        reset_num=int(saved_reset.get("reset_num", 5)),
    )


def print_task_keys(title: str, task_keys: list[str | None]) -> None:
    if not task_keys:
        return
    print(f"{title}:")
    for task_key in task_keys:
        print(f"  - {task_key}")


if __name__ == "__main__":
    main()
