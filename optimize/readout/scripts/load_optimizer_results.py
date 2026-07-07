from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[3]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType

from optimize.readout.readout_amplitude_optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from optimize.readout.utils.readout_sweep_artifacts import load_readout_task_manifest
from resources.load_profile import load_profile, load_task_manager


# Edit these values before running.
RUN_KEY = "17-42-50_sweep_q9_q10_q11_q12_q13_q14_q15_q16_q17_q18_q19"  # folder name or full run folder path
OUTPUT_ROOT = Path("data") / "readout_optimize"
PROFILE_BRANCH = None  # None means use metadata profile_name
WAIT_FOR_RESULTS = False


def main() -> None:
    run_dir = resolve_run_dir(RUN_KEY)
    metadata = load_metadata(run_dir)
    manifest = load_readout_task_manifest(run_dir)
    qubit_names = list(manifest["qubits"])
    workflow_settings = workflow_settings_from_metadata(metadata)

    profile = load_profile(PROFILE_BRANCH or workflow_settings.profile_name)
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
    if not WAIT_FOR_RESULTS and not summary["ready_to_collect"]:
        return

    if is_staged_kernel_run(manifest):
        result = optimizer.collect_kernels_submit_iq_blobs(
            run_dir,
            wait_for_iq_results=WAIT_FOR_RESULTS,
            save_results=True,
        )
    elif is_staged_iq_run(manifest):
        result = optimizer.collect_staged_iq_results(
            run_dir,
            save_results=True,
            wait=WAIT_FOR_RESULTS,
        )
    else:
        result = optimizer.collect_submitted_results(
            run_dir,
            save_results=True,
            wait=True,
        )

    if isinstance(result, dict) and "ready_to_collect" in result:
        print(result["message"])
    else:
        print(f"Collected and saved optimizer results to {run_dir.resolve()}")


def resolve_run_dir(run_key: str) -> Path:
    candidate = Path(run_key).expanduser()
    if candidate.exists():
        return candidate.resolve()

    matches = [
        manifest_path.parent
        for manifest_path in OUTPUT_ROOT.expanduser().rglob("task_manifest.json")
        if manifest_path.parent.name == run_key
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Could not find run folder {run_key!r} under {OUTPUT_ROOT}."
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


def is_staged_kernel_run(manifest: dict[str, Any]) -> bool:
    tasks = list(manifest.get("tasks", []))
    return bool(tasks) and any(task.get("node") == "kernels" for task in tasks) and not any(
        task.get("node") == "iq_blobs" for task in tasks
    )


def is_staged_iq_run(manifest: dict[str, Any]) -> bool:
    return any(
        task.get("stage") == "iq_blobs" or task.get("depends_on_stage") == "kernels"
        for task in manifest.get("tasks", [])
    )


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
