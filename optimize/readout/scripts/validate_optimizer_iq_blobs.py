from __future__ import annotations

import json
import os
import sys
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[3]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(WORKBENCH_ROOT / ".cache" / "matplotlib"))

from workbench_bootstrap import setup_workbench_environment

setup_workbench_environment()

from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util import settings as qratena_settings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ResetType

from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
    ReadoutFidelityWorkflowSettings,
)
from optimize.readout.scripts.load_optimizer_results import resolve_run_dir
from optimize.readout.scripts.run_iq_blobs_active_reset_comparison import (
    extract_fidelity_rows,
    plot_readout_fidelity_comparison,
    rows_by_qubit,
    save_dict_csv,
    save_json,
    save_markdown_report,
)
from resources.load_profile import load_profile, load_task_manager


# Edit these values before running.
ACTION = "submit"  # "submit", "status", or "collect"
RUN_KEY = "17-42-50_sweep_q9_q10_q11_q12_q13_q14_q15_q16_q17_q18_q19"
PROFILE_NAME = "main"
OUTPUT_ROOT = Path("data/readout_optimizer_validation_iq_blobs")

ACTIVE_RESET_NUM = 5
DO_EMULATION = False
SHOW_HANDLER_OUTPUT = False
TASK_STATUS_POLL_INTERVAL = 10.0
LOW_PRIORITY_TASKS = True
WAIT_FOR_RESULTS = False

# Per-qubit best amplitudes validate each qubit at its own optimizer maximum.
# Set to False to validate every qubit at summary["best_mean_amplitude"] instead.
USE_PER_QUBIT_BEST_AMPLITUDE = True


@dataclass(frozen=True)
class OptimizedReadoutParameters:
    qubit: str
    amplitude: float
    pulse_length_s: float


def main() -> None:
    if ACTION == "submit":
        submit_validation(RUN_KEY)
    elif ACTION == "status":
        check_validation_results(RUN_KEY)
    elif ACTION == "collect":
        collect_validation_results(RUN_KEY)
    else:
        raise ValueError(f"Unknown ACTION {ACTION!r}; use 'submit', 'status', or 'collect'.")


def submit_validation(optimizer_run_key: str) -> Path:
    optimizer_run_dir = resolve_run_dir(optimizer_run_key)
    summary = load_json(optimizer_run_dir / "summary.json")
    profile_snapshot = load_json(optimizer_run_dir / "profile.json")
    qubit_names = list(summary["qubits"])
    optimized_parameters = optimizer_readout_parameters(
        summary=summary,
        profile_snapshot=profile_snapshot,
        qubit_names=qubit_names,
        use_per_qubit_best_amplitude=USE_PER_QUBIT_BEST_AMPLITUDE,
    )

    current_profile = load_profile(PROFILE_NAME)
    optimized_profile = load_profile(PROFILE_NAME)
    apply_optimized_readout_parameters(optimized_profile, optimized_parameters)

    task_manager = load_task_manager()
    run_dir = create_validation_run_dir(OUTPUT_ROOT, optimizer_run_dir.name, qubit_names)
    task_entries = []

    for condition, reset in validation_conditions(ACTIVE_RESET_NUM):
        workflow = create_validation_workflow(
            qubit_names=qubit_names,
            profile=current_profile,
            task_manager=task_manager,
            reset=reset,
            task_execution_mode="submit_only",
            profile_name=PROFILE_NAME,
            run_kernels=False,
            run_iq_blobs=True,
        )
        workflow.run()
        task_entries.extend(
            submitted_task_entries(
                workflow=workflow,
                condition=condition,
                reset=reset,
                parameter_set="current",
                stage="current_iq_blobs",
            )
        )

    kernel_workflow = create_validation_workflow(
        qubit_names=qubit_names,
        profile=optimized_profile,
        task_manager=task_manager,
        reset=ResetSettings(),
        task_execution_mode="submit_only",
        profile_name=PROFILE_NAME,
        run_kernels=True,
        run_iq_blobs=False,
    )
    kernel_workflow.run()
    task_entries.extend(
        submitted_task_entries(
            workflow=kernel_workflow,
            condition="optimized_kernels",
            reset=ResetSettings(),
            parameter_set="optimized",
            stage="optimized_kernels",
        )
    )

    manifest = validation_manifest(
        run_dir=run_dir,
        optimizer_run_dir=optimizer_run_dir,
        qubit_names=qubit_names,
        optimized_parameters=optimized_parameters,
        task_entries=task_entries,
    )
    save_validation_submission(
        run_dir=run_dir,
        manifest=manifest,
        current_profile=current_profile,
        optimized_profile=optimized_profile,
    )

    print(f"Submitted validation tasks for optimizer run: {optimizer_run_dir.resolve()}")
    for params in optimized_parameters:
        print(
            f"  - {params.qubit}: amplitude={params.amplitude:.8g}, "
            f"length={params.pulse_length_s * 1e9:.1f} ns"
        )
    print(f"Validation run key: {run_dir.name}")
    print(f"Saved pending validation run to {run_dir.resolve()}")
    return run_dir


def check_validation_results(run_key: str) -> dict[str, Any]:
    run_dir = resolve_validation_run_dir(run_key)
    manifest = load_validation_manifest(run_dir)
    task_manager = load_task_manager()
    summary = update_validation_task_statuses(
        run_dir=run_dir,
        manifest=manifest,
        task_manager=task_manager,
    )
    print(summary["message"])
    print(f"Task counts: {summary['counts']}")
    print_task_keys("Pending task keys", summary["pending_task_keys"])
    print_task_keys("Failed/cancelled task keys", summary["failed_task_keys"])
    return summary


def collect_validation_results(run_key: str) -> dict[str, Any]:
    run_dir = resolve_validation_run_dir(run_key)
    manifest = load_validation_manifest(run_dir)
    task_manager = load_task_manager()
    status = update_validation_task_statuses(
        run_dir=run_dir,
        manifest=manifest,
        task_manager=task_manager,
    )
    print(status["message"])
    if status["failed"]:
        return status
    if not WAIT_FOR_RESULTS and not status["ready_to_collect"]:
        return status

    profile_name = str(manifest.get("profile_name", PROFILE_NAME))
    current_profile = load_profile(profile_name)
    optimized_profile = load_profile(profile_name)
    optimized_parameters = [
        OptimizedReadoutParameters(
            qubit=str(item["qubit"]),
            amplitude=float(item["amplitude"]),
            pulse_length_s=float(item["pulse_length_s"]),
        )
        for item in manifest["optimized_readout_parameters"]
    ]
    apply_optimized_readout_parameters(optimized_profile, optimized_parameters)

    if has_submitted_optimized_iq_tasks(manifest):
        return collect_submitted_validation_iq_results(
            run_dir=run_dir,
            manifest=manifest,
            task_manager=task_manager,
            current_profile=current_profile,
            optimized_profile=optimized_profile,
            profile_name=profile_name,
        )

    current_condition_results = collect_current_iq_results(
        manifest=manifest,
        task_manager=task_manager,
        profile=current_profile,
        profile_name=profile_name,
    )
    kernel_dir = run_dir / "optimized_kernel_files"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_tasks = [
        task
        for task in manifest["tasks"]
        if task.get("stage") == "optimized_kernels" or task.get("node") == "kernels"
    ]
    with temporary_kernel_trace_dir(kernel_dir):
        kernel_workflow = create_validation_workflow(
            qubit_names=list(manifest["qubits"]),
            profile=optimized_profile,
            task_manager=task_manager,
            reset=ResetSettings(),
            task_execution_mode="wait",
            profile_name=profile_name,
            run_kernels=True,
            run_iq_blobs=False,
        )
        kernel_workflow.collect_submitted_results(kernel_tasks)

        iq_task_entries = []
        for condition, reset in validation_conditions(int(manifest["active_reset_num"])):
            workflow = create_validation_workflow(
                qubit_names=list(manifest["qubits"]),
                profile=optimized_profile,
                task_manager=task_manager,
                reset=reset,
                task_execution_mode="submit_only",
                profile_name=profile_name,
                run_kernels=False,
                run_iq_blobs=True,
            )
            workflow.run()
            iq_task_entries.extend(
                submitted_task_entries(
                    workflow=workflow,
                    condition=condition,
                    reset=reset,
                    parameter_set="optimized",
                    stage="optimized_iq_blobs",
                    extra_fields={"kernel_dir": str(kernel_dir)},
                )
            )

    for task in kernel_tasks:
        task["result_status"] = "collected"
        task["kernel_dir"] = str(kernel_dir)
    manifest["tasks"].extend(iq_task_entries)
    manifest["task_count"] = len(manifest["tasks"])
    manifest["run_status"] = "optimized_iq_submitted"
    manifest["optimized_kernels_collected_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    manifest["current_condition_results"] = current_condition_results
    save_manifest(run_dir, manifest)

    if not WAIT_FOR_RESULTS:
        summary = update_validation_task_statuses(
            run_dir=run_dir,
            manifest=manifest,
            task_manager=task_manager,
        )
        print(summary["message"])
        return summary

    return collect_submitted_validation_iq_results(
        run_dir=run_dir,
        manifest=manifest,
        task_manager=task_manager,
        current_profile=current_profile,
        optimized_profile=optimized_profile,
        profile_name=profile_name,
    )


def collect_current_iq_results(
    *,
    manifest: dict[str, Any],
    task_manager: Any,
    profile: Any,
    profile_name: str,
) -> dict[str, list[dict[str, Any]]]:
    condition_results: dict[str, list[dict[str, Any]]] = {}
    for condition, reset in validation_conditions(int(manifest["active_reset_num"])):
        workflow = create_validation_workflow(
            qubit_names=list(manifest["qubits"]),
            profile=profile,
            task_manager=task_manager,
            reset=reset,
            task_execution_mode="wait",
            profile_name=profile_name,
            run_kernels=False,
            run_iq_blobs=True,
        )
        condition_tasks = [
            task
            for task in manifest["tasks"]
            if task.get("condition") == condition
            and task.get("parameter_set") == "current"
            and task.get("node") == "iq_blobs"
        ]
        result = workflow.collect_submitted_results(condition_tasks)
        condition_results[condition] = extract_fidelity_rows(
            list(manifest["qubits"]),
            result["iq_blobs"],
            condition,
        )
        for task in condition_tasks:
            task["result_status"] = "collected"
    return condition_results


def collect_submitted_validation_iq_results(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    task_manager: Any,
    current_profile: Any,
    optimized_profile: Any,
    profile_name: str,
) -> dict[str, Any]:
    current_condition_results = manifest.get("current_condition_results")
    if not current_condition_results:
        current_condition_results = collect_current_iq_results(
            manifest=manifest,
            task_manager=task_manager,
            profile=current_profile,
            profile_name=profile_name,
        )

    optimized_condition_results: dict[str, list[dict[str, Any]]] = {}
    kernel_dir = Path(manifest.get("optimized_kernel_dir") or run_dir / "optimized_kernel_files")
    for condition, reset in validation_conditions(int(manifest["active_reset_num"])):
        workflow = create_validation_workflow(
            qubit_names=list(manifest["qubits"]),
            profile=optimized_profile,
            task_manager=task_manager,
            reset=reset,
            task_execution_mode="wait",
            profile_name=profile_name,
            run_kernels=False,
            run_iq_blobs=True,
        )
        condition_tasks = [
            task
            for task in manifest["tasks"]
            if task.get("condition") == condition
            and task.get("parameter_set") == "optimized"
            and task.get("node") == "iq_blobs"
        ]
        with temporary_kernel_trace_dir(kernel_dir):
            result = workflow.collect_submitted_results(condition_tasks)
        optimized_condition_results[condition] = extract_fidelity_rows(
            list(manifest["qubits"]),
            result["iq_blobs"],
            condition,
        )
        for task in condition_tasks:
            task["result_status"] = "collected"

    current_report_rows = build_validation_report_rows(
        qubit_names=list(manifest["qubits"]),
        profile=current_profile,
        no_reset_rows=current_condition_results["without_active_reset"],
        active_reset_rows=current_condition_results["with_active_reset"],
    )
    optimized_report_rows = build_validation_report_rows(
        qubit_names=list(manifest["qubits"]),
        profile=optimized_profile,
        no_reset_rows=optimized_condition_results["without_active_reset"],
        active_reset_rows=optimized_condition_results["with_active_reset"],
    )
    save_validation_results(
        run_dir=run_dir,
        manifest=manifest,
        condition_results={
            "current": current_condition_results,
            "optimized": optimized_condition_results,
        },
        report_rows={
            "current": current_report_rows,
            "optimized": optimized_report_rows,
        },
    )
    print(f"Collected and saved validation results to {run_dir.resolve()}")
    return {
        "run_dir": run_dir,
        "condition_results": {
            "current": current_condition_results,
            "optimized": optimized_condition_results,
        },
        "report_rows": {
            "current": current_report_rows,
            "optimized": optimized_report_rows,
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_validation_run_dir(
    output_root: Path,
    optimizer_run_key: str,
    qubit_names: list[str],
) -> Path:
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    qubit_slug = "_".join(qubit_names)
    run_name = f"{timestamp}_validate_{optimizer_run_key}_{qubit_slug}"
    run_dir = output_root / date_folder / run_name

    suffix = 1
    while run_dir.exists():
        run_dir = output_root / date_folder / f"{run_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def resolve_validation_run_dir(run_key: str) -> Path:
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
            f"Could not find validation run folder {run_key!r} under {OUTPUT_ROOT}."
        )

    joined = "\n".join(str(path) for path in matches)
    raise RuntimeError(
        f"Validation run key {run_key!r} matched multiple folders:\n{joined}\n"
        "Use the full run folder path."
    )


def load_validation_manifest(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "task_manifest.json")


def validation_conditions(active_reset_num: int) -> list[tuple[str, ResetSettings]]:
    return [
        ("without_active_reset", ResetSettings()),
        (
            "with_active_reset",
            ResetSettings(reset_type=ResetType.ACTIVE, reset_num=active_reset_num),
        ),
    ]


def create_validation_workflow(
    *,
    qubit_names: list[str],
    profile: Any,
    task_manager: Any,
    reset: ResetSettings,
    task_execution_mode: str,
    profile_name: str,
    run_kernels: bool,
    run_iq_blobs: bool,
) -> ReadoutFidelityWorkflow:
    settings = ReadoutFidelityWorkflowSettings(
        profile_name=profile_name,
        do_emulation=DO_EMULATION,
        run_resonator=False,
        run_kernels=run_kernels,
        run_iq_blobs=run_iq_blobs,
        do_plotting=False,
        show_handler_output=SHOW_HANDLER_OUTPUT,
        report_timing=True,
        task_status_poll_interval=TASK_STATUS_POLL_INTERVAL,
        task_execution_mode=task_execution_mode,
        low_priority_tasks=LOW_PRIORITY_TASKS,
        reset=reset,
    )
    return ReadoutFidelityWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
    )


def submitted_task_entries(
    *,
    workflow: ReadoutFidelityWorkflow,
    condition: str,
    reset: ResetSettings,
    parameter_set: str,
    stage: str,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries = []
    for task_index, task in enumerate(workflow.submitted_tasks):
        task_entry = dict(task)
        task_entry.update(
            {
                "condition": condition,
                "condition_index": 0 if condition == "without_active_reset" else 1,
                "parameter_set": parameter_set,
                "stage": stage,
                "reset": reset.model_dump(mode="json"),
                "task_key": validation_task_key(condition, task, task_index),
            }
        )
        if extra_fields:
            task_entry.update(extra_fields)
        entries.append(task_entry)
    return entries


def validation_task_key(
    condition: str,
    task: dict[str, Any],
    task_index: int,
) -> str:
    qubits = "+".join(task.get("qubit_names", [])) or "no_qubits"
    node = task.get("node", "task")
    return f"validation/{condition}/{node}/{qubits}/{task_index:02d}"


def validation_manifest(
    *,
    run_dir: Path,
    optimizer_run_dir: Path,
    qubit_names: list[str],
    optimized_parameters: list[OptimizedReadoutParameters],
    task_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    created_at = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "run_status": "submitted_pending_results",
        "run_key": run_dir.name,
        "run_dir": str(run_dir),
        "created_at": created_at,
        "source_optimizer_run_key": optimizer_run_dir.name,
        "source_optimizer_run_dir": str(optimizer_run_dir),
        "profile_name": PROFILE_NAME,
        "qubits": list(qubit_names),
        "active_reset_num": ACTIVE_RESET_NUM,
        "use_per_qubit_best_amplitude": USE_PER_QUBIT_BEST_AMPLITUDE,
        "optimized_readout_parameters": [
            asdict(params) for params in optimized_parameters
        ],
        "conditions": [
            "without_active_reset",
            "with_active_reset",
        ],
        "stages": [
            "current_iq_blobs",
            "optimized_kernels",
            "optimized_iq_blobs",
        ],
        "task_count": len(task_entries),
        "tasks": task_entries,
    }


def save_validation_submission(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    current_profile: Any,
    optimized_profile: Any,
) -> None:
    save_manifest(run_dir, manifest)
    (run_dir / "metadata.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    current_snapshot = (
        current_profile.model_dump(mode="json")
        if hasattr(current_profile, "model_dump")
        else {}
    )
    optimized_snapshot = (
        optimized_profile.model_dump(mode="json")
        if hasattr(optimized_profile, "model_dump")
        else {}
    )
    (run_dir / "current_profile.json").write_text(
        json.dumps(current_snapshot, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "optimized_profile.json").write_text(
        json.dumps(optimized_snapshot, indent=2, default=str),
        encoding="utf-8",
    )
    report = [
        "# Readout Optimizer Validation Submission",
        "",
        f"Created at: {manifest['created_at']}",
        f"Run key: `{manifest['run_key']}`",
        f"Source optimizer run: `{manifest['source_optimizer_run_key']}`",
        f"Qubits: {', '.join(manifest['qubits'])}",
        f"Submitted tasks: {manifest['task_count']}",
        "",
        "This staged validation first submits current-profile IQ blobs and",
        "optimized-profile kernel tasks. After kernels finish, collect submits",
        "optimized-profile IQ blobs using the generated kernel files.",
        "",
        "Results have not been collected yet.",
        "Set `ACTION = \"collect\"` and `RUN_KEY` to this validation run key.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def optimizer_readout_parameters(
    *,
    summary: dict[str, Any],
    profile_snapshot: dict[str, Any],
    qubit_names: list[str],
    use_per_qubit_best_amplitude: bool,
) -> list[OptimizedReadoutParameters]:
    best_mean_amplitude = float(summary["best_mean_amplitude"])
    qubit_summaries = summary.get("qubit_summaries", {})

    parameters = []
    for qubit_name in qubit_names:
        qubit_summary = qubit_summaries.get(qubit_name, {})
        amplitude = (
            float(qubit_summary["best_amplitude"])
            if use_per_qubit_best_amplitude
            else best_mean_amplitude
        )
        pulse_length = readout_pulse_length_from_snapshot(profile_snapshot, qubit_name)
        parameters.append(
            OptimizedReadoutParameters(
                qubit=qubit_name,
                amplitude=amplitude,
                pulse_length_s=pulse_length,
            )
        )

    return parameters


def readout_pulse_length_from_snapshot(
    profile_snapshot: dict[str, Any],
    qubit_name: str,
) -> float:
    try:
        readout_const = profile_snapshot["qubits"][qubit_name]["properties"]["pulses"][
            "readout"
        ]["const"]["properties"]
        return float(readout_const["readout_duration"])
    except KeyError as error:
        raise KeyError(
            f"Could not find const readout_duration for {qubit_name} "
            "in optimizer profile.json."
        ) from error


def apply_optimized_readout_parameters(
    profile: Any,
    parameters: list[OptimizedReadoutParameters],
) -> None:
    for params in parameters:
        readout_pulse = profile.qubits[params.qubit].pulses[SUPPORTED_PULSE_TYPES.readout][
            SUPPORTED_PULSE_SHAPES.const
        ]
        readout_pulse.readout_amplitude = params.amplitude
        readout_pulse.readout_duration = params.pulse_length_s


def update_validation_task_statuses(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    task_manager: Any,
) -> dict[str, Any]:
    counts = {
        "completed": 0,
        "queued": 0,
        "running": 0,
        "failed": 0,
        "cancelled": 0,
        "unknown": 0,
    }
    pending_tasks = []
    failed_tasks = []
    completed_tasks = []

    checked_at = datetime.now().isoformat(timespec="seconds")
    for task in manifest.get("tasks", []):
        task_id = str(task["task_id"])
        status = submitted_task_status(task_manager, task_id)
        task["task_status"] = status
        task["checked_at"] = checked_at

        if status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1

        if status == "completed":
            task["result_status"] = "ready"
            completed_tasks.append(task)
        elif status in ("failed", "cancelled"):
            task["result_status"] = status
            failed_tasks.append(task)
        else:
            task["result_status"] = "pending"
            pending_tasks.append(task)

    total = len(manifest.get("tasks", []))
    ready_to_collect = total > 0 and len(completed_tasks) == total
    manifest["last_checked_at"] = checked_at
    manifest["run_status"] = (
        "ready_to_collect" if ready_to_collect else "submitted_pending_results"
    )
    save_manifest(run_dir, manifest)

    return {
        "run_dir": str(run_dir),
        "ready_to_collect": ready_to_collect,
        "total": total,
        "counts": counts,
        "completed": len(completed_tasks),
        "pending": len(pending_tasks),
        "failed": len(failed_tasks),
        "pending_task_keys": [task.get("task_key") for task in pending_tasks],
        "failed_task_keys": [task.get("task_key") for task in failed_tasks],
        "message": validation_status_message(
            ready_to_collect=ready_to_collect,
            pending_count=len(pending_tasks),
            failed_count=len(failed_tasks),
        ),
    }


def submitted_task_status(task_manager: Any, task_id: str) -> str:
    get_status = getattr(task_manager, "get_status", None)
    if callable(get_status):
        try:
            return normalize_task_status(get_status(task_id))
        except Exception as error:
            return f"unknown:{type(error).__name__}"

    is_done = getattr(task_manager, "is_done", None)
    if callable(is_done):
        try:
            return "completed" if is_done(task_id) else "running"
        except Exception as error:
            return f"unknown:{type(error).__name__}"

    return "unknown"


def normalize_task_status(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def validation_status_message(
    *,
    ready_to_collect: bool,
    pending_count: int,
    failed_count: int,
) -> str:
    if ready_to_collect:
        return "All submitted validation tasks are complete; results are ready to collect."
    if failed_count:
        return (
            f"{failed_count} submitted validation task(s) failed or were cancelled; "
            "inspect task_manifest.json before collecting."
        )
    return (
        f"{pending_count} submitted validation task(s) are still queued/running; "
        "wait before collecting results."
    )


def print_task_keys(title: str, task_keys: list[str | None]) -> None:
    if not task_keys:
        return
    print(f"{title}:")
    for task_key in task_keys:
        print(f"  - {task_key}")


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "task_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def build_validation_report_rows(
    *,
    qubit_names: list[str],
    profile: Any,
    no_reset_rows: list[dict[str, Any]],
    active_reset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    no_reset_by_qubit = rows_by_qubit(no_reset_rows)
    active_reset_by_qubit = rows_by_qubit(active_reset_rows)
    rows = []

    for qubit_name in qubit_names:
        qubit = profile.qubits[qubit_name]
        readout_pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][
            SUPPORTED_PULSE_SHAPES.const
        ]
        no_reset_row = no_reset_by_qubit.get(qubit_name, {})
        active_reset_row = active_reset_by_qubit.get(qubit_name, {})
        no_reset_fidelity = optional_float(no_reset_row.get("readout_fidelity"))
        active_reset_fidelity = optional_float(active_reset_row.get("readout_fidelity"))

        rows.append(
            {
                "qubit": qubit_name,
                "readout_resonator_frequency_hz": optional_float(
                    getattr(qubit.readout_resonator_frequency, "value", None)
                ),
                "readout_pulse_length_s": optional_float(
                    readout_pulse.readout_duration
                ),
                "readout_amplitude": optional_float(readout_pulse.readout_amplitude),
                "fidelity_without_active_reset": no_reset_fidelity,
                "fidelity_without_active_reset_error": optional_float(
                    no_reset_row.get("readout_fidelity_error")
                ),
                "fidelity_with_active_reset": active_reset_fidelity,
                "fidelity_with_active_reset_error": optional_float(
                    active_reset_row.get("readout_fidelity_error")
                ),
                "fidelity_delta_active_minus_no_reset": optional_difference(
                    active_reset_fidelity,
                    no_reset_fidelity,
                ),
                "separation_without_active_reset": optional_float(
                    no_reset_row.get("separation")
                ),
                "separation_with_active_reset": optional_float(
                    active_reset_row.get("separation")
                ),
            }
        )

    return rows


def save_validation_results(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    condition_results: dict[str, dict[str, list[dict[str, Any]]]],
    report_rows: dict[str, list[dict[str, Any]]],
) -> None:
    all_rows = []
    for parameter_set, results_by_condition in condition_results.items():
        for rows in results_by_condition.values():
            for row in rows:
                row = dict(row)
                row["parameter_set"] = parameter_set
                all_rows.append(row)
    save_dict_csv(
        run_dir / "readout_fidelities_active_reset_comparison.csv",
        all_rows,
        [
            "parameter_set",
            "qubit",
            "condition",
            "readout_fidelity",
            "readout_fidelity_error",
            "separation",
            "status",
        ],
    )
    combined_report_rows = []
    for parameter_set, rows in report_rows.items():
        for row in rows:
            row = dict(row)
            row["parameter_set"] = parameter_set
            combined_report_rows.append(row)
    save_dict_csv(
        run_dir / "readout_report.csv",
        combined_report_rows,
        [
            "parameter_set",
            "qubit",
            "readout_resonator_frequency_hz",
            "readout_pulse_length_s",
            "readout_amplitude",
            "fidelity_without_active_reset",
            "fidelity_without_active_reset_error",
            "fidelity_with_active_reset",
            "fidelity_with_active_reset_error",
            "fidelity_delta_active_minus_no_reset",
            "separation_without_active_reset",
            "separation_with_active_reset",
        ],
    )
    save_json(run_dir / "readout_report.json", combined_report_rows)
    for parameter_set, rows in report_rows.items():
        save_markdown_report(
            run_dir / f"{parameter_set}_readout_report.md",
            rows,
            int(manifest["active_reset_num"]),
        )
        figure = plot_readout_fidelity_comparison(
            qubit_names=list(manifest["qubits"]),
            no_reset_rows=condition_results[parameter_set]["without_active_reset"],
            active_reset_rows=condition_results[parameter_set]["with_active_reset"],
            active_reset_num=int(manifest["active_reset_num"]),
            run_datetime=datetime.now(),
        )
        figure.savefig(
            run_dir / f"{parameter_set}_readout_fidelities_active_reset_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )

    manifest["run_status"] = "complete"
    manifest["collected_at"] = datetime.now().isoformat(timespec="seconds")
    for task in manifest.get("tasks", []):
        task["result_status"] = "collected"
    save_manifest(run_dir, manifest)


def has_submitted_optimized_iq_tasks(manifest: dict[str, Any]) -> bool:
    return any(
        task.get("stage") == "optimized_iq_blobs"
        or (
            task.get("parameter_set") == "optimized"
            and task.get("node") == "iq_blobs"
        )
        for task in manifest.get("tasks", [])
    )


@contextlib.contextmanager
def temporary_kernel_trace_dir(kernel_dir: Path):
    previous_dir = qratena_settings.KERNEL_TRACES_DIR_PATH
    qratena_settings.KERNEL_TRACES_DIR_PATH = Path(kernel_dir)
    try:
        yield
    finally:
        qratena_settings.KERNEL_TRACES_DIR_PATH = previous_dir


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


if __name__ == "__main__":
    main()
