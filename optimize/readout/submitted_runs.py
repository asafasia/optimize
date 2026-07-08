from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from optimize.readout.utils.readout_sweep_artifacts import (
    load_readout_task_manifest,
    save_pending_readout_submission,
    update_readout_task_manifest,
)


class SubmittedReadoutRunMixin:
    def _workflow_settings_for_measurement(self) -> ReadoutFidelityWorkflowSettings:
        if not self.settings.submit_only:
            return self.settings.workflow_settings

        return replace(
            self.settings.workflow_settings,
            task_execution_mode="submit_only",
        )

    def _record_submitted_tasks(
        self,
        amplitude: float,
        workflow: Any,
    ) -> None:
        amplitude_index = len(self.submitted_amplitudes)
        self.submitted_amplitudes.append(float(amplitude))
        self.submitted_task_entries.extend(
            self._submitted_task_entries_for_workflow(
                amplitude=amplitude,
                amplitude_index=amplitude_index,
                workflow=workflow,
            )
        )

    def _submitted_task_entries_for_workflow(
        self,
        *,
        amplitude: float,
        amplitude_index: int,
        workflow: Any,
        extra_fields: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        entries = []
        for task_index, task in enumerate(workflow.submitted_tasks):
            task_entry = dict(task)
            task_entry.update(
                {
                    "amplitude": float(amplitude),
                    "sweep_index": amplitude_index,
                    "sweep_parameters": {"readout_amplitude": float(amplitude)},
                    "task_key": self._submitted_task_key(
                        amplitude_index=amplitude_index,
                        amplitude=amplitude,
                        task=task,
                        task_index=task_index,
                    ),
                }
            )
            if extra_fields:
                task_entry.update(extra_fields)
            entries.append(task_entry)
        return entries

    def _submitted_task_key(
        self,
        *,
        amplitude_index: int,
        amplitude: float,
        task: dict[str, Any],
        task_index: int,
    ) -> str:
        qubits = "+".join(task.get("qubit_names", [])) or "no_qubits"
        node = task.get("node", "task")
        return (
            f"sweep/{amplitude_index:04d}/"
            f"readout_amplitude={amplitude:.6g}/{node}/{qubits}/{task_index:02d}"
        )

    def _save_pending_submission(self) -> None:
        if self.run_dir is None:
            return

        save_pending_readout_submission(
            run_dir=self.run_dir,
            qubit_names=self.qubit_names,
            amplitudes=[float(amplitude) for amplitude in self.settings.amplitudes],
            scan_method=str(ReadoutScanMethod(self.settings.method).value),
            task_entries=self.submitted_task_entries,
            profile=self.profile,
            profile_path=self.settings.profile_path,
            optimizer_settings=self.settings,
            workflow_settings=self.settings.workflow_settings,
        )
        print(f"Saved pending readout optimizer submission to {self.run_dir.resolve()}")

    def collect_submitted_results(
        self,
        run_dir: str | Path,
        *,
        save_results: bool = True,
        wait: bool = True,
    ) -> dict[float, dict[str, Any]] | dict[str, Any]:
        if not wait:
            return self.check_submitted_results(run_dir)

        manifest = load_readout_task_manifest(run_dir)
        task_entries = list(manifest.get("tasks", []))
        tasks_by_amplitude = self._tasks_by_amplitude(task_entries)

        self._prepare_profile_for_states()
        self.workflows = {}
        self.results = {}
        self.measured_amplitudes = []
        self.fidelities = {qubit_name: [] for qubit_name in self.qubit_names}
        self.fidelity_errors = {qubit_name: [] for qubit_name in self.qubit_names}
        self.separations = {qubit_name: [] for qubit_name in self.qubit_names}
        self.roundnesses = {qubit_name: [] for qubit_name in self.qubit_names}
        self.resonator_frequencies = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.iq_blob_figures = {}
        self.kernel_figures = {}
        self.resonator_figures = {}
        self.measurement_errors = {}
        self.run_dir = Path(run_dir)

        for amplitude in sorted(tasks_by_amplitude):
            self._set_readout_amplitude(amplitude)
            workflow = self._create_readout_workflow(
                qubit_names=self.qubit_names,
                profile=self.profile,
                task_manager=self.task_manager,
                settings=replace(
                    self.settings.workflow_settings,
                    task_execution_mode="wait",
                ),
            )
            result = workflow.collect_submitted_results(tasks_by_amplitude[amplitude])
            self.workflows[amplitude] = workflow
            self.results[amplitude] = result
            self.measured_amplitudes.append(amplitude)
            self._record_fidelities(result)
            self._record_resonator_frequencies(workflow)
            self.readout_frequencies = self._readout_frequencies()
            self._record_iq_blob_figures(amplitude, workflow)
            self._record_kernel_figures(amplitude, workflow)
            self._record_resonator_figures(amplitude, workflow)

        manifest["run_status"] = "complete"
        manifest["collected_at"] = datetime.now().isoformat(timespec="seconds")
        for task in manifest.get("tasks", []):
            task["result_status"] = "collected"
        update_readout_task_manifest(run_dir, manifest)

        if save_results:
            self.save_results(output_dir=Path(run_dir).parent, run_dir=run_dir)

        return self.results

    def check_submitted_results(self, run_dir: str | Path) -> dict[str, Any]:
        """Check saved task IDs without waiting for results."""
        manifest = load_readout_task_manifest(run_dir)
        tasks = list(manifest.get("tasks", []))
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
        for task in tasks:
            task_id = str(task["task_id"])
            status = self._submitted_task_status(task_id)
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

        total = len(tasks)
        ready_to_collect = total > 0 and len(completed_tasks) == total
        manifest["last_checked_at"] = checked_at
        manifest["run_status"] = (
            "ready_to_collect"
            if ready_to_collect
            else "submitted_pending_results"
        )
        update_readout_task_manifest(run_dir, manifest)

        summary = {
            "run_dir": str(Path(run_dir)),
            "ready_to_collect": ready_to_collect,
            "total": total,
            "counts": counts,
            "completed": len(completed_tasks),
            "pending": len(pending_tasks),
            "failed": len(failed_tasks),
            "pending_task_keys": [task.get("task_key") for task in pending_tasks],
            "failed_task_keys": [task.get("task_key") for task in failed_tasks],
            "message": self._submitted_results_status_message(
                ready_to_collect=ready_to_collect,
                pending_count=len(pending_tasks),
                failed_count=len(failed_tasks),
            ),
        }
        print(summary["message"])
        return summary

    def _submitted_task_status(self, task_id: str) -> str:
        get_status = getattr(self.task_manager, "get_status", None)
        if callable(get_status):
            try:
                status = get_status(task_id)
            except Exception as error:
                return f"unknown:{type(error).__name__}"
            return self._normalize_task_status(status)

        is_done = getattr(self.task_manager, "is_done", None)
        if callable(is_done):
            try:
                return "completed" if is_done(task_id) else "running"
            except Exception as error:
                return f"unknown:{type(error).__name__}"

        return "unknown"

    def _normalize_task_status(self, status: Any) -> str:
        value = getattr(status, "value", status)
        return str(value).lower()

    def _submitted_results_status_message(
        self,
        *,
        ready_to_collect: bool,
        pending_count: int,
        failed_count: int,
    ) -> str:
        if ready_to_collect:
            return "All submitted readout tasks are complete; results are ready to collect."
        if failed_count:
            return (
                f"{failed_count} submitted readout task(s) failed or were cancelled; "
                "inspect task_manifest.json before collecting."
            )
        return (
            f"{pending_count} submitted readout task(s) are still queued/running; "
            "wait before collecting results."
        )

    def _tasks_by_amplitude(
        self,
        task_entries: list[dict[str, Any]],
    ) -> dict[float, list[dict[str, Any]]]:
        tasks_by_amplitude: dict[float, list[dict[str, Any]]] = {}
        for entry in task_entries:
            amplitude = float(entry["amplitude"])
            tasks_by_amplitude.setdefault(amplitude, []).append(entry)
        return tasks_by_amplitude
