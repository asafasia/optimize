from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from workbench_bootstrap import setup_workbench_environment

setup_workbench_environment()

from matplotlib.figure import Figure
from qratena.system.components_params.profile import Profile

from optimize.readout.optimizer_figures import ReadoutFigureMixin
from optimize.readout.optimizer_metrics import ReadoutMetricsMixin
from optimize.readout.optimizer_settings import ReadoutAmplitudeSweepSettings
from optimize.readout.profile_access import ReadoutProfileAccessMixin
from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
)
from optimize.readout.submitted_runs import SubmittedReadoutRunMixin
from optimize.readout.utils.readout_live_html_plotter import ReadoutLiveHtmlPlotter
from optimize.readout.utils.readout_scan_methods import scan_method_for
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from optimize.readout.utils.readout_sweep_analysis import ReadoutAmplitudeSweepAnalysis
from optimize.readout.utils.readout_sweep_artifacts import (
    ReadoutAmplitudeSweepSaver,
    create_readout_run_dir,
)
from optimize.readout.utils.readout_sweep_plotter import ReadoutAmplitudeSweepPlotter

if TYPE_CHECKING:
    from qigeon.io.task_submitter import TaskSubmitterAsync
else:
    TaskSubmitterAsync = Any


class ReadoutAmplitudeSweepWorkflow(
    SubmittedReadoutRunMixin,
    ReadoutProfileAccessMixin,
    ReadoutMetricsMixin,
    ReadoutFigureMixin,
):
    """Run ReadoutFidelityWorkflow for several readout amplitudes."""

    def __init__(
        self,
        qubit_names: list[str],
        profile: Profile,
        task_manager: TaskSubmitterAsync,
        settings: ReadoutAmplitudeSweepSettings,
    ) -> None:
        self.qubit_names = qubit_names
        self.profile = profile
        self.task_manager = task_manager
        self.settings = settings

        self.workflows: dict[float, ReadoutFidelityWorkflow] = {}
        self.results: dict[float, dict[str, Any]] = {}
        self.measured_amplitudes: list[float] = []
        self.fidelities: dict[str, list[float]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.fidelity_errors: dict[str, list[float | None]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.separations: dict[str, list[float | None]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.roundnesses: dict[str, list[float | None]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.resonator_frequencies: dict[str, list[float | None]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.iq_blob_figures: dict[float, list[Figure]] = {}
        self.kernel_figures: dict[float, list[Figure]] = {}
        self.resonator_figures: dict[float, list[Figure]] = {}
        self.measurement_errors: dict[float, str] = {}
        self.submitted_amplitudes: list[float] = []
        self.submitted_task_entries: list[dict[str, Any]] = []
        self.initial_amplitudes = self._readout_amplitudes()
        self.readout_lengths = self._readout_lengths()
        self.readout_frequencies = self._readout_frequencies()
        self.reset_label = self._reset_label()
        self.interrupted = False
        self.interrupt_reason: str | None = None
        self.live_plotter: ReadoutLiveHtmlPlotter | None = None
        self.run_dir: Path | None = None
        self.figure: Figure | None = None

    def run(self) -> dict[float, dict[str, Any]]:
        self._validate_execution_mode()
        self._prepare_profile_for_states()
        self.workflows = {}
        self.results = {}
        self.measured_amplitudes = []
        self.submitted_amplitudes = []
        self.submitted_task_entries = []
        self.fidelities = {qubit_name: [] for qubit_name in self.qubit_names}
        self.fidelity_errors = {qubit_name: [] for qubit_name in self.qubit_names}
        self.separations = {qubit_name: [] for qubit_name in self.qubit_names}
        self.roundnesses = {qubit_name: [] for qubit_name in self.qubit_names}
        self.resonator_frequencies = {qubit_name: [] for qubit_name in self.qubit_names}
        self.iq_blob_figures = {}
        self.kernel_figures = {}
        self.resonator_figures = {}
        self.measurement_errors = {}
        self.interrupted = False
        self.interrupt_reason = None
        self.run_dir = None
        self.figure = None

        if self.settings.submit_only:
            self.run_dir = create_readout_run_dir(
                output_dir=self.settings.live_html_output_dir,
                scan_method=str(ReadoutScanMethod(self.settings.method).value),
                qubit_names=self.qubit_names,
            )
        else:
            self._start_live_plotter()
        try:
            scan_method_for(self).run()
        except (KeyboardInterrupt, EOFError) as error:
            self.interrupted = True
            self.interrupt_reason = type(error).__name__
            if not self.settings.fill_unfinished_on_interrupt:
                raise
            self._fill_unfinished_fidelities()
            self._update_live_plotter()
            print("\nReadout optimization interrupted; padded unfinished fidelities.")
        finally:
            if self.settings.submit_only:
                self._save_pending_submission()
            else:
                self._finish_live_plotter()

        completed_items = (
            self.submitted_amplitudes
            if self.settings.submit_only
            else self.measured_amplitudes
        )
        completed = len(completed_items)
        self._finish_progress(completed)
        self._auto_save_after_run()
        return self.results

    def _validate_execution_mode(self) -> None:
        if not self.settings.submit_only:
            return

        method = ReadoutScanMethod(self.settings.method)
        if method != ReadoutScanMethod.SWEEP:
            raise ValueError("submit_only is supported only for sweep mode.")
        if self.settings.workflow_settings.do_emulation:
            raise ValueError("submit_only requires a task manager; disable do_emulation.")

    def _prepare_profile_for_states(self) -> None:
        if "f" not in self.settings.workflow_settings.states:
            return

        for qubit_name in self.qubit_names:
            self.profile.ensure_pi_ef_pulse(qubit_name, overwrite=False)

    def plot(self) -> Figure:
        if not self.measured_amplitudes:
            raise RuntimeError("Run the amplitude sweep before plotting.")

        plotter = ReadoutAmplitudeSweepPlotter(
            self.qubit_names,
            self.measured_amplitudes,
            self.fidelities,
        )
        plotter.initial_amplitudes = self.initial_amplitudes
        plotter.readout_lengths = self.readout_lengths
        plotter.reset_label = self.reset_label
        plotter.fidelity_errors = self.fidelity_errors
        plotter.separations = self.separations
        plotter.roundnesses = self.roundnesses

        return plotter.plot()

    def analyze(self) -> dict[str, Any]:
        if not self.measured_amplitudes:
            raise RuntimeError("Run the amplitude sweep before analyzing results.")

        summary = ReadoutAmplitudeSweepAnalysis(
            qubit_names=self.qubit_names,
            amplitudes=self.measured_amplitudes,
            fidelities=self.fidelities,
            initial_amplitudes=self.initial_amplitudes,
        ).summary()
        summary["interrupted"] = self.interrupted
        summary["interrupt_reason"] = self.interrupt_reason
        summary["measurement_errors"] = self.measurement_errors
        summary["resonator_frequencies"] = self.resonator_frequencies
        summary["readout_frequencies"] = self.readout_frequencies
        summary["roundnesses"] = self.roundnesses
        return summary

    def save_results(
        self,
        output_dir: str | Path = Path("data") / "readout_optimize",
        figure: Figure | None = None,
        run_dir: str | Path | None = None,
    ) -> str:
        if not self.measured_amplitudes:
            raise RuntimeError("Run the amplitude sweep before saving results.")

        saver = ReadoutAmplitudeSweepSaver(
            qubit_names=self.qubit_names,
            amplitudes=self.measured_amplitudes,
            fidelities=self.fidelities,
            results=self.results,
            profile=self.profile,
            initial_amplitudes=self.initial_amplitudes,
            readout_lengths=self.readout_lengths,
            fidelity_errors=self.fidelity_errors,
            separations=self.separations,
            roundnesses=self.roundnesses,
            resonator_frequencies=self.resonator_frequencies,
            readout_frequencies=self.readout_frequencies,
            profile_path=self.settings.profile_path,
        )
        saver.iq_blob_figures = self.iq_blob_figures
        saver.kernel_figures = self.kernel_figures
        saver.resonator_figures = self.resonator_figures
        saver.interrupted = self.interrupted
        saver.interrupt_reason = self.interrupt_reason
        saver.reset_label = self.reset_label
        saver.scan_method = str(ReadoutScanMethod(self.settings.method).value)
        saver.measurement_errors = self.measurement_errors

        run_dir = saver.save(
            output_dir=output_dir,
            figure=figure,
            run_dir=run_dir or self.run_dir,
        )
        if self.live_plotter is not None:
            self.live_plotter.write_standalone_html(status="saved")
        print(f"Saved readout optimizer results to {Path(run_dir).resolve()}")
        return run_dir

    def _auto_save_after_run(self) -> None:
        if self.settings.submit_only or not self.settings.auto_save_results:
            return
        if not self.measured_amplitudes:
            return

        figure = self.plot()
        self.figure = figure
        self.save_results(
            output_dir=self.settings.live_html_output_dir,
            figure=figure,
        )
        if self.settings.close_auto_saved_figure:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def _measure_amplitude(self, amplitude: float) -> float:
        amplitude = float(amplitude)
        if amplitude in self.results and not self.settings.submit_only:
            return self._score_result(self.results[amplitude])

        self._set_readout_amplitude(amplitude)

        workflow = ReadoutFidelityWorkflow(
            qubit_names=self.qubit_names,
            profile=self.profile,
            task_manager=self.task_manager,
            settings=self._workflow_settings_for_measurement(),
        )

        self.workflows[amplitude] = workflow
        try:
            result = workflow.run()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as error:
            if not self.settings.continue_on_measurement_error:
                raise

            return self._record_failed_measurement(amplitude, error)

        if self.settings.submit_only:
            self._record_submitted_tasks(amplitude, workflow)
            self.results[amplitude] = {
                "status": "submitted_pending_results",
                "tasks": list(workflow.submitted_tasks),
            }
            return float(self.settings.failed_measurement_fidelity)

        self.results[amplitude] = result
        self.measured_amplitudes.append(amplitude)
        self._record_fidelities(result)
        self._record_resonator_frequencies(workflow)
        self.readout_frequencies = self._readout_frequencies()
        self._record_iq_blob_figures(amplitude, workflow)
        self._record_kernel_figures(amplitude, workflow)
        self._record_resonator_figures(amplitude, workflow)
        self._update_live_plotter(latest_amplitude=amplitude)

        return self._score_result(result)

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
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        amplitude_index = len(self.submitted_amplitudes)
        self.submitted_amplitudes.append(float(amplitude))
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
            self.submitted_task_entries.append(task_entry)

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
        self.resonator_frequencies = {qubit_name: [] for qubit_name in self.qubit_names}
        self.iq_blob_figures = {}
        self.kernel_figures = {}
        self.resonator_figures = {}
        self.measurement_errors = {}
        self.run_dir = Path(run_dir)

        for amplitude in sorted(tasks_by_amplitude):
            self._set_readout_amplitude(amplitude)
            workflow = ReadoutFidelityWorkflow(
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

    def _score_result(self, result: dict[str, Any]) -> float:
        if "iq_blobs" not in result:
            return float(self.settings.failed_measurement_fidelity)

        iq_results = result["iq_blobs"]
        return float(
            np.mean(
                [
                    iq_results[qubit_name]["readout_fidelity"]
                    for qubit_name in self.qubit_names
                ]
            )
        )

    def _record_failed_measurement(
        self,
        amplitude: float,
        error: Exception,
    ) -> float:
        error_message = f"{type(error).__name__}: {error}"
        self.results[amplitude] = {"error": error_message}
        self.measurement_errors[amplitude] = error_message
        self.measured_amplitudes.append(amplitude)

        for qubit_name in self.qubit_names:
            self.fidelities[qubit_name].append(float(self.settings.failed_measurement_fidelity))
            self.fidelity_errors[qubit_name].append(None)
            self.separations[qubit_name].append(None)
            self.roundnesses[qubit_name].append(None)
            self.resonator_frequencies[qubit_name].append(None)

        print(
            f"\nMeasurement failed at amplitude {amplitude:.6g}; "
            f"recorded fidelity={self.settings.failed_measurement_fidelity}. "
            f"{error_message}"
        )
        self._update_live_plotter(latest_amplitude=amplitude)
        return float(self.settings.failed_measurement_fidelity)

    def _start_live_plotter(self) -> None:
        if not self.settings.use_live_html_plotter:
            self.live_plotter = None
            return

        self.run_dir = create_readout_run_dir(
            output_dir=self.settings.live_html_output_dir,
            scan_method=str(ReadoutScanMethod(self.settings.method).value),
            qubit_names=self.qubit_names,
        )
        self.live_plotter = ReadoutLiveHtmlPlotter(
            output_dir=self.run_dir,
            refresh_interval_seconds=self.settings.live_html_refresh_seconds,
            open_browser=self.settings.live_html_open_browser,
        )
        qubits = ", ".join(self.qubit_names)
        self.live_plotter.start(
            title=f"Readout amplitude optimizer - {qubits}",
            workflow_label=self._workflow_label(),
            scan_method=str(ReadoutScanMethod(self.settings.method).value),
            optimization_parameters=self._optimization_parameters(),
        )

    def _update_live_plotter(self, latest_amplitude: float | None = None) -> None:
        if self.live_plotter is None:
            return

        latest_iq_figures = None
        if latest_amplitude is not None:
            latest_iq_figures = self.iq_blob_figures.get(float(latest_amplitude))
        latest_kernel_figures = None
        if latest_amplitude is not None:
            latest_kernel_figures = self.kernel_figures.get(float(latest_amplitude))
        latest_resonator_figures = None
        if latest_amplitude is not None:
            latest_resonator_figures = self.resonator_figures.get(float(latest_amplitude))

        self.live_plotter.update(
            qubit_names=self.qubit_names,
            amplitudes=self.measured_amplitudes,
            fidelities=self.fidelities,
            fidelity_errors=self.fidelity_errors,
            separations=self.separations,
            roundnesses=self.roundnesses,
            resonator_frequencies=self.resonator_frequencies,
            initial_amplitudes=self.initial_amplitudes,
            readout_lengths=self.readout_lengths,
            readout_frequencies=self.readout_frequencies,
            reset_label=self.reset_label,
            latest_amplitude=latest_amplitude,
            latest_iq_figures=latest_iq_figures,
            latest_kernel_figures=latest_kernel_figures,
            latest_resonator_figures=latest_resonator_figures,
            workflow_label=self._workflow_label(),
            scan_method=str(ReadoutScanMethod(self.settings.method).value),
            optimization_parameters=self._optimization_parameters(),
        )

    def _finish_live_plotter(self) -> None:
        if self.live_plotter is not None:
            self.live_plotter.finish()

    def _set_readout_amplitude(self, amplitude: float) -> None:
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            readout_pulse.readout_amplitude = amplitude

    def _readout_amplitudes(self) -> dict[str, float]:
        amplitudes = {}
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            amplitudes[qubit_name] = float(readout_pulse.readout_amplitude)
        return amplitudes

    def _readout_lengths(self) -> dict[str, float]:
        lengths = {}
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            lengths[qubit_name] = float(readout_pulse.readout_duration)
        return lengths

    def _readout_frequencies(self) -> dict[str, float]:
        frequencies = {}
        for qubit_name in self.qubit_names:
            frequency = getattr(
                self.profile.qubits[qubit_name].readout_resonator_frequency,
                "value",
                None,
            )
            if frequency is not None:
                frequencies[qubit_name] = float(frequency)
        return frequencies

    def _reset_label(self) -> str:
        reset = self.settings.workflow_settings.reset
        if reset.reset_type == ResetType.ACTIVE:
            return f"active reset on ({reset.reset_num}x)"

        return "active reset off"

    def _record_fidelities(self, result: dict[str, Any]) -> None:
        iq_results = result["iq_blobs"]

        for qubit_name in self.qubit_names:
            qubit_result = iq_results[qubit_name]
            fidelity = qubit_result["readout_fidelity"]
            self.fidelities[qubit_name].append(fidelity)
            self.fidelity_errors[qubit_name].append(
                self._first_metric_value(
                    qubit_result,
                    [
                        "readout_fidelity_std",
                        "readout_fidelity_error",
                        "readout_fidelity_err",
                        "average_readout_fidelity_std",
                        "averaged_readout_fidelity_std",
                        "fidelity_std",
                        "fidelity_error",
                    ],
                )
            )
            self.separations[qubit_name].append(
                self._first_metric_value(
                    qubit_result,
                    [
                        "separation",
                        "readout_separation",
                        "iq_separation",
                        "state_separation",
                    ],
                )
            )
            self.roundnesses[qubit_name].append(
                self._first_metric_value(
                    qubit_result,
                    [
                        "average_roundness",
                        "averaged_roundness",
                        "roundness",
                    ],
                )
            )

    def _record_resonator_frequencies(
        self,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        handlers = self._workflow_handlers(workflow, "resonator")
        for qubit_name in self.qubit_names:
            frequency = None
            for handler in handlers:
                handler_data = getattr(handler, "data", {}) or {}
                qubit_data = handler_data.get(qubit_name, {}) or {}
                if "optimal_resonance_freq" in qubit_data:
                    frequency = float(qubit_data["optimal_resonance_freq"])
                    break
            self.resonator_frequencies[qubit_name].append(frequency)

    def _first_metric_value(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> float | None:
        for key in keys:
            if key in data and data[key] is not None:
                return float(data[key])
        return None

    def _record_iq_blob_figures(
        self,
        amplitude: float,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        handler = workflow.iq_blobs_handler
        if handler is None:
            return

        figures = self._handler_figures(handler)
        if figures:
            self.iq_blob_figures[float(amplitude)] = figures

    def _record_kernel_figures(
        self,
        amplitude: float,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        figures = self._workflow_figures(workflow, "kernel")
        if figures:
            self.kernel_figures[float(amplitude)] = figures

    def _record_resonator_figures(
        self,
        amplitude: float,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        figures = self._workflow_figures(workflow, "resonator")
        if figures:
            self.resonator_figures[float(amplitude)] = figures

    def _workflow_handlers(
        self,
        workflow: ReadoutFidelityWorkflow,
        experiment: str,
    ) -> list[Any]:
        plural_name = f"{experiment}_handlers"
        handlers = list(getattr(workflow, plural_name, []) or [])
        if handlers:
            return handlers

        handler = getattr(workflow, f"{experiment}_handler", None)
        return [handler] if handler is not None else []

    def _workflow_figures(
        self,
        workflow: ReadoutFidelityWorkflow,
        experiment: str,
    ) -> list[Figure]:
        figures = []
        for handler in self._workflow_handlers(workflow, experiment):
            figures.extend(self._handler_figures(handler))
        return self._unique_figures(figures)

    def _workflow_label(self) -> str:
        enabled_nodes = []
        workflow_settings = self.settings.workflow_settings
        if workflow_settings.run_resonator:
            enabled_nodes.append("resonator")
        if workflow_settings.run_kernels:
            enabled_nodes.append("kernels")
        if workflow_settings.run_iq_blobs:
            enabled_nodes.append("iq blobs")

        mode = "local emulation" if workflow_settings.do_emulation else "task manager"
        return f"{' -> '.join(enabled_nodes) or 'none'} ({mode})"

    def _optimization_parameters(self) -> dict[str, Any]:
        method = ReadoutScanMethod(self.settings.method)
        parameters: dict[str, Any] = {
            "amplitudes": len(list(self.settings.amplitudes)),
        }
        if method == ReadoutScanMethod.ZOOM_IN:
            parameters.update(
                {
                    "zoom_in_iterations": self.settings.zoom_in_iterations,
                    "zoom_in_shrink_factor": self.settings.zoom_in_shrink_factor,
                }
            )
        elif method == ReadoutScanMethod.GRADIENT:
            parameters.update(
                {
                    "gradient_max_iterations": self.settings.gradient_max_iterations,
                    "gradient_initial_step": self.settings.gradient_initial_step,
                    "gradient_min_step": self.settings.gradient_min_step,
                    "gradient_fidelity_tolerance": self.settings.gradient_fidelity_tolerance,
                }
            )
        elif method == ReadoutScanMethod.GOLDEN_SECTION:
            parameters.update(
                {
                    "golden_section_max_iterations": self.settings.golden_section_max_iterations,
                    "golden_section_interval_tolerance": self.settings.golden_section_interval_tolerance,
                    "fidelity_tolerance": self.settings.gradient_fidelity_tolerance,
                }
            )

        return parameters

    def _handler_figures(self, handler: Any) -> list[Figure]:
        figures = []
        for attribute_name in ("workflow_figures", "figs", "figures"):
            figures.extend(self._extract_figures(getattr(handler, attribute_name, None)))

        figure = getattr(handler, "fig", None)
        figures.extend(self._extract_figures(figure))

        return self._unique_figures(figures)

    def _unique_figures(self, figures: list[Figure]) -> list[Figure]:
        unique_figures = []
        seen_ids = set()
        for figure in figures:
            figure_id = id(figure)
            if figure_id in seen_ids:
                continue
            unique_figures.append(figure)
            seen_ids.add(figure_id)

        return unique_figures

    def _extract_figures(self, value: Any) -> list[Figure]:
        if isinstance(value, Figure):
            return [value]
        if hasattr(value, "figure") and isinstance(value.figure, Figure):
            return [value.figure]
        if isinstance(value, dict):
            figures = []
            for item in value.values():
                figures.extend(self._extract_figures(item))
            return figures
        if isinstance(value, (list, tuple, set)):
            figures = []
            for item in value:
                figures.extend(self._extract_figures(item))
            return figures
        return []

    def _fill_unfinished_fidelities(self) -> None:
        for amplitude in self._unfinished_amplitudes():
            self.measured_amplitudes.append(float(amplitude))
            for qubit_name in self.qubit_names:
                self.fidelities[qubit_name].append(float(self.settings.unfinished_fidelity))
                self.fidelity_errors[qubit_name].append(None)
                self.separations[qubit_name].append(None)
                self.roundnesses[qubit_name].append(None)
                self.resonator_frequencies[qubit_name].append(None)

    def _unfinished_amplitudes(self) -> list[float]:
        configured_amplitudes = [float(amplitude) for amplitude in self.settings.amplitudes]
        measured = set(self.measured_amplitudes)
        return [amplitude for amplitude in configured_amplitudes if amplitude not in measured]

    def _show_progress(self, index: int, total: int, amplitude: float) -> None:
        if not self.settings.show_progress:
            return

        width = 30
        completed = int(width * (index - 1) / total)
        bar = "#" * completed + "-" * (width - completed)
        percent = 100 * (index - 1) / total
        print(
            f"\rReadout optimization [{bar}] {index - 1}/{total} "
            f"({percent:5.1f}%) current={amplitude:.6g}",
            end="",
            flush=True,
        )

    def _finish_progress(self, total: int) -> None:
        if not self.settings.show_progress:
            return

        bar = "#" * 30
        print(f"\rReadout optimization [{bar}] {total}/{total} (100.0%) complete")


if __name__ == "__main__":
    from qratena.system.components_params.reset_settings import ResetSettings
    from qratena.util.enums import ResetType

    from optimize.readout.readout_amplitude_optimizer import (
        ReadoutAmplitudeSweepSettings,
        ReadoutAmplitudeSweepWorkflow,
    )
    from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
    from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
    from resources.load_profile import load_profile, load_task_manager

    profile = load_profile("main")
    task_manager = load_task_manager()

    workflow_settings = ReadoutFidelityWorkflowSettings(
        profile_name="main",
        do_emulation=False,
        run_resonator=False,
        run_kernels=True,
        run_iq_blobs=True,
        do_plotting=False,
        show_handler_output=False,
        low_priority_tasks=True,
        states=["g", "e"],
        reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=5),
    )

    optimizer_settings = ReadoutAmplitudeSweepSettings(
        amplitudes=np.arange(0.005, 0.1, 0.01),
        method=ReadoutScanMethod.SWEEP,
        use_live_html_plotter=False,
        workflow_settings=workflow_settings,
        submit_only=True,
    )

    optimizer = ReadoutAmplitudeSweepWorkflow(
        qubit_names=["q6"],
        profile=profile,
        task_manager=task_manager,
        settings=optimizer_settings,
    )

    optimizer.run()
    figure = optimizer.plot()
    run_dir = optimizer.save_results(figure=figure)


    print(run_dir)
    
