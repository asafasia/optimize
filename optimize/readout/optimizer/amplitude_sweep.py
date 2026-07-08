from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from workbench_bootstrap import setup_workbench_environment

setup_workbench_environment()

from matplotlib.figure import Figure
from qratena.system.components_params.profile import Profile

from optimize.readout.optimizer.figures import ReadoutFigureMixin
from optimize.readout.optimizer.metrics import ReadoutMetricsMixin
from optimize.readout.optimizer.settings import ReadoutAmplitudeSweepSettings
from optimize.readout.optimizer.profile_access import ReadoutProfileAccessMixin
from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
)
from optimize.readout.optimizer.submitted_runs import SubmittedReadoutRunMixin
from optimize.readout.optimizer.live_html_plotter import ReadoutLiveHtmlPlotter
from optimize.readout.optimizer.scan_methods import scan_method_for
from optimize.readout.optimizer.scan_types import ReadoutScanMethod
from optimize.readout.optimizer.analysis import ReadoutAmplitudeSweepAnalysis
from optimize.readout.optimizer.artifacts import (
    ReadoutAmplitudeSweepSaver,
    create_readout_run_dir,
)
from optimize.readout.optimizer.plotter import ReadoutAmplitudeSweepPlotter

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

        workflow = self._create_readout_workflow(
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

    def _create_readout_workflow(
        self,
        *,
        qubit_names: list[str],
        profile: Profile,
        task_manager: TaskSubmitterAsync,
        settings: Any,
    ) -> ReadoutFidelityWorkflow:
        return ReadoutFidelityWorkflow(
            qubit_names=qubit_names,
            profile=profile,
            task_manager=task_manager,
            settings=settings,
        )

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
