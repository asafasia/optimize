from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from qigeon.io.task_submitter import TaskSubmitterAsync
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ResetType

from workbench.optimize.readout.utils.readout_scan_methods import scan_method_for
from workbench.optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from workbench.optimize.readout.utils.readout_live_html_plotter import ReadoutLiveHtmlPlotter
from workbench.optimize.readout.utils.readout_sweep_analysis import ReadoutAmplitudeSweepAnalysis
from workbench.optimize.readout.utils.readout_sweep_artifacts import (
    ReadoutAmplitudeSweepSaver,
    create_readout_run_dir,
)
from workbench.optimize.readout.utils.readout_sweep_plotter import ReadoutAmplitudeSweepPlotter
from workbench.optimize.readout.readout_workflow import ReadoutFidelityWorkflow, ReadoutFidelityWorkflowSettings
from workbench.resources.load_profile import load_task_manager


@dataclass(slots=True)
class ReadoutAmplitudeSweepSettings:
    amplitudes: Any
    method: ReadoutScanMethod | str = ReadoutScanMethod.SWEEP
    zoom_in_iterations: int = 3
    zoom_in_shrink_factor: float = 0.5
    gradient_max_iterations: int = 5
    gradient_initial_step: float | None = None
    gradient_min_step: float = 0.001
    gradient_fidelity_tolerance: float = 0.01
    golden_section_max_iterations: int = 8
    golden_section_interval_tolerance: float = 0.001
    fill_unfinished_on_interrupt: bool = True
    unfinished_fidelity: float = 0.5
    continue_on_measurement_error: bool = False
    failed_measurement_fidelity: float = 0.5
    profile_path: str | Path | None = None
    show_progress: bool = True
    use_live_html_plotter: bool = True
    live_html_output_dir: str | Path = Path("data") / "readout_optimize"
    live_html_refresh_seconds: float = 1.0
    live_html_open_browser: bool = False
    workflow_settings: ReadoutFidelityWorkflowSettings = field(
        default_factory=ReadoutFidelityWorkflowSettings
    )


class ReadoutAmplitudeSweepWorkflow:
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
        self.initial_amplitudes = self._readout_amplitudes()
        self.readout_lengths = self._readout_lengths()
        self.readout_frequencies = self._readout_frequencies()
        self.reset_label = self._reset_label()
        self.interrupted = False
        self.interrupt_reason: str | None = None
        self.live_plotter: ReadoutLiveHtmlPlotter | None = None
        self.run_dir: Path | None = None

    def run(self) -> dict[float, dict[str, Any]]:
        self._prepare_profile_for_states()
        self.workflows = {}
        self.results = {}
        self.measured_amplitudes = []
        self.fidelities = {qubit_name: [] for qubit_name in self.qubit_names}
        self.fidelity_errors = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.separations = {qubit_name: [] for qubit_name in self.qubit_names}
        self.roundnesses = {qubit_name: [] for qubit_name in self.qubit_names}
        self.resonator_frequencies = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.iq_blob_figures = {}
        self.kernel_figures = {}
        self.resonator_figures = {}
        self.measurement_errors = {}
        self.interrupted = False
        self.interrupt_reason = None
        self.run_dir = None

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
            self._finish_live_plotter()

        self._finish_progress(len(self.measured_amplitudes))
        return self.results

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
            raise RuntimeError(
                "Run the amplitude sweep before analyzing results.")

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
    ) -> str:
        if not self.measured_amplitudes:
            raise RuntimeError(
                "Run the amplitude sweep before saving results.")

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
            run_dir=self.run_dir,
        )
        if self.live_plotter is not None:
            self.live_plotter.write_standalone_html(status="saved")
        return run_dir

    def _measure_amplitude(self, amplitude: float) -> float:
        amplitude = float(amplitude)
        if amplitude in self.results:
            return self._score_result(self.results[amplitude])

        self._set_readout_amplitude(amplitude)

        workflow = ReadoutFidelityWorkflow(
            qubit_names=self.qubit_names,
            profile=self.profile,
            task_manager=self.task_manager,
            settings=self.settings.workflow_settings,
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
            self.fidelities[qubit_name].append(
                float(self.settings.failed_measurement_fidelity)
            )
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
            latest_iq_figures = self.iq_blob_figures.get(
                float(latest_amplitude))
        latest_kernel_figures = None
        if latest_amplitude is not None:
            latest_kernel_figures = self.kernel_figures.get(
                float(latest_amplitude))
        latest_resonator_figures = None
        if latest_amplitude is not None:
            latest_resonator_figures = self.resonator_figures.get(
                float(latest_amplitude))

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
            readout_pulse = self.profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout][SUPPORTED_PULSE_SHAPES.const]
            readout_pulse.readout_amplitude = amplitude

    def _readout_amplitudes(self) -> dict[str, float]:
        amplitudes = {}
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout][SUPPORTED_PULSE_SHAPES.const]
            amplitudes[qubit_name] = float(readout_pulse.readout_amplitude)
        return amplitudes

    def _readout_lengths(self) -> dict[str, float]:
        lengths = {}
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout][SUPPORTED_PULSE_SHAPES.const]
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
        handler = workflow.resonator_handler
        for qubit_name in self.qubit_names:
            frequency = None
            if handler is not None:
                handler_data = getattr(handler, "data", {}) or {}
                qubit_data = handler_data.get(qubit_name, {}) or {}
                if "optimal_resonance_freq" in qubit_data:
                    frequency = float(qubit_data["optimal_resonance_freq"])
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
        handler = workflow.kernel_handler
        if handler is None:
            return

        figures = self._handler_figures(handler)
        if figures:
            self.kernel_figures[float(amplitude)] = figures

    def _record_resonator_figures(
        self,
        amplitude: float,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        handler = workflow.resonator_handler
        if handler is None:
            return

        figures = self._handler_figures(handler)
        if figures:
            self.resonator_figures[float(amplitude)] = figures

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
            figures.extend(self._extract_figures(
                getattr(handler, attribute_name, None)))

        figure = getattr(handler, "fig", None)
        figures.extend(self._extract_figures(figure))

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
                self.fidelities[qubit_name].append(
                    float(self.settings.unfinished_fidelity)
                )
                self.fidelity_errors[qubit_name].append(None)
                self.separations[qubit_name].append(None)
                self.roundnesses[qubit_name].append(None)
                self.resonator_frequencies[qubit_name].append(None)

    def _unfinished_amplitudes(self) -> list[float]:
        configured_amplitudes = [
            float(amplitude)
            for amplitude in self.settings.amplitudes
        ]
        measured = set(self.measured_amplitudes)
        return [
            amplitude
            for amplitude in configured_amplitudes
            if amplitude not in measured
        ]

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
        print(
            f"\rReadout optimization [{bar}] {total}/{total} (100.0%) complete")


if __name__ == "__main__":
    from workbench.resources import *
    from workbench.resources.load_profile import load_profile

    profile = load_profile()
    profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

    task_manager = load_task_manager()

    workflow_settings = ReadoutFidelityWorkflowSettings(
        profile_name="main",
        do_emulation=False,
        run_resonator=False,
        run_kernels=True,
        run_iq_blobs=True,
        do_plotting=False,
        show_handler_output=False,
        reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
    )

    optimizer_settings = ReadoutAmplitudeSweepSettings(
        amplitudes=np.linspace(0.001, 0.11, 4),
        workflow_settings=workflow_settings,
        method=ReadoutScanMethod.ZOOM_IN,
        zoom_in_iterations=4

    )
    qubits = sorted([q for q in profile.qubits.keys()],
                    key=lambda q: int(q[1:]))

    # qubits = ['q6']

    for qubit_name in qubits:

        optimizer = ReadoutAmplitudeSweepWorkflow(
            qubit_names=[qubit_name],
            profile=profile,
            task_manager=task_manager,
            settings=optimizer_settings,
        )

        optimizer.run()
        fig = optimizer.plot()
        optimizer.save_results(figure=fig)
        plt.show()

# %%
