from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from qigeon.io.task_submitter import TaskSubmitterAsync
from qratena.system.components_params.profile import Profile
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES

from optimize.readout_sweep_artifacts import (
    ReadoutAmplitudeSweepAnalysis,
    ReadoutAmplitudeSweepPlotter,
    ReadoutAmplitudeSweepSaver,
)
from optimize.readout_workflow import ReadoutFidelityWorkflow, ReadoutFidelityWorkflowSettings
from resources.load_profile import load_task_manager


@dataclass(slots=True)
class ReadoutAmplitudeSweepSettings:
    amplitudes: Any
    profile_path: str | Path | None = None
    show_progress: bool = True
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
        self.fidelities: dict[str, list[float]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.iq_blob_figures: dict[float, list[Figure]] = {}
        self.initial_amplitudes = self._readout_amplitudes()
        self.readout_lengths = self._readout_lengths()

    def run(self) -> dict[float, dict[str, Any]]:
        self.workflows = {}
        self.results = {}
        self.fidelities = {qubit_name: [] for qubit_name in self.qubit_names}
        self.iq_blob_figures = {}

        amplitudes = list(self.settings.amplitudes)
        for index, amplitude in enumerate(amplitudes, start=1):
            self._show_progress(index, len(amplitudes), float(amplitude))
            self._set_readout_amplitude(amplitude)

            workflow = ReadoutFidelityWorkflow(
                qubit_names=self.qubit_names,
                profile=self.profile,
                task_manager=self.task_manager,
                settings=self.settings.workflow_settings,
            )

            self.workflows[amplitude] = workflow
            result = workflow.run()
            self.results[amplitude] = result
            self._record_fidelities(result)
            self._record_iq_blob_figures(amplitude, workflow)

        self._finish_progress(len(amplitudes))
        return self.results

    def plot(self) -> Figure:
        if not self.results:
            raise RuntimeError("Run the amplitude sweep before plotting.")

        plotter = ReadoutAmplitudeSweepPlotter(
            self.qubit_names,
            self.settings.amplitudes,
            self.fidelities,
        )
        plotter.initial_amplitudes = self.initial_amplitudes
        plotter.readout_lengths = self.readout_lengths

        return plotter.plot()

    def analyze(self) -> dict[str, Any]:
        if not self.results:
            raise RuntimeError("Run the amplitude sweep before analyzing results.")

        return ReadoutAmplitudeSweepAnalysis(
            qubit_names=self.qubit_names,
            amplitudes=self.settings.amplitudes,
            fidelities=self.fidelities,
            initial_amplitudes=self.initial_amplitudes,
        ).summary()

    def save_results(
        self,
        output_dir: str | Path = Path("data") / "readout_optimize",
        figure: Figure | None = None,
    ) -> str:
        if not self.results:
            raise RuntimeError("Run the amplitude sweep before saving results.")

        saver = ReadoutAmplitudeSweepSaver(
            qubit_names=self.qubit_names,
            amplitudes=self.settings.amplitudes,
            fidelities=self.fidelities,
            results=self.results,
            profile=self.profile,
            initial_amplitudes=self.initial_amplitudes,
            readout_lengths=self.readout_lengths,
            profile_path=self.settings.profile_path,
        )
        saver.iq_blob_figures = self.iq_blob_figures

        return saver.save(output_dir=output_dir, figure=figure)

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

    def _record_fidelities(self, result: dict[str, Any]) -> None:
        iq_results = result["iq_blobs"]

        for qubit_name in self.qubit_names:
            fidelity = iq_results[qubit_name]["readout_fidelity"]
            self.fidelities[qubit_name].append(fidelity)

    def _record_iq_blob_figures(
        self,
        amplitude: float,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        handler = workflow.iq_blobs_handler
        if handler is None:
            return

        figures = getattr(handler, "workflow_figures", [])
        if figures:
            self.iq_blob_figures[float(amplitude)] = figures

    def _show_progress(self, index: int, total: int, amplitude: float) -> None:
        if not self.settings.show_progress:
            return

        width = 30
        completed = int(width * (index - 1) / total)
        bar = "#" * completed + "-" * (width - completed)
        percent = 100 * (index - 1) / total
        print(
            f"\rAmplitude sweep [{bar}] {index - 1}/{total} "
            f"({percent:5.1f}%) current={amplitude:.6g}",
            end="",
            flush=True,
        )

    def _finish_progress(self, total: int) -> None:
        if not self.settings.show_progress:
            return

        bar = "#" * 30
        print(f"\rAmplitude sweep [{bar}] {total}/{total} (100.0%) complete")


if __name__ == "__main__":
    from resources import *
    from resources.load_profile import load_profile

    profile = load_profile()
    
    task_manager = load_task_manager()


    workflow_settings = ReadoutFidelityWorkflowSettings(
        profile_name="main",
        do_emulation=False,
        run_resonator=False,
        run_kernels=True,
        run_iq_blobs=True,
        display_handler_plots=False,
        suppress_handler_output=False,
    )

    optimizer_settings = ReadoutAmplitudeSweepSettings(
        amplitudes=np.linspace(0.001, 0.2, 30),
        workflow_settings=workflow_settings,
    )

    qubits = ['q5']
    
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
