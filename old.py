from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from matplotlib.figure import Figure
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.dsl.session import Session

from qratena.experiments.base_experiment import ResetSettings
from qratena.experiments.iq_blobs import (
    IQBlobsHandler,
    IQBlobsSettings,
)
from qratena.experiments.kernel_traces_calculation import (
    KernelTracesCalculationHandler,
    KernelTracesSettings,
)
from qratena.experiments.resonator_spectroscopy import (
    ResonatorSpectroscopyHandler,
    ResonatorSpectroscopySettings,
)
from qratena.optimize.readout.readout_optimization_result import (
    AdaptiveReadoutOptimizationAnalysis,
    ReadoutOptimizationAnalysis,
    ReadoutOptimizationPlotter,
    ReadoutOptimizationResult,
    ReadoutOptimizationSample,
)
from qratena.optimize.readout.gradient import (
    FiniteDifferenceAmplitudeOptimizer,
    ObjectiveResult,
)
from qratena.system.components_params.profile import Profile
from qratena.system.qratena_platform import create_platform
from qratena.util.enums import (
    ExportationMethod,
    SUPPORTED_PULSE_SHAPES,
    SUPPORTED_PULSE_TYPES,
    ResetType,
    UpdateParamsMethod,
)
from qratena.util.kernels import pickle_kernels, unpickle_kernels
from qratena.util.sweeps_utils import MidI


@dataclass(slots=True)
class OptimizationSettings:
    color_logs: bool = True
    console_logs: bool = True
    do_emulation: bool = True
    num_resonator_shots: int = 100
    num_kernel_shots: int = 3000
    num_iq_shots: int = 5000
    states_to_measure: list[str] = field(default_factory=lambda: ["g", "e"])
    plot: bool = True
    reset: ResetSettings = field(default_factory=ResetSettings)
    method: Literal["gradient", "sweep"] = "gradient"
    amplitude_bounds: tuple[float, float] = (0.0, 0.2)
    max_iterations: int = 8
    initial_step: float = 0.02
    finite_difference_step: float = 0.005
    min_step: float = 0.001
    score_tolerance: float = 1e-3
    recalibrate_frequency_each_step: bool = True
    recalibrate_kernels_each_step: bool = True


class ReadoutOptimizationHandler:
    """Optimize readout amplitude by running resonator spectroscopy, kernel traces,
    and IQ blobs for each amplitude value.
    """

    def __init__(
        self,
        qubit_names: list[str],
        amplitudes: NDArray[np.floating],
        readout_length: float = 500e-9,
        profile: Profile | None = None,
        settings: OptimizationSettings | None = None,
    ) -> None:

        self.qubit_names = qubit_names
        self.profile = profile or Profile.default()
        self.settings = settings or OptimizationSettings()

        self.readout_length = readout_length
        self.amplitudes = amplitudes

        self.initial_traces = {
            qubit_name: unpickle_kernels(qubit_name=qubit_name)
            for qubit_name in self.qubit_names
        }
        self.initial_amplitudes: dict[str, float] = (
            self._initial_readout_amplitudes()
        )

        self._update_readout_length(self.readout_length)

        self.platform = create_platform(self.profile)
        self.session = Session(self.platform.setup, configure_logging=False)
        self.session.connect(do_emulation=self.settings.do_emulation)

        self.fidelities: dict[str, list[float]] = {
            qubit_name: [] for qubit_name in self.qubit_names
        }
        self.samples: list[ReadoutOptimizationSample] = []

    def __enter__(self) -> ReadoutOptimizationHandler:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.session.disconnect()
        for qubit_name, traces in self.initial_traces.items():
            pickle_kernels(
                qubit_name=qubit_name,
                qubit_traces=traces
            )

    def _set_readout_amplitude(self, amplitude: float) -> None:
        for qubit_name in self.qubit_names:
            qubit = self.profile.qubits[qubit_name]
            readout_pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            readout_pulse.readout_amplitude = amplitude

    def _update_readout_frequency(self, frequency: float) -> None:
        for qubit_name in self.qubit_names:
            qubit = self.profile.qubits[qubit_name]
            qubit.readout_resonator_frequency.value = frequency

    def _update_readout_length(self, readout_length: float) -> None:
        for qubit_name in self.qubit_names:
            qubit = self.profile.qubits[qubit_name]
            readout_pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            readout_pulse.readout_duration = readout_length

    def _initial_readout_amplitudes(self) -> dict[str, float]:
        initial_amplitudes = {}
        for qubit_name in self.qubit_names:
            qubit = self.profile.qubits[qubit_name]
            readout_pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            amplitude = readout_pulse.readout_amplitude
            initial_amplitudes[qubit_name] = amplitude
        return initial_amplitudes

    def run_workflow(self) -> dict[str, Any]:
        
        settings = IQBlobsSettings(
            num_shots=self.settings.num_iq_shots,
            color_logs=False,
            console_logs=False,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            exportation_method=ExportationMethod.NONE,
            states_to_measure=self.settings.states_to_measure,
            do_emulation=self.settings.do_emulation,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            configure_logging=False,
            reset=self.settings.reset,
            plot=self.settings.plot,
        )

        handler = IQBlobsHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            profile=self.profile,
        )

        handler.run()

        for qubit_name in self.qubit_names:
            matrix = handler.data[qubit_name]["readout_fidelity_matrix"]

            mean_fidelity = float(
                np.mean(
                    [
                        matrix[0, 0],
                        matrix[1, 1],
                    ]
                )
            )

            self.fidelities[qubit_name].append(mean_fidelity)
        return {
            qubit_name: self.fidelities[qubit_name][-1]
            for qubit_name in self.qubit_names
        }

    def _measure_amplitude(
        self,
        amplitude: float,
        iteration: int,
        label: str,
    ) -> ObjectiveResult:
        self._set_readout_amplitude(float(amplitude))
        if self.settings.recalibrate_frequency_each_step:
            self._resonator_spectroscopy()
        if self.settings.recalibrate_kernels_each_step:
            self._run_kernel_traces_calculation()
        fidelities = self._run_iq_blobs()
        score = float(np.mean(list(fidelities.values())))
        self.samples.append(
            ReadoutOptimizationSample(
                amplitude=float(amplitude),
                fidelities=fidelities,
                score=score,
                iteration=iteration,
                label=label,
            )
        )
        return score, fidelities

    def run(self) -> ReadoutOptimizationResult:
        if self.settings.method == "sweep":
            return self._run_sweep()

        return self._run_gradient_optimization()

    def _run_sweep(self) -> ReadoutOptimizationResult:
        for iteration, amplitude in enumerate(self.amplitudes):
            self._measure_amplitude(
                amplitude=float(amplitude),
                iteration=iteration,
                label="sweep",
            )

        analysis = ReadoutOptimizationAnalysis(
            amplitudes=self.amplitudes,
            fidelities=self.fidelities,
            initial_amplitudes=self.initial_amplitudes,
        )

        self.result = analysis.analyze()
        self.samples = self.result.samples
        best_amplitude = float(
            np.mean(list(self.result.optimized_amplitudes.values()))
        )
        self._set_readout_amplitude(best_amplitude)
        return self.result

    def _run_gradient_optimization(self) -> ReadoutOptimizationResult:
        initial_amplitude = (
            float(self.amplitudes[0])
            if len(self.amplitudes) > 0
            else float(np.mean(list(self.initial_amplitudes.values())))
        )
        optimizer = FiniteDifferenceAmplitudeOptimizer(
            self._measure_amplitude,
            initial_amplitude=initial_amplitude,
            bounds=self.settings.amplitude_bounds,
            max_iterations=self.settings.max_iterations,
            initial_step=self.settings.initial_step,
            finite_difference_step=self.settings.finite_difference_step,
            min_step=self.settings.min_step,
            score_tolerance=self.settings.score_tolerance,
        )
        gradient_result = optimizer.optimize()
        self._set_readout_amplitude(gradient_result.best_amplitude)

        analysis = AdaptiveReadoutOptimizationAnalysis(
            samples=self.samples,
            initial_amplitudes=self.initial_amplitudes,
            initial_fidelities=gradient_result.initial_fidelities,
        )
        self.result = analysis.analyze()
        return self.result

    def plot(self) -> Figure:
        if not hasattr(self, "result"):
            raise RuntimeError("Run the optimization before plotting.")

        return ReadoutOptimizationPlotter(self.result).plot()

    def report(self) -> str:
        if not hasattr(self, "result"):
            raise RuntimeError(
                "Run the optimization before generating a report.")

        return self.result.report()


if __name__ == "__main__":
    profile = Profile.default()

    settings = OptimizationSettings(
        do_emulation=False,
        reset=ResetSettings(reset_type=ResetType.ACTIVE),
        plot=False,
    )

    handler = ReadoutOptimizationHandler(
        qubit_names=['q5'],
        readout_length=300e-9,
        amplitudes=np.linspace(0.02, 0.1, 10),
        profile=profile,
        settings=settings,
    )

    result = handler.run()

    print(result.report())

    fig = handler.plot()

    plt.show()
