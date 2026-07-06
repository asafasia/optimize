"""Minimal Hahn echo experiment for T2 echo checks.

This is the single-refocusing-pulse version of the local CPMG experiment:
pi/2 - tau - pi - tau - pi/2 - readout.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from laboneq.serializers import from_json
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    LinearSweepParameter,
    SweepParameter,
)
from qratena.experiments.base_experiment import BaseExperiment, ExperimentSettings
from qratena.experiments.experiment_handler import ExperimentHandler
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.pulse_factory import PulseFactory
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    SUPPORTED_PULSE_TYPES,
    UpdateParamsMethod,
)

from measure_resonator_thermal_population.decay_fit import (
    SECONDS_TO_MICROSECONDS,
    fit_exponential_decay,
    fit_gaussian_decay,
)


EXPERIMENT_NAME = "hahn_echo"
PROFILE_NAME = "main"
QUBITS = ["q8"]
ECHO_DELAY_SWEEP_START = 40e-8
ECHO_DELAY_SWEEP_STOP = 50e-6
NUM_SWEEP_POINTS = 151


class HahnEcho(BaseExperiment):
    """Hahn echo sequence with swept total echo delay."""

    def __init__(
        self,
        qubit_names: list[str],
        configuration_params: Profile,
        echo_delay_sweep_list: list[SweepParameter],
        final_pi_half_phase: float = np.pi / 2,
        qubit_names_to_measure: list[str] | None = None,
        settings: ExperimentSettings | None = None,
    ) -> None:
        self.settings = settings or ExperimentSettings()
        super().__init__(
            experiment_name=EXPERIMENT_NAME,
            qubit_names=qubit_names,
            qubit_names_to_measure=qubit_names_to_measure,
            configuration_params=configuration_params,
            settings=self.settings,
        )
        self.echo_delay_sweep_list = echo_delay_sweep_list
        self.final_pi_half_phase = final_pi_half_phase
        self.pi_pulse_shape: SUPPORTED_PULSE_SHAPES = self.settings.pulse_shape

    def define_experiment_sequence(self) -> None:
        with self.acquire_loop_rt(
            uid=f"{EXPERIMENT_NAME}_shots",
            count=self.num_shots,
            acquisition_type=self.acquisition_type,
            averaging_mode=self.averaging_mode,
        ):
            with self.sweep(
                uid="echo_delay_sweep",
                parameter=self.echo_delay_sweep_list,
            ):
                reset_section_uid = self.add_reset_primitive(self.reset_conf)

                with self.section(uid="hahn_echo_sequence", play_after=reset_section_uid):
                    for qubit_index, qubit_name in enumerate(self.qubit_names):
                        echo_delay = self.echo_delay_sweep_list[qubit_index]
                        pi_pulse_duration = self._get_pi_pulse_duration(qubit_name)
                        free_evolution_time = (echo_delay - pi_pulse_duration) * 0.5

                        self._play_pi_half(
                            qubit_name=qubit_name,
                            uid=f"first_pi_half_{qubit_name}",
                            phase=np.pi / 2,
                        )
                        self.delay(signal=f"drive_{qubit_name}", time=free_evolution_time)
                        # self._play_pi(qubit_name=qubit_name, uid=f"pi_{qubit_name}")
                        self.delay(signal=f"drive_{qubit_name}", time=free_evolution_time)
                        self._play_pi_half(
                            qubit_name=qubit_name,
                            uid=f"final_pi_half_{qubit_name}",
                            phase=self.final_pi_half_phase,
                        )

                self.add_readout_primitive(
                    uid="readout_section",
                    play_after="hahn_echo_sequence",
                )

    def _play_pi(self, qubit_name: str, uid: str) -> None:
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.pi,
            self.pi_pulse_shape,
        )
        pulse = PulseFactory.create(SUPPORTED_PULSE_TYPES.pi, pulse_params)
        pulse.section_play(self, signal=f"drive_{qubit_name}", uid=uid)

    def _get_pi_pulse_duration(self, qubit_name: str) -> float:
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.pi,
            self.pi_pulse_shape,
        )
        return float(pulse_params.pi_pulse_duration)

    def _play_pi_half(self, qubit_name: str, uid: str, phase: float) -> None:
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.pi,
            self.pi_pulse_shape,
        )
        pulse = PulseFactory.create(SUPPORTED_PULSE_TYPES.pi, pulse_params)
        pulse.section_play(
            self,
            signal=f"drive_{qubit_name}",
            uid=uid,
            amplitude=0.5,
            phase=phase,
        )


class HahnEchoHandler(ExperimentHandler):
    """Minimal Hahn echo handler for compile/run smoke tests."""

    def __init__(
        self,
        qubit_names: list[str],
        echo_delay_sweep_start: float,
        echo_delay_sweep_stop: float,
        num_sweep_points: int,
        final_pi_half_phase: float = np.pi / 2,
        qubit_names_to_measure: list[str] | None = None,
        settings: ExperimentSettings | None = None,
        configuration_params: Profile | None = None,
    ) -> None:
        super().__init__(
            experiment_name=EXPERIMENT_NAME,
            qubit_names=qubit_names,
            qubit_names_to_measure=qubit_names_to_measure,
            settings=settings,
            configuration_params=configuration_params,
        )
        self.echo_delay_sweep_start = echo_delay_sweep_start
        self.echo_delay_sweep_stop = echo_delay_sweep_stop
        self.num_sweep_points = num_sweep_points
        self.final_pi_half_phase = final_pi_half_phase
        self.pi_pulse_shape: SUPPORTED_PULSE_SHAPES = self.settings.pulse_shape
        self._validate_echo_delay()

    def define_experiment(self) -> HahnEcho:
        self.echo_delay_sweep_list = [
            LinearSweepParameter(
                start=self.echo_delay_sweep_start,
                stop=self.echo_delay_sweep_stop,
                count=self.num_sweep_points,
            )
            for _ in self.qubit_names
        ]

        experiment = HahnEcho(
            qubit_names=self.qubit_names,
            qubit_names_to_measure=self.qubit_names_to_measure,
            configuration_params=self.configuration_params,
            echo_delay_sweep_list=self.echo_delay_sweep_list,
            final_pi_half_phase=self.final_pi_half_phase,
            settings=self.settings,
        )
        experiment.define_experiment_sequence()
        return experiment

    def analyze(self) -> list[dict[str, Any]]:
        self.x_time_data_points = self.echo_delay_sweep_list[0].values
        results: list[dict[str, Any]] = []

        for qubit_name in self.qubit_names:
            acquired_results: NDArray[np.complex64] = self.experiment_result.get_data(
                f"handle_{qubit_name}"
            )
            y_real = np.real(acquired_results)
            y_abs = np.abs(acquired_results)
            signal = y_abs
            fit_result = fit_exponential_decay(self.x_time_data_points, y_abs)
            gaussian_fit_result = fit_gaussian_decay(self.x_time_data_points, y_abs)

            self.data[qubit_name] = {
                "echo_delay_points": self.x_time_data_points,
                "echo_delay_points_us": (
                    self.x_time_data_points * SECONDS_TO_MICROSECONDS
                ),
                "acquired_results": acquired_results,
                "y_real": y_real,
                "y_abs": y_abs,
                "signal": signal,
                "y_abs_fitted": fit_result["fitted"],
                "fitted_t2_echo": fit_result["tau"],
                "fit_amplitude": fit_result["amplitude"],
                "fit_offset": fit_result["offset"],
                "fit_r2": fit_result["r2"],
                "fit_score": fit_result["score"],
                "fit_points": fit_result["fit_points"],
                "y_abs_gaussian_fitted": gaussian_fit_result["fitted"],
                "fitted_gaussian_t2_echo": gaussian_fit_result["tau"],
                "gaussian_fit_amplitude": gaussian_fit_result["amplitude"],
                "gaussian_fit_offset": gaussian_fit_result["offset"],
                "gaussian_fit_r2": gaussian_fit_result["r2"],
                "gaussian_fit_score": gaussian_fit_result["score"],
                "gaussian_fit_points": gaussian_fit_result["fit_points"],
                "contrast_estimate": float(np.max(y_real) - np.min(y_real)),
            }
            results.append(self.data[qubit_name])

        return results

    def plot(self) -> list[Figure]:
        figs = []
        for qubit_name in self.qubit_names:
            qubit_data = self.data[qubit_name]
            echo_delay_points_us = qubit_data["echo_delay_points_us"]
            fig, ax = plt.subplots()
            ax.plot(
                echo_delay_points_us,
                qubit_data["signal"],
                "o-",
                label="signal",
            )
            has_exponential_fit = len(qubit_data["y_abs_fitted"]) > 0
            has_gaussian_fit = len(qubit_data["y_abs_gaussian_fitted"]) > 0
            if has_exponential_fit:
                fitted_t2_us = qubit_data["fitted_t2_echo"] * SECONDS_TO_MICROSECONDS
                ax.plot(
                    echo_delay_points_us,
                    qubit_data["y_abs_fitted"],
                    "r-",
                    label=(
                        f"exp fit, T2 echo={fitted_t2_us:.2f} us, "
                        f"R^2={qubit_data['fit_r2']:.3f}"
                    ),
                )
                ax.set_title(
                    f"Hahn echo {qubit_name}, T2 echo={fitted_t2_us:.2f} us"
                )
            if has_gaussian_fit:
                gaussian_t2_us = (
                    qubit_data["fitted_gaussian_t2_echo"] * SECONDS_TO_MICROSECONDS
                )
                ax.plot(
                    echo_delay_points_us,
                    qubit_data["y_abs_gaussian_fitted"],
                    "g--",
                    label=(
                        f"Gaussian fit, T2 echo={gaussian_t2_us:.2f} us, "
                        f"R^2={qubit_data['gaussian_fit_r2']:.3f}"
                    ),
                )
            if not has_exponential_fit and not has_gaussian_fit:
                ax.set_title(f"Hahn echo {qubit_name}, fit failed")
            elif not has_exponential_fit:
                ax.set_title(f"Hahn echo {qubit_name}, Gaussian fit only")
            ax.set_xlabel("Echo delay [us]")
            ax.set_ylabel("Acquired signal")
            ax.legend()
            figs.append(fig)
        return figs

    def update_system_params(self) -> None:
        return None

    def export_data(self, figs: list[Figure] | None = None) -> None:
        return None

    def _validate_echo_delay(self) -> None:
        min_pi_duration = max(
            self.configuration_params.qubits[qubit_name]
            .pulses[SUPPORTED_PULSE_TYPES.pi][self.pi_pulse_shape]
            .pi_pulse_duration
            for qubit_name in self.qubit_names
        )
        if self.echo_delay_sweep_start < min_pi_duration:
            raise ValueError(
                "echo_delay_sweep_start must be at least the pi pulse duration "
                f"({min_pi_duration:.3e} s for the selected qubits/pulse shape)"
            )


def main() -> None:
    from resources.load_profile import load_profile, load_task_manager

    profile = load_profile(PROFILE_NAME)
    task_manager = load_task_manager()

    settings = ExperimentSettings(
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        update_params_method=UpdateParamsMethod.NONE,
        exportation_method=ExportationMethod.FULL,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        num_shots=2500,
        reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=5),
    )

    handler = HahnEchoHandler(
        qubit_names=QUBITS,
        echo_delay_sweep_start=ECHO_DELAY_SWEEP_START,
        echo_delay_sweep_stop=ECHO_DELAY_SWEEP_STOP,
        num_sweep_points=NUM_SWEEP_POINTS,
        settings=settings,
        configuration_params=profile,
    )
    compiled_experiment = handler.get_compiled_experiment()

    task_id = task_manager.submit_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name=PROFILE_NAME,
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=False,
    )
    task_result = task_manager.wait_for_result(task_id)

    handler.experiment_result = from_json(task_result.raw_data)
    handler.analysis_result = handler.analyze()
    handler.figs = handler.plot()
    handler.export_data(figs=handler.figs)
    plt.show()


if __name__ == "__main__":
    main()
