"""Modified T1 experiment with configurable initial-state preparation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from laboneq.simple import AveragingMode, LinearSweepParameter, SweepParameter
from qratena.analysis.base_analysis_result import AnalysisResult
from qratena.analysis.t1_analysis import T1Analysis, T1AnalysisResult
from qratena.experiments.base_experiment import BaseExperiment, ExperimentSettings
from qratena.experiments.experiment_handler import ExperimentHandler
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.pulse_factory import PulseFactory
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES
from qratena.util.parameter_updater import ProfileFieldUpdateConfig
from qratena.util.qratena_logging import qratena_logger


InitialState = Literal["g", "e", "f"]
EXPERIMENT_NAME = "modified_t1"


class ModifiedT1(BaseExperiment):
    """T1-style wait sweep with configurable prepared initial state."""

    def __init__(
        self,
        qubit_names: list[str],
        configuration_params: Profile,
        initial_state: InitialState,
        qubit_names_to_measure: list[str] | None = None,
        settings: ExperimentSettings | None = None,
    ) -> None:
        super().__init__(
            experiment_name=EXPERIMENT_NAME,
            qubit_names=qubit_names,
            configuration_params=configuration_params,
            qubit_names_to_measure=qubit_names_to_measure,
            settings=settings,
        )
        self.initial_state = validate_initial_state(initial_state)
        self.pi_pulse_shape: SUPPORTED_PULSE_SHAPES = self.settings.pulse_shape

    def define_experiment_sequence(
        self,
        num_shots: int,
        decay_time_sweep_list: list[SweepParameter],
    ) -> None:
        with self.acquire_loop_rt(
            uid=f"{EXPERIMENT_NAME}_shots",
            count=num_shots,
            acquisition_type=self.acquisition_type,
            averaging_mode=self.averaging_mode,
        ):
            with self.sweep(uid="decay_time_sweep", parameter=decay_time_sweep_list):
                reset_section_uid = self.add_reset_primitive(self.reset_conf)

                with self.section(uid="initial_state_preparation", play_after=reset_section_uid):
                    for qubit_index, qubit_name in enumerate(self.qubit_names):
                        self._prepare_initial_state(qubit_name)
                        self.delay(
                            signal=f"measure_{qubit_name}",
                            time=decay_time_sweep_list[qubit_index],
                        )

                self.add_readout_primitive(
                    uid="readout_section",
                    play_after="initial_state_preparation",
                )

    def _prepare_initial_state(self, qubit_name: str) -> None:
        if self.initial_state == "g":
            return

        self._play_qubit_pulse(
            qubit_name=qubit_name,
            pulse_type=SUPPORTED_PULSE_TYPES.pi,
            uid=f"ge_pi_{qubit_name}",
        )

        if self.initial_state == "f":
            self._play_qubit_pulse(
                qubit_name=qubit_name,
                pulse_type=SUPPORTED_PULSE_TYPES.pi_ef,
                uid=f"ef_pi_{qubit_name}",
            )

    def _play_qubit_pulse(
        self,
        qubit_name: str,
        pulse_type: SUPPORTED_PULSE_TYPES,
        uid: str,
    ) -> None:
        pulse_params = self.get_qubit_params(qubit_name, pulse_type, self.pi_pulse_shape)
        pulse = PulseFactory.create(pulse_type, pulse_params)
        pulse.section_play(self, signal=f"drive_{qubit_name}", uid=uid)


class ModifiedT1Handler(ExperimentHandler):
    """Handler for the modified T1 experiment."""

    def __init__(
        self,
        qubit_names: list[str],
        initial_state: InitialState,
        decay_time_sweep_interval_length: float,
        num_sweep_points: int,
        relaxation_time_t1_factor: int = 7,
        qubit_names_to_measure: list[str] | None = None,
        settings: ExperimentSettings | None = None,
        configuration_params: Profile | None = None,
    ) -> None:
        super().__init__(
            experiment_name=f"{EXPERIMENT_NAME}_{initial_state}",
            qubit_names=qubit_names,
            qubit_names_to_measure=qubit_names_to_measure,
            settings=settings,
            configuration_params=configuration_params,
        )

        self.initial_state = validate_initial_state(initial_state)
        self.num_sweep_points = num_sweep_points
        self.num_shots = self.settings.num_shots
        self.decay_time_sweep_interval_length = decay_time_sweep_interval_length
        self.relaxation_time_t1_factor = relaxation_time_t1_factor
        self.pi_pulse_shape: SUPPORTED_PULSE_SHAPES = self.settings.pulse_shape

    def define_experiment(self) -> ModifiedT1:
        experiment = ModifiedT1(
            qubit_names=self.qubit_names,
            qubit_names_to_measure=self.qubit_names_to_measure,
            configuration_params=self.configuration_params,
            initial_state=self.initial_state,
            settings=self.settings,
        )

        self.decay_time_sweep_list = [
            LinearSweepParameter(
                start=0,
                stop=self.decay_time_sweep_interval_length,
                count=self.num_sweep_points,
            )
            for _ in self.qubit_names
        ]

        experiment.define_experiment_sequence(
            num_shots=self.num_shots,
            decay_time_sweep_list=self.decay_time_sweep_list,
        )

        return experiment

    def analyze(self) -> list[AnalysisResult]:
        self.x_time_data_points = self.decay_time_sweep_list[0].values
        results = []

        for qubit_name in self.qubit_names:
            self.data[qubit_name] = {}

            acquired_results: NDArray[np.complex64] = self.experiment_result.get_data(
                f"handle_{qubit_name}"
            )
            result: AnalysisResult = T1Analysis(self.x_time_data_points, acquired_results).analyse()

            self.data[qubit_name]["initial_state"] = self.initial_state
            self.data[qubit_name]["pulse_shape"] = self.pi_pulse_shape
            self.data[qubit_name]["y_abs_amplitudes"] = np.abs(acquired_results)
            self.data[qubit_name]["y_fitted_data"] = []
            self.data[qubit_name]["fitted_t1"] = 0

            if isinstance(result, T1AnalysisResult):
                self.data[qubit_name]["y_fitted_data"] = result.fit
                self.data[qubit_name]["fitted_t1"] = result.fitted_t1

            self.data[qubit_name]["suggested_relaxation_time"] = (
                self.data[qubit_name]["fitted_t1"] * self.relaxation_time_t1_factor
            )
            self.data[qubit_name]["x_time_data_points"] = self.x_time_data_points
            results.append(result)

        return results

    def plot(self) -> list[Figure]:
        return self.plotter.plot_experiment(
            "t1",
            self.data,
            x_time_data_points=self.x_time_data_points,
        )

    def update_system_params(self) -> None:
        if self.initial_state != "e":
            qratena_logger.info(
                "Skipping T1 profile update for initial_state=%s",
                self.initial_state,
            )
            return

        results = self.analysis_result
        updates = []
        for i, qubit_name in enumerate(self.qubit_names):
            if isinstance(results[i], T1AnalysisResult):
                updates.append(
                    ProfileFieldUpdateConfig(
                        component_name=qubit_name,
                        field_path="relaxation_delay_time",
                        description="relaxation delay time",
                        location_type="direct",
                        new_value=self.data[qubit_name]["suggested_relaxation_time"],
                    )
                )
                updates.append(
                    ProfileFieldUpdateConfig(
                        component_name=qubit_name,
                        field_path="t1",
                        description="t1 value",
                        location_type="direct",
                        new_value=results[i].fitted_t1,
                    )
                )
            else:
                qratena_logger.info(
                    "Skipping update for %s because of EmptyAnalysisResult",
                    qubit_name,
                )

        self.parameter_updater.update_profile_fields(updates)

    def export_data(self, figs: list[Figure] | None = None) -> None:
        self.data["initial_state"] = self.initial_state
        self.data["num_shots"] = self.num_shots
        self.data["num_sweep_points"] = self.num_sweep_points
        self.data["decay_time_sweep_interval_length"] = self.decay_time_sweep_interval_length

        super().export_data(figs)


def validate_initial_state(initial_state: str) -> InitialState:
    if initial_state not in {"g", "e", "f"}:
        raise ValueError("initial_state must be one of: 'g', 'e', 'f'")
    return initial_state



