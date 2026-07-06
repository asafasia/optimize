"""Hahn echo experiment with a weak readout drive during the echo sequence."""

from collections.abc import Iterable, Sequence

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.dsl.parameter import LinearSweepParameter
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.simple import pulse_library
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from qratena.analysis.base_analysis_result import AnalysisResult
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

EXPERIMENT_NAME = "hahn_echo_continuous_readout"


class HahnEchoContinuousReadout(BaseExperiment):
    """Hahn echo sequence with an extra non-acquiring readout pulse."""

    def __init__(
        self,
        qubit_names: list[str],
        configuration_params: Profile,
        echo_time_sweep_list: list[LinearSweepParameter],
        qubit_names_to_measure: list[str] | None = None,
        settings: ExperimentSettings | None = None,
        continuous_readout_amplitude_prefactor: float | None = 0.01,
        continuous_readout_duration: float | None = None,
    ) -> None:
        super().__init__(
            experiment_name=EXPERIMENT_NAME,
            qubit_names=qubit_names,
            configuration_params=configuration_params,
            qubit_names_to_measure=qubit_names_to_measure,
            settings=settings,
        )
        self.pi_pulse_shape = settings.pulse_shape
        self.echo_time_sweep_list = echo_time_sweep_list
        self.continuous_readout_amplitude_prefactor = (
            continuous_readout_amplitude_prefactor
        )
        self.continuous_readout_duration = continuous_readout_duration

    def define_experiment_sequence(self) -> None:
        if self.reset_conf.reset_type == ResetType.ACTIVE:
            raise ValueError(
                "Active reset adds readout acquisition. Use passive reset for the "
                "continuous-readout Hahn echo variant."
            )

        with self.acquire_loop_rt(
            uid=f"{EXPERIMENT_NAME}_shots",
            count=self.num_shots,
            acquisition_type=self.acquisition_type,
            averaging_mode=self.averaging_mode,
        ):
            with self.sweep(uid="echo_time_sweep", parameter=self.echo_time_sweep_list):
                reset_section_uid = self.add_reset_primitive(self.reset_conf)
                readout_waveforms = {
                    qubit_name: self._readout_waveform(qubit_name)
                    for qubit_name in self.qubit_names_to_measure
                }

                main_section_uid = "main_section"
                main_section_lengths = [
                    self._main_section_length(qubit_name, echo_time)
                    for qubit_name, echo_time in zip(
                        self.qubit_names,
                        self.echo_time_sweep_list,
                    )
                ]
                main_section_length = max(main_section_lengths)

                with self.section(
                    uid=main_section_uid,
                    play_after=reset_section_uid,
                    length=main_section_length,
                ):
                    with self.section(
                        uid="weak_readout_during_echo",
                        length=main_section_length,
                
                    ) as weak_readout_section:
                        for qubit_name in self.qubit_names_to_measure:
                            weak_readout_section.play(
                                signal=f"measure_{qubit_name}",
                                pulse=readout_waveforms[qubit_name],
                                length=main_section_length,
                                amplitude=(
                                    self.continuous_readout_amplitude_prefactor
                                ),
                            )

                    with self.section(
                        uid="echo_drive_sequence",
                        length=main_section_length,
                    ) as echo_section:
                        for qubit_index, qubit_name in enumerate(self.qubit_names):
                            echo_time = self.echo_time_sweep_list[qubit_index]
                            pi_pulse = self._pi_pulse(qubit_name)

                            pi_pulse.section_play(
                                echo_section,
                                signal=f"drive_{qubit_name}",
                                uid=f"{qubit_name}_initial_pi_over_2",
                                amplitude=0.5,
                            )
                            self.delay(signal=f"drive_{qubit_name}", time=echo_time)
                            pi_pulse.section_play(
                                echo_section,
                                signal=f"drive_{qubit_name}",
                                uid=f"{qubit_name}_echo_pi",
                            )
                            self.delay(signal=f"drive_{qubit_name}", time=echo_time)
                            pi_pulse.section_play(
                                echo_section,
                                signal=f"drive_{qubit_name}",
                                uid=f"{qubit_name}_final_pi_over_2",
                                amplitude=0.5,
                            )

                self.add_readout_primitive(
                    uid="readout_section",
                    play_after=main_section_uid,
                )

    def _pi_pulse(self, qubit_name: str):
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.pi,
            self.pi_pulse_shape,
        )
        return PulseFactory.create(SUPPORTED_PULSE_TYPES.pi, pulse_params)

    def _readout_pulse(self, qubit_name: str):
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.readout,
            SUPPORTED_PULSE_SHAPES.const,
        )
        return PulseFactory.create(SUPPORTED_PULSE_TYPES.readout, pulse_params)

    def _readout_waveform(self, qubit_name: str):
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.readout,
            SUPPORTED_PULSE_SHAPES.const,
        )
        return pulse_library.const(
            uid=f"{qubit_name}_shared_readout_waveform",
            length=pulse_params.readout_duration,
            amplitude=pulse_params.readout_amplitude,
        )

    def _pi_duration(self, qubit_name: str) -> float:
        pulse_params = self.get_qubit_params(
            qubit_name,
            SUPPORTED_PULSE_TYPES.pi,
            self.pi_pulse_shape,
        )
        return float(pulse_params.pi_pulse_duration)

    def _readout_phase(self, qubit_name: str) -> float:
        return float(self.configuration_params.qubits[qubit_name].readout_phase.value)

    def _continuous_readout_duration(self, qubit_name: str, echo_time) -> float:
        if self.continuous_readout_duration is not None:
            return self.continuous_readout_duration

        echo_time_values = getattr(echo_time, "values", None)
        if echo_time_values is not None:
            return float(max(echo_time_values))

        echo_time_stop = getattr(echo_time, "stop", None)
        if echo_time_stop is not None:
            return float(echo_time_stop)

        return float(echo_time)

    def _main_section_length(self, qubit_name: str, echo_time) -> float:
        return (
            2 * self._continuous_readout_duration(qubit_name, echo_time)
            + 3 * self._pi_duration(qubit_name)
        )


class HahnEchoContinuousReadoutHandler(ExperimentHandler):
    """Handler for Hahn echo with a non-acquiring continuous readout drive."""

    def __init__(
        self,
        qubit_names: list[str],
        echo_time_sweep_interval_length: float,
        num_sweep_points: int,
        qubit_names_to_measure: list[str] | None = None,
        settings: ExperimentSettings | None = None,
        configuration_params: Profile | None = None,
        continuous_readout_amplitude_prefactor: float | None = 0.01,
        continuous_readout_duration: float | None = None,
    ) -> None:
        super().__init__(
            experiment_name=EXPERIMENT_NAME,
            qubit_names=qubit_names,
            qubit_names_to_measure=qubit_names_to_measure,
            settings=settings,
            configuration_params=configuration_params,
        )
        self.echo_time_sweep_interval_length = echo_time_sweep_interval_length
        self.num_sweep_points = num_sweep_points
        self.continuous_readout_amplitude_prefactor = (
            continuous_readout_amplitude_prefactor
        )
        self.continuous_readout_duration = continuous_readout_duration

    def define_experiment(self) -> HahnEchoContinuousReadout:
        self.echo_time_sweep_list = [
            LinearSweepParameter(
                start=0,
                stop=self.echo_time_sweep_interval_length,
                count=self.num_sweep_points,
            )
            for _ in self.qubit_names
        ]

        experiment = HahnEchoContinuousReadout(
            qubit_names=self.qubit_names,
            qubit_names_to_measure=self.qubit_names_to_measure,
            configuration_params=self.configuration_params,
            echo_time_sweep_list=self.echo_time_sweep_list,
            settings=self.settings,
            continuous_readout_amplitude_prefactor=(
                self.continuous_readout_amplitude_prefactor
            ),
            continuous_readout_duration=self.continuous_readout_duration,
        )
        experiment.define_experiment_sequence()
        return experiment

    def analyze(self) -> Sequence[AnalysisResult]:
        self.x_time_data_points = self.echo_time_sweep_list[0].values
        self.data["note"] = (
            "Weak readout pulse is acquired into ignore_weak_readout_* handles; "
            "normal final readout acquisition is stored in handle_*."
        )
        return []

    def plot(self) -> list[Figure]:
        return []

    def update_system_params(self) -> None:
        return None

    def export_data(self, figs: Iterable[Figure] | None = None) -> None:
        self.data["num_shots"] = self.num_shots
        self.data["num_sweep_points"] = self.num_sweep_points
        self.data["continuous_readout_amplitude_prefactor"] = (
            self.continuous_readout_amplitude_prefactor
        )
        self.data["continuous_readout_duration"] = self.continuous_readout_duration
        super().export_data(figs or [])


if __name__ == "__main__":
    profile = Profile.default()
    profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

    reset = ResetSettings(
        reset_type=ResetType.PASSIVE,
    )

    settings = ExperimentSettings(
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        update_params_method=UpdateParamsMethod.NONE,
        exportation_method=ExportationMethod.EXPRESSIVE,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        num_shots=2500,
        reset=reset,
    )

    qubits = sorted(list(profile.qubits.keys()), key=lambda q: int(q[1:]))
    qubits = ["q8"]

    handler = HahnEchoContinuousReadoutHandler(
        qubit_names=qubits,
        echo_time_sweep_interval_length=8e-6,
        num_sweep_points=201,
        settings=settings,
        configuration_params=profile,
        # Weak readout drive: 1% of the configured readout amplitude.
        continuous_readout_amplitude_prefactor=0.01,
        continuous_readout_duration=None,
    )

    compiled_experiment = handler.get_compiled_experiment()
    plot_simulation(compiled_experiment, start_time=0, length=10e-6)

    plt.show()
