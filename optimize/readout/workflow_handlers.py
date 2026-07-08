from __future__ import annotations

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.kernel_traces_calculation import KernelTracesCalculationHandler
from qratena.experiments.resonator_spectroscopy import ResonatorSpectroscopyHandler
from qratena.system.qratena_platform import create_platform
from qratena.util.enums import (
    SUPPORTED_PULSE_SHAPES,
    ExportationMethod,
    UpdateParamsMethod,
)
from qratena.util.sweeps_utils import MidIntervalArray


class ReadoutWorkflowHandlerFactoryMixin:
    def _update_profile_from_resonator(self, handler) -> None:
        for qubit_name in handler.qubit_names:
            optimal_frequency = handler.data[qubit_name]["optimal_resonance_freq"]
            self.profile.qubits[qubit_name].readout_resonator_frequency.value = (
                optimal_frequency
            )

    def _build_resonator_handler(self, qubit_name: str):
        settings = ExperimentSettings(
            log_level=0,
            num_shots=300,
            exportation_method=ExportationMethod.NONE,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            update_params_method=UpdateParamsMethod.NONE,
            do_emulation=True,
        )
        return ResonatorSpectroscopyHandler(
            x_resonator_frequency_arrays=[
                MidIntervalArray(mid_point=None, interval=150e6, num_points=120)
            ],
            long_drive_pulse=False,
            qubit_names=[qubit_name],
            settings=settings,
            profile=self.profile,
            session=self.session,
            states=self.settings.states,
        )

    def _build_kernel_handler(self):
        settings = ExperimentSettings(
            log_level=0,
            num_shots=20000,
            exportation_method=ExportationMethod.NONE,
            update_params_method=UpdateParamsMethod.UPDATE,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
            do_emulation=True,
        )
        return KernelTracesCalculationHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            profile=self.profile,
            session=self.session,
            states=self.settings.states,
        )

    def _build_iq_blobs_handler(self):
        settings = ExperimentSettings(
            num_shots=10000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            exportation_method=ExportationMethod.NONE,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            reset=self.settings.reset,
            do_emulation=True,
        )

        handler = IQBlobsHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            states=self.settings.states,
        )
        handler.configuration_params = self.profile
        handler.platform = create_platform(self.profile)
        handler.device_setup = handler.platform.setup
        if self.session is not None:
            handler.session = self.session

        return handler
