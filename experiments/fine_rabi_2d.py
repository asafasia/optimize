# Scan parameter currently supports amplitude only!!!
# Active reset is supported only with DRAG pulses!!!


import numpy as np
from qratena.experiments.fine_rabi.fine_rabi_2d import FineRabi2DHandler, FineRabi2DSettings, RotationType, ScanParameter
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, ExportationMethod, ResetType, UpdateParamsMethod

from workbench.resources.load_profile import load_profile, load_task_manager
from laboneq.simple import from_json


qubits = ["q4",]

settings = FineRabi2DSettings(
    do_emulation=True,
    exportation_method=ExportationMethod.NONE,
    update_params_method=UpdateParamsMethod.NONE,
    num_shots=500,
    rotation_type=RotationType.PI,
    scan_parameter=ScanParameter.AMPLITUDE,
    pulse_shape=SUPPORTED_PULSE_SHAPES.drag,
    reset=ResetSettings(reset_type=ResetType.ACTIVE),
)

Nx = 51
Ny = 51

beta_values = np.linspace(-0.2, 0.2, Nx)
detuning_values = np.linspace(-20e6, 20e6, Nx)
amplitude_values = np.linspace(0.85, 1.15, Nx)

scan_values = {
    ScanParameter.BETA: beta_values,
    ScanParameter.DETUNING: detuning_values,
    ScanParameter.AMPLITUDE: amplitude_values,
}

profile = load_profile()

task_manager = load_task_manager()

pulse = profile.get_pi_params(qubits[0], pulse_shape=settings.pulse_shape)

# print(pulse.pi_pulse_amplitude)

handler = FineRabi2DHandler(
    qubit_names=qubits,
    repetitions=np.arange(1, Ny),
    scan_values=scan_values[settings.scan_parameter],
    settings=settings,
    profile=profile,
)

compiled_experiment = handler.get_compiled_experiment()

task_id = task_manager.submit_compiled_experiment(
    experiment_name=handler.experiment_name,
    profile_name="main",
    qubit_names=handler.qubit_names,
    compiled_experiment=compiled_experiment,
    do_emulation=False,
)

task_result = task_manager.wait_for_result(task_id)

handler.experiment_result = from_json(task_result.raw_data)


handler.analyze()
handler.plot()
