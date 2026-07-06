# Scan parameter currently supports amplitude only!!!
# Active reset is supported only with DRAG pulses!!!


from laboneq.serializers import from_json
import numpy as np
from qratena.experiments.fine_rabi.fine_rabi_2d import FineRabi2DHandler, FineRabi2DSettings, RotationType, ScanParameter
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, ExportationMethod, ResetType, UpdateParamsMethod

from resources.load_profile import load_profile, load_task_manager

# from workbench.resources.load_profile import load_profile, load_task_manager
# from laboneq.simple import from_json


qubits = ["q8",]

settings = FineRabi2DSettings(
    do_emulation=True,
    exportation_method=ExportationMethod.EXPRESSIVE,
    update_params_method=UpdateParamsMethod.NONE,
    num_shots=500,
    rotation_type=RotationType.PI_HALF,
    scan_parameter=ScanParameter.AMPLITUDE,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    reset=ResetSettings(reset_type=ResetType.ACTIVE),
)

Nx = 151
Ny = 151

beta_values = np.linspace(-0.2, 0.2, Nx)
detuning_values = np.linspace(-20e6, 20e6, Nx)
amplitude_values = np.linspace(0.95, 1.05, Nx)

scan_values = {
    ScanParameter.BETA: beta_values,
    ScanParameter.DETUNING: detuning_values,
    ScanParameter.AMPLITUDE: amplitude_values,
}

profile = load_profile()

task_manager = load_task_manager()

pulse = profile.get_pi_params(qubits[0], pulse_shape=settings.pulse_shape)

# print(pulse.pi_pulse_amplitude)


def print_optimized_parameters(handler: FineRabi2DHandler) -> None:
    for result in handler.results:
        qubit_name = result.qubit_name
        x_intersect = result.x_intersect
        scan_parameter = handler.settings.scan_parameter

        if scan_parameter is ScanParameter.AMPLITUDE:
            base_amplitude = handler.current_pi_amplitudes[qubit_name]
            optimized_amplitude = x_intersect * base_amplitude
            if handler.settings.rotation_type is RotationType.PI_HALF:
                optimized_amplitude *= 2

            print(
                f"{qubit_name}: optimized pi_pulse_amplitude="
                f"{optimized_amplitude:.7f} (x_intersect={x_intersect:.7f})"
            )
        elif scan_parameter is ScanParameter.BETA:
            print(f"{qubit_name}: optimized beta={x_intersect:.7f}")
        else:
            print(
                f"{qubit_name}: optimized {scan_parameter.value}="
                f"{x_intersect:.7f}"
            )


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
print_optimized_parameters(handler)
figs = handler.plot()
handler.export_data(figs)
    