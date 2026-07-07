from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings, ResetSettings
from qratena.experiments.kernel_traces_calculation import (
    KernelTracesCalculationHandler,
)
from qratena.system.components_params.profile import Profile, ResetType
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, ExportationMethod, UpdateParamsMethod


from laboneq.simple import from_json
from qratena.experiments.iq_blobs import IQBlobsHandler

from resources.load_profile import load_profile, load_task_manager, push_profile

task_manager = load_task_manager()

q = "q3"

profile_name = "main"

profile = load_profile(profile_name)

all_qubits = list(profile.qubits.keys())


qubits = [q]
states = ["g", "e"]


pulse = profile.qubits[qubits[0]].pulses['readout']['const']

print(
    f"Readout pulse for {qubits[0]}: {pulse.readout_amplitude}, {pulse.readout_duration}")

settings = ExperimentSettings(
    num_shots=20_000,
    exportation_method=ExportationMethod.NONE,
    update_params_method=UpdateParamsMethod.UPDATE,
)

handler = KernelTracesCalculationHandler(
    qubit_names=qubits,
    settings=settings,
    states=states,
    profile=profile,
)

compiled_experiment = handler.get_compiled_experiment()

task_id = task_manager.submit_compiled_experiment(
    experiment_name=handler.experiment_name,
    profile_name=profile_name,
    qubit_names=handler.qubit_names,
    compiled_experiment=compiled_experiment,
    do_emulation=False,
)

task_result = task_manager.wait_for_result(task_id)

handler.experiment_result = from_json(task_result.raw_data)

# %%

handler.analyze()
handler.plot()
handler.update_system_params()


reset = ResetSettings(
    reset_type=ResetType.ACTIVE,
    reset_num=5,
)

settings = ExperimentSettings(
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.SINGLE_SHOT,
    update_params_method=UpdateParamsMethod.NONE,
    exportation_method=ExportationMethod.NONE,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    num_shots=5000,
    reset=reset,
)

qubits = sorted(list(profile.qubits.keys()), key=lambda q: int(q[1:]))

qubits = [q]
handler = IQBlobsHandler(
    qubit_names=qubits,
    settings=settings,
    # profile=profile,
    # or ["g", "e", "f"] depending on the system and goals
)


compiled_experiment = handler.get_compiled_experiment()


# plot_simulation(compiled_experiment)

task_id = task_manager.submit_compiled_experiment(
    experiment_name=handler.experiment_name,
    profile_name=profile_name,
    qubit_names=handler.qubit_names,
    compiled_experiment=compiled_experiment,
    do_emulation=False,
)

task_result = task_manager.wait_for_result(task_id)

handler.experiment_result = from_json(task_result.raw_data)  # %%

# %%

handler.analyze()

handler.plot()

# %%
