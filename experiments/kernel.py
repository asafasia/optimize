from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.kernel_traces_calculation import (
    KernelTracesCalculationHandler,
)
from qratena.system.components_params.profile import Profile
from qratena.util.enums import ExportationMethod, UpdateParamsMethod


from laboneq.simple import from_json

from resources.load_profile import load_profile, load_task_manager, push_profile

task_manager = load_task_manager()


profile_name = "main_asaf"

profile = load_profile(profile_name)


# profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)


all_qubits = sorted(list(profile.qubits.keys()), key=lambda x: int(x[1:]))

for q in all_qubits:
    print(f"Running experiment for qubit: {q}")

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


# %%
profile = Profile.default()

push_profile(profile, profile_name)
