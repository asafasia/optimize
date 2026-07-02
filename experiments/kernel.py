from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.kernel_traces_calculation import (
    KernelTracesCalculationHandler,
)
from qratena.system.components_params.profile import Profile
from qratena.util.enums import ExportationMethod, UpdateParamsMethod

from workbench.resources.load_profile import load_profile, load_task_manager

from laboneq.simple import from_json

task_manager = load_task_manager()

# profile = load_profile()

profile = Profile.default()

profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

qubits = ["q8"]
states = ["g", "e"]

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

task_result = task_manager.wait(
    task_manager.run_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name="main",
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=False,
    )
)


handler.experiment_result = from_json(task_result.raw_data)

# %%

handler.analyze()

handler.plot()


# %%
