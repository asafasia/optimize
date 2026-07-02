from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.resonator_spectroscopy import ResonatorSpectroscopyHandler
from qratena.system.components_params.profile import Profile
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ExportationMethod, UpdateParamsMethod
from qratena.util.sweeps_utils import MidIntervalArray

from workbench.resources.load_profile import load_profile, load_task_manager


task_manager = load_task_manager()

# profile = load_profile()


profile = Profile.default()


profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

qubits = ['q4']
# pulse.readout_amplitude = 0.01


settings = ExperimentSettings(
    num_shots=400,
    exportation_method=ExportationMethod.NONE,
    update_params_method=UpdateParamsMethod.UPDATE,
)

handler = ResonatorSpectroscopyHandler(
    qubit_names=qubits,
    x_resonator_frequency_arrays=[MidIntervalArray(
        mid_point=None, interval=50e6, num_points=100)],
    long_drive_pulse=False,

    settings=settings,
    states=["g", "e", 'f'],
    profile=profile,
)


compiled_experiment = handler.get_compiled_experiment()


task_result = task_manager.wait(
    task_manager.run_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name="main",
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=False
    ))


handler.load_result(task_result)
# %%

handler.analyze()

handler.plot()

handler.update_system_params()


# %%


# push_profile(profile)
