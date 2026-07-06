from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.simple import from_json
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.transmon_spectroscopy import (
    QubitSpectroscopyHandler,
    QubitSpectroscopyHandlerConfig,
)
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    UpdateParamsMethod,
)
from qratena.util.sweeps_utils import MidIntervalArray

from resources import load_profile, load_task_manager


profile = load_profile("main")

profile = Profile.default()

task_manager = load_task_manager()
profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)


# %%
# --- Compiled Experiment ---

reset = ResetSettings(
    reset_type=ResetType.ACTIVE,
    reset_num=5,
)

settings = ExperimentSettings(
    acquisition_type=AcquisitionType.DISCRIMINATION,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.NONE,
    exportation_method=ExportationMethod.NONE,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    num_shots=1000,
    reset=reset,
)

qubits = sorted(list(profile.qubits.keys()), key=lambda q: int(q[1:]))

qubits = ["q7"]
handler = QubitSpectroscopyHandler(
    handler_configs=[
        QubitSpectroscopyHandlerConfig(
            qubit_name=qubit,
            qubit_frequency=MidIntervalArray(
                mid_point=None,
                interval=20e6,
                num_points=101,
            ),
            drive_amplitude=0.001,
        )
        for qubit in qubits
    ],
    excited_state=1,
    fitting_method="lorentzian",
    detuning_plot_style=False,
    settings=settings,
)


compiled_experiment = handler.get_compiled_experiment()


# plot_simulation(compiled_experiment)

task_id = task_manager.submit_compiled_experiment(
    experiment_name=handler.experiment_name,
    profile_name="main",
    qubit_names=handler.qubit_names,
    compiled_experiment=compiled_experiment,
    do_emulation=False,
)

task_result = task_manager.wait_for_result(task_id)

handler.experiment_result = from_json(task_result.raw_data)

# %%

handler.analyze()

handler.plot()

# data = handler.data["q7"]

# data.keys()


# %%
