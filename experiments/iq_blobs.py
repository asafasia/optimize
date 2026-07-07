from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import json
import os
import pprint
from unittest import result


from matplotlib import pyplot as plt
import numpy as np
from qigeon import TaskSubmitterAsync
from qratena.experiments.amplitude_rabi import AmplitudeRabiHandler
from qratena.experiments.base_experiment import ExperimentSettings
from laboneq.dsl.experiment.experiment import Experiment
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.ramsey import RamseyHandler
from qratena.system.components_params import reset_settings
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.system.profile_manager import ProfileManager
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ExportationMethod, ResetType, UpdateParamsMethod

from laboneq.simple import from_json

from resources import load_profile, load_task_manager


# profile = load_profile('main')

profile = Profile.default()

task_manager = load_task_manager()
profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)


# %%
# --- Compiled Experiment ---

reset = ResetSettings(
    reset_type=ResetType.PASSIVE,
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

qubits = ["q8"]
handler = IQBlobsHandler(
    qubit_names=qubits,
    settings=settings,
    # or ["g", "e", "f"] depending on the system and goals
)

# handler.run()


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

handler.experiment_result = from_json(task_result.raw_data)  # %%

# # %%

handler.analyze()

handler.plot()
# data = handler.data["q8"]

# data.keys()


# # %%
