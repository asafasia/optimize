from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import json
import os
import pprint
from unittest import result


from matplotlib import pyplot as plt
import numpy as np
from qigeon import TaskSubmitterAsync
from qratena.experiments.amplitude_rabi import AmplitudeRabiHandler
from laboneq.simple import from_json
from laboneq.dsl.experiment.experiment import Experiment
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.ramsey import RamseyHandler
from qratena.system.components_params import reset_settings
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.system.profile_manager import ProfileManager
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ExportationMethod, ResetType, UpdateParamsMethod


from resources import load_profile, load_task_manager


profile = load_profile()

# profile = Profile.default()

profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

task_manager = load_task_manager()

qubits = profile.qubits.keys()

# Sort qubits by their number (e.g., q1, q2, q3, ...)
qubits = sorted(qubits, key=lambda x: int(x[1:]))

qubits = qubits[:-1]

qubits = ["q3"]  # Select the last three qubits (e.g., q3, q4, q5)

# %%
# --- Compiled Experiment ---

settings = ExperimentSettings(
    acquisition_type=AcquisitionType.SPECTROSCOPY,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.UPDATE,
    exportation_method=ExportationMethod.FULL,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    num_shots=500,
    reset=ResetSettings(ResetType.PASSIVE, reset_num=5),
)

for qubit in qubits:

    handler = AmplitudeRabiHandler(
        qubit_names=[qubit],
        settings=settings,
        num_sweep_points=100,
        amplitude_amplification_factor=3,
        profile=profile,
        transition="ef"
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

    # handler.update_system_params()

# %%
