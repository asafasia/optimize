import numpy as np
from laboneq.simple import from_json
from qratena.experiments.fine_rabi import fine_rabi_1d as fine_rabi_1d_module
from qratena.experiments.fine_rabi.fine_rabi_1d import (
    FineRabi1DHandler,
    RotationType,
    SettingsFineRabi,
)
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, ExportationMethod, ResetType, UpdateParamsMethod

from resources.load_profile import load_profile, load_task_manager

# from workbench.resources.load_profile import load_profile, load_task_manager


qubits = ["q8"]

settings = SettingsFineRabi(
    do_emulation=True,
    exportation_method=ExportationMethod.NONE,
    update_params_method=UpdateParamsMethod.NONE,
    num_shots=500,
    rotation_type=RotationType.PI_HALF,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    reset=ResetSettings(reset_type=ResetType.ACTIVE),
)

repetitions = np.arange(0, 100, 1)

profile = load_profile()

# FineRabi1D currently reads the profile from its module during construction.
fine_rabi_1d_module.profile = profile

task_manager = load_task_manager()

handler = FineRabi1DHandler(
    qubit_names=qubits,
    repetitions=repetitions,
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
