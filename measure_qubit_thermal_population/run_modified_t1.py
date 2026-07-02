"""Run modified T1 measurements for thermal-population checks."""

from __future__ import annotations

from laboneq.simple import AcquisitionType, AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    UpdateParamsMethod,
)

from measure_qubit_thermal_population.modified_t1 import ModifiedT1Handler
from resources.load_profile import load_profile


PROFILE_NAME = "main"
QUBITS = ["q3"]
INITIAL_STATES = ["e", "g"]
DECAY_TIME_SWEEP_INTERVAL_LENGTH = 200e-6
NUM_SWEEP_POINTS = 101


profile = load_profile(PROFILE_NAME)
profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)
# task_manager = load_task_manager()

settings = ExperimentSettings(
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.NONE,
    exportation_method=ExportationMethod.FULL,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    num_shots=500,
    reset=ResetSettings(ResetType.PASSIVE, reset_num=5),
)

handlers = []

for initial_state in INITIAL_STATES:
    handler = ModifiedT1Handler(
        qubit_names=QUBITS,
        initial_state=initial_state,
        decay_time_sweep_interval_length=DECAY_TIME_SWEEP_INTERVAL_LENGTH,
        num_sweep_points=NUM_SWEEP_POINTS,
        settings=settings,
        configuration_params=profile,
    )

    compiled_experiment = handler.get_compiled_experiment()

    # Hardware execution is intentionally disabled while validating the sequence.
    # task_result = task_manager.wait(
    #     task_manager.run_compiled_experiment(
    #         experiment_name=handler.experiment_name,
    #         profile_name=PROFILE_NAME,
    #         qubit_names=handler.qubit_names,
    #         compiled_experiment=compiled_experiment,
    #         do_emulation=False,
    #     )
    # )

    # Analysis/export are disabled until real result loading is re-enabled.
    # handler.experiment_result = from_json(task_result.raw_data)
    # handler.analysis_result = handler.analyze()
    # handler.figs = handler.plot()
    # handler.export_data(figs=handler.figs)
    handlers.append(handler)
