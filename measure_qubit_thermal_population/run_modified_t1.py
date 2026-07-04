"""Run modified T1 measurements for thermal-population checks."""

from __future__ import annotations

from laboneq.serializers import from_json
from laboneq.simple import AcquisitionType, AveragingMode
import matplotlib.pyplot as plt
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    UpdateParamsMethod,
)

from measure_qubit_thermal_population.modified_t1 import ModifiedT1Handler
from resources.load_profile import load_profile, load_task_manager


PROFILE_NAME = "main"
QUBITS = ["q8"]
INITIAL_STATES = ["e", "g"]
DECAY_TIME_SWEEP_INTERVAL_LENGTH = 200e-6
NUM_SWEEP_POINTS = 101


profile = load_profile(PROFILE_NAME)
profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)
task_manager = load_task_manager()

settings = ExperimentSettings(
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.NONE,
    exportation_method=ExportationMethod.FULL,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    num_shots=2500,
    reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=5),
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

    task_id = task_manager.submit_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name=PROFILE_NAME,
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=False,
    )
    task_result = task_manager.wait_for_result(task_id)

    handler.experiment_result = from_json(task_result.raw_data)
    handler.analysis_result = handler.analyze()
    handler.figs = handler.plot()
    handler.export_data(figs=handler.figs)
    handlers.append(handler)

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

for handler in handlers:
    for qubit_name in handler.qubit_names:
        qubit_data = handler.data[qubit_name]
        ax.plot(
            qubit_data["x_time_data_points"],
            qubit_data["y_abs_amplitudes"],
            marker="o",
            label=f"{qubit_name}, initial {handler.initial_state}",
        )

ax.set_xlabel("Decay time (s)")
ax.set_ylabel("Abs amplitude")
ax.set_title("Modified T1: initial g/e comparison")
ax.legend()
ax.grid(True)
fig.tight_layout()
ax.set_ylim(0, 1)
plt.show()

# %%
