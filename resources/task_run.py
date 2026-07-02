

from laboneq.serializers import from_json, to_json, load
import io

from qratena.experiments.base_experiment import ExperimentSettings

from workbench.resources import load_task_manager

import os
import sys
from pathlib import Path

import numpy as np

# Allow running without installing qigeon/qratena — add source directories to sys.path
# These must come BEFORE any qratena/qigeon imports so source versions take priority
# over any stale installed copies in site-packages.
# _TEST_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(_TEST_ROOT / "qigeon"))
# sys.path.insert(0, str(_TEST_ROOT / "qratena"))

from qigeon import TaskSubmitterAsync
from laboneq.simple import AcquisitionType, AveragingMode
# from qompute.rb.notebooks.simultaneous_rb_exp_new import do_emulation
# from qompute.rb.notebooks.simultaneous_rb_exp_new import do_emulation
from qratena.experiments import AmplitudeRabiHandler
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, ExportationMethod, UpdateParamsMethod
from qratena.util.sweeps_utils import MidIntervalArray, StartStopArray, InclusiveIntegerArray
from qratena.experiments.transmon_spectroscopy.qubit_spectroscopy import QubitSpectroscopyHandlerConfig
from qratena.experiments.transmon_spectroscopy.qubit_flux_spectroscopy import QubitFluxSpectroscopyHandlerConfig
from qratena.experiments.transmon_spectroscopy.coupler_spectroscopy import CouplerSpectroscopyHandlerConfig
from qratena.experiments.cz_calibration.cphase_coupling import CPhaseCouplingHandlerConfig
from qratena.experiments.cz_calibration.cphase_oscillations import CPhaseOscillationsHandlerConfig
from qratena.experiments.cz_calibration.detuning_minimization import DetuningMinimizationHandlerConfig
from qratena.experiments.cz_calibration.phase_shift_correction import PhaseShiftCorrectionHandlerConfig
from qratena.experiments.cz_calibration.cz_validation import CZValidationHandlerConfig
try:
    from laboneq.core.types import CompiledExperiment
except ImportError:
    CompiledExperiment = None  # type: ignore[assignment,misc]


PROFILE_NAME = 'main'


def run_experiment(
    ts: TaskSubmitterAsync,
    experiment_name: str,
    experiment_setting: ExperimentSettings,
    **kwargs,
) -> None:
    qubit_names = kwargs.pop("qubit_names")
    print(f"Submitting {experiment_name!r} on {qubit_names} ...")
    try:
        result = ts.wait(ts.run_experiment(
            profile_name=PROFILE_NAME,
            experiment_name=experiment_name,
            qubit_names=qubit_names,
            do_emulation=experiment_setting.do_emulation,
            num_shots=experiment_setting.num_shots,
            acquisition_type=experiment_setting.acquisition_type.name,
            averaging_mode=experiment_setting.averaging_mode.name,
            pulse_shape=experiment_setting.pulse_shape.value,
            states=experiment_setting.states,
            compiler_settings=experiment_setting.compiler_settings,
            update_params_method=experiment_setting.update_params_method.value,
            reset_type=experiment_setting.reset.reset_type.value,
            reset_num=experiment_setting.reset.reset_num,
            experiment_kwargs=kwargs,
        ))

        print("  Task completed")
        print(f"  experiment : {result.experiment_name}")
        print(f"  qubits     : {result.qubit_names}")
        print(f"  data keys  : {list(result.data.keys())}")
        print(f"  analysis   : {result.analysis_result}")
        print(f"  figures    : {len(result.figures)} PNG(s)")

        return result

    except Exception as e:
        print(f"  Task FAILED: {e}")
        return None


ts = load_task_manager()


settings = ExperimentSettings(
    acquisition_type=AcquisitionType.DISCRIMINATION,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.NONE,
    exportation_method=ExportationMethod.FULL,
    pulse_shape=SUPPORTED_PULSE_SHAPES.cos,
    num_shots=30,
    do_emulation=False,
)


result = run_experiment(ts, "amplitude_rabi", experiment_setting=settings, qubit_names=[
                        "q3"], amplitude_amplification_factor=3, num_sweep_points=20)


# %%

# %%
# from laboneq.serializers import from_json, to_json, load
# from IPython.display import Image, display

# print(result.__dict__.keys())


result = result.__dict__

# print(result.data.keys())


# laboneq_result = from_json(result.data["experiment_results"])


# analysis_results = result.analysis_result

# figures = result.figures

# for fig in figures:

#     display(Image(data=fig))


# %%
