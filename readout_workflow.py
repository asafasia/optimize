from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from qigeon import TaskSubmitterAsync
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.experiment_handler import ExperimentHandler
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.kernel_traces_calculation import KernelTracesCalculationHandler
from qratena.experiments.resonator_spectroscopy import ResonatorSpectroscopyHandler
from qratena.system.components_params.profile import Profile
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ExportationMethod, UpdateParamsMethod
from qratena.util.sweeps_utils import MidIntervalArray
from qratena.util.sweeps_utils import MidIntervalArray

from resources.load_profile import load_profile, load_task_manager
from resources import *


from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode


@dataclass(slots=True)
class ReadoutFidelityWorkflowSettings:
    profile_name: str = "main"
    do_emulation: bool = False
    run_resonator: bool = True
    run_kernels: bool = True
    run_iq_blobs: bool = True
    plot_handlers: bool = True
    display_handler_plots: bool = False
    suppress_handler_output: bool = False


class ReadoutFidelityWorkflow:
    """Workflow:

    1. Resonator spectroscopy
    2. Kernel traces calculation
    3. IQ blobs
    """

    def __init__(
        self,
        qubit_names: list[str],
        profile: Profile,
        task_manager: TaskSubmitterAsync,
        settings: ReadoutFidelityWorkflowSettings | None = None,
    ) -> None:
        self.qubit_names = qubit_names
        self.profile = profile
        self.task_manager = task_manager
        self.settings = settings or ReadoutFidelityWorkflowSettings()

        self.resonator_handler = None
        self.kernel_handler = None
        self.iq_blobs_handler = None

        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        if self.settings.run_resonator:
            self.results["resonator"] = self.run_resonator_node()

        if self.settings.run_kernels:
            self.results["kernels"] = self.run_kernel_node()

        if self.settings.run_iq_blobs:
            self.results["iq_blobs"] = self.run_iq_blobs_node()

        return self.results

    def run_resonator_node(self) -> Any:
        handler = self._build_resonator_handler()
        self.resonator_handler = handler

        with self._optional_output_suppression():
            result = self._submit_handler(handler)
        self._load_handler_result(handler, result)

        self._update_profile_from_resonator(handler)

        return handler.data

    def run_kernel_node(self) -> Any:
        handler = self._build_kernel_handler()
        self.kernel_handler = handler

        with self._optional_output_suppression():
            result = self._submit_kernel_handler(handler)

        self._load_handler_result(handler, result)
        # self._analyze_handler_result(handler)

        return handler.data

    def run_iq_blobs_node(self) -> Any:
        handler = self._build_iq_blobs_handler()
        self.iq_blobs_handler = handler

        with self._optional_output_suppression():
            result = self._submit_handler(handler)
        self._load_handler_result(handler, result)

        return handler.data

    def _submit_handler(self, handler) -> Any:
        compiled_experiment = handler.get_compiled_experiment()

        return self._submit_compiled_experiment(handler, compiled_experiment)

    def _submit_compiled_experiment(self, handler, compiled_experiment) -> Any:
        return self.task_manager.wait(
            self.task_manager.run_compiled_experiment(
                handler.experiment_name,
                self.settings.profile_name,
                handler.qubit_names,
                compiled_experiment,
                do_emulation=self.settings.do_emulation,
            )
        )

    def _submit_kernel_handler(self, handler: KernelTracesCalculationHandler) -> Any:
        handler.define_experiment()

        compiled_experiment_0 = self._compile_kernel_experiment(
            handler,
            handler.experiment_0,
        )
        compiled_experiment_1 = self._compile_kernel_experiment(
            handler,
            handler.experiment_1,
        )

        handler.experiment_0_result = self._submit_compiled_experiment(
            handler,
            compiled_experiment_0,
        )
        handler.experiment_1_result = self._submit_compiled_experiment(
            handler,
            compiled_experiment_1,
        )

        return handler.experiment_0_result, handler.experiment_1_result

    def _compile_kernel_experiment(
        self,
        handler: KernelTracesCalculationHandler,
        experiment: Any,
    ) -> Any:
        original_experiment = getattr(handler, "experiment", None)
        had_experiment = hasattr(handler, "experiment")
        original_compiled_experiment = getattr(
            handler, "compiled_experiment", None)
        had_compiled_experiment = hasattr(handler, "compiled_experiment")

        handler.experiment = experiment
        if had_compiled_experiment:
            delattr(handler, "compiled_experiment")

        try:
            compiled_experiment = handler.execution_core.get_compiled_experiment(
                experiment
            )
            handler.compiled_experiment = compiled_experiment
            return compiled_experiment
        finally:
            if had_experiment:
                handler.experiment = original_experiment
            else:
                delattr(handler, "experiment")

            if had_compiled_experiment:
                handler.compiled_experiment = original_compiled_experiment

    def _load_handler_result(self, handler: ExperimentHandler, result: Any) -> None:
        """Convert task-manager result into the handler's normal data format.

        Implement this once in BaseExperimentHandler if possible.
        """
        with self._optional_output_suppression():
            handler.load_result(result)
        self._analyze_handler_result(handler)

    def _analyze_handler_result(self, handler: ExperimentHandler) -> None:
        with self._optional_output_suppression():
            handler.analyze()

        if not self.settings.plot_handlers:
            return

        with self._optional_output_suppression():
            figures = self._plot_handler(handler)

        if figures:
            handler.workflow_figures = figures
            if not self.settings.display_handler_plots:
                for figure in figures:
                    plt.close(figure)

    def _plot_handler(self, handler: ExperimentHandler) -> list[Figure]:
        existing_figures = set(plt.get_fignums())
        plot_result = handler.plot()

        if isinstance(plot_result, Figure):
            return [plot_result]
        if isinstance(plot_result, (list, tuple)):
            return [figure for figure in plot_result if isinstance(figure, Figure)]

        new_figures = set(plt.get_fignums()) - existing_figures
        return [plt.figure(number) for number in sorted(new_figures)]

    @contextlib.contextmanager
    def _optional_output_suppression(self):
        if not self.settings.suppress_handler_output:
            yield
            return

        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            yield

    def _update_profile_from_resonator(self, handler) -> None:
        for qubit_name in self.qubit_names:

            optimal_frequency = handler.data[qubit_name][
                "optimal_resonance_freq"
            ]
            self.profile.qubits[
                qubit_name
            ].readout_resonator_frequency.value = optimal_frequency

    def _build_resonator_handler(self):
        settings = ExperimentSettings(
            log_level=0,
            num_shots=1000,
            exportation_method=ExportationMethod.NONE,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            update_params_method=UpdateParamsMethod.NONE,
            configure_logging=self.settings.suppress_handler_output,
        )
        handler = ResonatorSpectroscopyHandler(
            x_resonator_frequency_arrays=[MidIntervalArray(
                mid_point=None, interval=200e6, num_points=120)],
            long_drive_pulse=False,
            qubit_names=[self.qubit_names[0]],
            settings=settings,
            profile=self.profile,
        )

        return handler

    def _build_kernel_handler(self):
        settings = ExperimentSettings(
            log_level=0,
            num_shots=5000,
            exportation_method=ExportationMethod.NONE,
            update_params_method=UpdateParamsMethod.UPDATE,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
            configure_logging=self.settings.suppress_handler_output,
        )
        handler = KernelTracesCalculationHandler(
            qubit_names=[self.qubit_names[0]],
            settings=settings,
            profile=self.profile,
        )

        return handler

    def _build_iq_blobs_handler(self):
        settings = ExperimentSettings(
            num_shots=10000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            exportation_method=ExportationMethod.NONE,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            configure_logging=self.settings.suppress_handler_output,
        )

        handler = IQBlobsHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            profile=self.profile,
        )

        return handler


if __name__ == "__main__":

    qubit_names = ["q9"]

    profile = load_profile()

    task_manager = load_task_manager()

    readout_pulse = profile.qubits[qubit_names[0]].pulses[
        SUPPORTED_PULSE_TYPES.readout][SUPPORTED_PULSE_SHAPES.const]


    # %%
    settings = ReadoutFidelityWorkflowSettings(
        profile_name="main",
        do_emulation=False,
        run_resonator=False,
        run_kernels=True,
        run_iq_blobs=True,
        display_handler_plots=True,
        suppress_handler_output=False,
    )

    workflow = ReadoutFidelityWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
    )

    workflow.run()


# %%
