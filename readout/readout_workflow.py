from __future__ import annotations

import contextlib
import io
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from qigeon import TaskSubmitterAsync
from qratena import data
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.experiment_handler import ExperimentHandler
from qratena.experiments.iq_blobs import IQBlobsHandler, IQBlobsSettings
from qratena.experiments.kernel_traces_calculation import KernelTracesCalculationHandler
from qratena.experiments.resonator_spectroscopy import ResonatorSpectroscopyHandler
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.system.qratena_platform import create_platform
from qratena.system.qratena_platform import create_platform
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ExportationMethod, ResetType, UpdateParamsMethod
from qratena.util.sweeps_utils import MidIntervalArray
from qratena.util.sweeps_utils import MidIntervalArray

from resources.load_profile import load_profile, load_task_manager
from resources import *


from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.dsl.session import Session

import httpx


@dataclass(slots=True)
class ReadoutFidelityWorkflowSettings:
    profile_name: str = "main"
    do_emulation: bool = False
    run_resonator: bool = True
    run_kernels: bool = True
    run_iq_blobs: bool = True
    do_plotting: bool = False
    show_handler_output: bool = True
    report_timing: bool = True
    task_status_poll_interval: float = 10.0
    reset: ResetSettings = field(default_factory=ResetSettings)


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
        session: Session | None = None,
    ) -> None:

        self.session = session

        self.qubit_names = qubit_names
        self.profile = profile
        self.task_manager = task_manager
        self.settings = settings or ReadoutFidelityWorkflowSettings()

        self.resonator_handler = None
        self.kernel_handler = None
        self.iq_blobs_handler = None

        self.results: dict[str, Any] = {}
        self.timings: dict[str, float] = {}

    def run(self) -> dict[str, Any]:
        workflow_start = perf_counter()
        if self.settings.run_resonator:
            self.results["resonator"] = self._timed_step(
                "resonator",
                self.run_resonator_node,
            )

        if self.settings.run_kernels:
            self.results["kernels"] = self._timed_step(
                "kernels",
                self.run_kernel_node,
            )

        if self.settings.run_iq_blobs:
            self.results["iq_blobs"] = self._timed_step(
                "iq_blobs",
                self.run_iq_blobs_node,
            )

        self.timings["total"] = perf_counter() - workflow_start
        self._timing_print(
            f"workflow finished in {self._format_duration(self.timings['total'])}"
        )
        return self.results

    def run_resonator_node(self) -> Any:
        handler = self._build_resonator_handler()
        self.resonator_handler = handler

        if self.settings.do_emulation:
            self._run_handler_locally(handler)
        else:
            with self._optional_output_suppression():
                result = self._submit_handler(handler)
            self._load_handler_result(handler, result)

        self._update_profile_from_resonator(handler)

        return handler.data

    def run_kernel_node(self) -> Any:
        handler = self._build_kernel_handler()
        self.kernel_handler = handler

        if self.settings.do_emulation:
            self._run_handler_locally(handler)
        else:
            with self._optional_output_suppression():
                result_0, result_1 = self._submit_kernel_handler(handler)
            self._load_kernel_handler_result(handler, result_0, result_1)

        return handler.data

    def run_iq_blobs_node(self) -> Any:
        handler = self._build_iq_blobs_handler()
        self.iq_blobs_handler = handler

        if self.settings.do_emulation:
            self._run_handler_locally(handler)
        else:
            with self._optional_output_suppression():
                result = self._submit_handler(handler)
            self._load_handler_result(handler, result)

        return handler.data

    def _run_handler_locally(self, handler: ExperimentHandler) -> Any:
        """Run a handler without the task manager and retain any figures it creates."""
        existing_figures = set(plt.get_fignums())
        with self._optional_output_suppression():
            result = handler.run()

        self._retain_handler_figures(handler, existing_figures)

        return result

    def _submit_handler(self, handler: ExperimentHandler) -> Any:
        compiled_experiment = handler.get_compiled_experiment()

        return self._submit_compiled_experiment(handler, compiled_experiment)

    def _submit_compiled_experiment(
        self,
        handler: ExperimentHandler,
        compiled_experiment: Any,
    ) -> Any:
        label = handler.experiment_name
        submit_start = perf_counter()
        task = self.task_manager.run_compiled_experiment(
            handler.experiment_name,
            self.settings.profile_name,
            handler.qubit_names,
            compiled_experiment,
            do_emulation=False,
        )
        self._timing_print(
            f"{label} submitted in {self._format_duration(perf_counter() - submit_start)}"
        )
        return self._wait_for_task(task, label)

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

    def _load_kernel_handler_result(
        self,
        handler: KernelTracesCalculationHandler,
        result_0: Any,
        result_1: Any,
    ) -> None:
        with self._optional_output_suppression():
            if hasattr(handler, "load_results"):
                handler.load_results(result_0, result_1)
            else:
                handler.load_result((result_0, result_1))
        self._analyze_handler_result(handler)

    def _analyze_handler_result(self, handler: ExperimentHandler) -> None:
        existing_figures = set(plt.get_fignums())
        with self._optional_output_suppression():
            handler.analyze()
            handler.update_system_params()

        self._retain_handler_figures(handler, existing_figures)

    def _retain_handler_figures(
        self,
        handler: ExperimentHandler,
        existing_figures: set[int],
    ) -> None:
        new_figures = set(plt.get_fignums()) - existing_figures
        figures = self._handler_figures(handler)
        figures.extend(plt.figure(number) for number in sorted(new_figures))
        figures = self._unique_figures(figures)

        if not figures:
            figures = self._plot_handler_for_artifacts(
                handler,
                suppress_display=not self.settings.do_plotting,
            )

        if figures:
            handler.workflow_figures = figures
            if not self.settings.do_plotting:
                for figure in figures:
                    plt.close(figure)

    def _plot_handler_for_artifacts(
        self,
        handler: ExperimentHandler,
        suppress_display: bool,
    ) -> list[Figure]:
        existing_figures = set(plt.get_fignums())

        if suppress_display:
            original_show = plt.show
            try:
                plt.show = lambda *args, **kwargs: None
                with plt.ioff(), self._optional_output_suppression():
                    plot_result = handler.plot()
            finally:
                plt.show = original_show
        else:
            with self._optional_output_suppression():
                plot_result = handler.plot()

        if isinstance(plot_result, Figure):
            figures = [plot_result]
        elif isinstance(plot_result, (list, tuple)):
            figures = [
                figure for figure in plot_result if isinstance(figure, Figure)
            ]
        else:
            figures = self._handler_figures(handler)

        new_figures = set(plt.get_fignums()) - existing_figures
        figures.extend(plt.figure(number) for number in sorted(new_figures))
        return self._unique_figures(figures)

    def _handler_figures(self, handler: ExperimentHandler) -> list[Figure]:
        figures = []
        for attribute_name in ("workflow_figures", "figs", "figures"):
            figures.extend(self._extract_figures(
                getattr(handler, attribute_name, None)))

        figure = getattr(handler, "fig", None)
        figures.extend(self._extract_figures(figure))

        return self._unique_figures(figures)

    def _extract_figures(self, value: Any) -> list[Figure]:
        if isinstance(value, Figure):
            return [value]
        if hasattr(value, "figure") and isinstance(value.figure, Figure):
            return [value.figure]
        if isinstance(value, dict):
            figures = []
            for item in value.values():
                figures.extend(self._extract_figures(item))
            return figures
        if isinstance(value, (list, tuple, set)):
            figures = []
            for item in value:
                figures.extend(self._extract_figures(item))
            return figures
        return []

    def _unique_figures(self, figures: list[Figure]) -> list[Figure]:
        unique_figures = []
        seen_ids = set()
        for figure in figures:
            figure_id = id(figure)
            if figure_id in seen_ids:
                continue
            unique_figures.append(figure)
            seen_ids.add(figure_id)
        return unique_figures

    def _timed_step(self, name: str, callback) -> Any:
        self._timing_print(f"{name} started")
        start = perf_counter()
        try:
            result = callback()
        except Exception:
            elapsed = perf_counter() - start
            self.timings[name] = elapsed
            self._timing_print(
                f"{name} failed after {self._format_duration(elapsed)}"
            )
            raise

        elapsed = perf_counter() - start
        self.timings[name] = elapsed
        self._timing_print(
            f"{name} finished in {self._format_duration(elapsed)}")
        return result

    def _wait_for_task(self, task: Any, label: str) -> Any:
        wait_start = perf_counter()
        initial_status = self._task_status(task)
        if initial_status:
            self._timing_print(f"{label} status: {initial_status}")

        stop_event = threading.Event()
        poll_thread = self._start_status_polling(task, label, stop_event)
        try:
            return self.task_manager.wait(task)
        finally:
            stop_event.set()
            if poll_thread is not None:
                poll_thread.join(timeout=0.2)

            elapsed = perf_counter() - wait_start
            final_status = self._task_status(task)
            status_suffix = f" final status: {final_status}" if final_status else ""
            self._timing_print(
                f"{label} wait finished in {self._format_duration(elapsed)}"
                f"{status_suffix}"
            )

    def _start_status_polling(
        self,
        task: Any,
        label: str,
        stop_event: threading.Event,
    ) -> threading.Thread | None:
        interval = float(self.settings.task_status_poll_interval)
        if interval <= 0:
            return None

        def poll_status() -> None:
            last_status = self._task_status(task)
            while not stop_event.wait(interval):
                status = self._task_status(task)
                if status and status != last_status:
                    self._timing_print(f"{label} status: {status}")
                    last_status = status
                else:
                    elapsed = self._format_duration(
                        perf_counter() - wait_start)
                    self._timing_print(f"{label} waiting for {elapsed}")

        wait_start = perf_counter()
        thread = threading.Thread(target=poll_status, daemon=True)
        thread.start()
        return thread

    def _task_status(self, task: Any) -> str | None:
        for source in (task, self.task_manager):
            status = self._status_from_source(source, task)
            if status:
                return status
        return None

    def _status_from_source(self, source: Any, task: Any) -> str | None:
        for name in ("status", "state", "task_status", "task_state"):
            value = getattr(source, name, None)
            status = self._read_status_value(value, task)
            if status:
                return status

        for name in ("get_status", "get_task_status", "get_task_state"):
            method = getattr(source, name, None)
            if callable(method):
                for args in ((task,), ()):
                    try:
                        status = method(*args)
                    except TypeError:
                        continue
                    except Exception:
                        break
                    if status is not None:
                        return str(status)

        return None

    def _read_status_value(self, value: Any, task: Any) -> str | None:
        if value is None:
            return None
        if callable(value):
            for args in ((), (task,)):
                try:
                    status = value(*args)
                except TypeError:
                    continue
                except Exception:
                    return None
                if status is not None:
                    return str(status)
            return None
        return str(value)

    def _timing_print(self, message: str) -> None:
        if self.settings.report_timing:
            print(f"[readout workflow] {message}", flush=True)

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes, remainder = divmod(seconds, 60)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{remainder:04.1f}"
        if minutes:
            return f"{minutes:d}:{remainder:04.1f}"
        return f"{remainder:.1f}s"

    @contextlib.contextmanager
    def _optional_output_suppression(self):
        if self.settings.show_handler_output:
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
            num_shots=300,
            exportation_method=ExportationMethod.NONE,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            update_params_method=UpdateParamsMethod.NONE,
            configure_logging=self.settings.show_handler_output,
            do_emulation=True,
            do_plotting=self.settings.do_plotting,
        )
        handler = ResonatorSpectroscopyHandler(
            x_resonator_frequency_arrays=[MidIntervalArray(
                mid_point=None, interval=150e6, num_points=120)],
            long_drive_pulse=False,
            qubit_names=[self.qubit_names[0]],
            settings=settings,
            profile=self.profile,
            session=self.session,
        )

        return handler

    def _build_kernel_handler(self):
        settings = ExperimentSettings(
            log_level=0,
            num_shots=20000,
            exportation_method=ExportationMethod.NONE,
            update_params_method=UpdateParamsMethod.UPDATE,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
            configure_logging=self.settings.show_handler_output,
            do_emulation=True,
            do_plotting=self.settings.do_plotting,
        )
        handler = KernelTracesCalculationHandler(
            qubit_names=[self.qubit_names[0]],
            settings=settings,
            profile=self.profile,
            session=self.session,
        )

        return handler

    def _build_iq_blobs_handler(self):
        settings = IQBlobsSettings(
            num_shots=10000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            exportation_method=ExportationMethod.NONE,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            configure_logging=self.settings.show_handler_output,
            reset=self.settings.reset,
            do_emulation=True,
            do_plotting=self.settings.do_plotting,
            iq_plane_analysis='kde',
        )

        handler = IQBlobsHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            profile=self.profile,
            session=self.session,
        )

        return handler


if __name__ == "__main__":

    qubit_names = ["q6"]

    profile = load_profile()

    task_manager = load_task_manager()

    qubit = profile.qubits[qubit_names[0]]

    readout_pulse = qubit.pulses[
        SUPPORTED_PULSE_TYPES.readout][SUPPORTED_PULSE_SHAPES.const]

    readout_pulse.readout_amplitude = 0.02
    readout_pulse.readout_duration = 1e-6

    # %%
    settings = ReadoutFidelityWorkflowSettings(
        profile_name="main",
        do_emulation=False,
        run_resonator=True,
        run_kernels=True,
        run_iq_blobs=True,
        show_handler_output=True,
        reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
        do_plotting=True,
    )

    workflow = ReadoutFidelityWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
    )

    workflow.run()

    # figures_dir = Path("data") / "readout_workflow_example_figures"
    # figures_dir.mkdir(parents=True, exist_ok=True)

    # handlers = {
    #     "resonator": workflow.resonator_handler,
    #     "kernels": workflow.kernel_handler,
    #     "iq_blobs": workflow.iq_blobs_handler,
    # }

    # for handler_name, handler in handlers.items():
    #     if handler is None:
    #         continue

    #     figures = getattr(handler, "workflow_figures", [])
    #     print(f"{handler_name}: {len(figures)} figures")
    #     for figure_index, figure in enumerate(figures, start=1):
    #         path = figures_dir / f"{handler_name}_{figure_index:02d}.png"
    #         figure.savefig(path, dpi=200, bbox_inches="tight")
    #         print(f"saved {path}")
