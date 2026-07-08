from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if __name__ == "__main__" and __package__ in (None, ""):
    WORKBENCH_ROOT = Path(__file__).resolve().parents[2]
    if str(WORKBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKBENCH_ROOT))

    from workbench_bootstrap import setup_workbench_environment

    setup_workbench_environment()

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.dsl.session import Session
from laboneq.simple import from_json
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.experiments.experiment_handler import ExperimentHandler
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.kernel_traces_calculation import KernelTracesCalculationHandler
from qratena.experiments.resonator_spectroscopy import ResonatorSpectroscopyHandler
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.system.qratena_platform import create_platform
from qratena.util.enums import (
    SUPPORTED_PULSE_SHAPES,
    ExportationMethod,
    ResetType,
    UpdateParamsMethod,
)
from qratena.util.sweeps_utils import MidIntervalArray

if TYPE_CHECKING:
    from qigeon import TaskSubmitterAsync
else:
    TaskSubmitterAsync = Any


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
    task_execution_mode: str = "wait"
    low_priority_tasks: bool = False
    reset: ResetSettings = field(default_factory=ResetSettings)
    states: list[str] = field(default_factory=lambda: ["g", "e"])


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
        self.resonator_handlers = []
        self.kernel_handlers = []

        self.results: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self.submitted_tasks: list[dict[str, Any]] = []

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
        self._timing_print(f"workflow finished in {self._format_duration(self.timings['total'])}")
        return self.results

    def run_resonator_node(self) -> Any:
        self.resonator_handlers = []
        resonator_data = {}

        for qubit_name in self.qubit_names:
            handler = self._build_resonator_handler(qubit_name)
            self.resonator_handlers.append(handler)
            self.resonator_handler = handler

            if self.settings.do_emulation:
                self._run_handler_locally(handler)
            else:
                with self._optional_output_suppression():
                    result = self._submit_handler(handler)
                if self._submit_only:
                    resonator_data[qubit_name] = result
                    continue
                self._load_handler_result(handler, result)

            self._update_profile_from_resonator(handler)
            resonator_data.update(handler.data)

        return resonator_data

    def run_kernel_node(self) -> Any:
        self._validate_kernel_states()
        self.kernel_handlers = []
        handler = self._build_kernel_handler()
        self.kernel_handlers.append(handler)
        self.kernel_handler = handler

        if self.settings.do_emulation:
            self._run_handler_locally(handler)
        else:
            with self._optional_output_suppression():
                result = self._submit_handler(handler)
            if self._submit_only:
                return result
            self._load_handler_result(handler, result)

        return handler.data

    def _validate_kernel_states(self) -> None:
        if self.settings.states not in (["g", "e"], ["g", "e", "f"]):
            raise ValueError(
                "Kernel traces calculation states must be ['g', 'e'] or ['g', 'e', 'f']."
            )

    def run_iq_blobs_node(self) -> Any:
        handler = self._build_iq_blobs_handler()
        self.iq_blobs_handler = handler

        if self.settings.do_emulation:
            self._run_handler_locally(handler)
        else:
            with self._optional_output_suppression():
                result = self._submit_handler(handler)
            if self._submit_only:
                return result
            self._load_handler_result(handler, result)
            self.iq_blobs_handler.export_data()  # ensure data is exported if not done in analyze()
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
        handler.compiled_experiment = compiled_experiment

        return self._submit_compiled_experiment(handler, compiled_experiment)

    def _submit_compiled_experiment(
        self,
        handler: ExperimentHandler,
        compiled_experiment: Any,
    ) -> Any:
        label = handler.experiment_name
        submit_start = perf_counter()
        task_id = self.task_manager.submit_compiled_experiment(
            handler.experiment_name,
            self.settings.profile_name,
            handler.qubit_names,
            compiled_experiment,
            do_emulation=False,
            low_priority=self.settings.low_priority_tasks,
        )
        self._timing_print(
            f"{label} submitted in {self._format_duration(perf_counter() - submit_start)}"
        )
        task_record = self._task_record(handler, task_id)
        self.submitted_tasks.append(task_record)
        if self._submit_only:
            return task_record
        return self._wait_for_task(task_id, label)

    @property
    def _submit_only(self) -> bool:
        return self.settings.task_execution_mode == "submit_only"

    def _task_record(self, handler: ExperimentHandler, task_id: Any) -> dict[str, Any]:
        experiment_name = str(handler.experiment_name)
        qubit_names = list(getattr(handler, "qubit_names", []) or [])
        return {
            "task_id": self._task_id_value(task_id),
            "task_id_repr": repr(task_id),
            "task_key": self._task_key(experiment_name, qubit_names),
            "experiment_name": experiment_name,
            "qubit_names": qubit_names,
            "node": self._node_name(experiment_name),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "low_priority": bool(self.settings.low_priority_tasks),
            "result_status": "pending",
            "result_path": None,
        }

    def _task_id_value(self, task_id: Any) -> str:
        if isinstance(task_id, (str, int)):
            return str(task_id)

        for attribute_name in ("task_id", "id", "uid"):
            value = getattr(task_id, attribute_name, None)
            if value is not None:
                return str(value)

        return str(task_id)

    def _task_key(self, experiment_name: str, qubit_names: list[str]) -> str:
        qubit_slug = "+".join(qubit_names) if qubit_names else "no_qubits"
        return f"{experiment_name}/{qubit_slug}"

    def _node_name(self, experiment_name: str) -> str:
        lowered = experiment_name.lower()
        if "resonator" in lowered:
            return "resonator"
        if "kernel" in lowered:
            return "kernels"
        if "iq" in lowered or "blob" in lowered:
            return "iq_blobs"
        return experiment_name

    def collect_submitted_results(
        self,
        task_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Fetch saved task IDs and analyze them through freshly built handlers."""
        previous_mode = self.settings.task_execution_mode
        self.settings.task_execution_mode = "wait"
        try:
            self.results = {}
            self.timings = {}
            entries_by_node = self._entries_by_node(task_entries)

            if self.settings.run_resonator:
                self.results["resonator"] = self._collect_resonator_results(
                    entries_by_node.get("resonator", [])
                )

            if self.settings.run_kernels:
                self.results["kernels"] = self._collect_kernel_results(
                    entries_by_node.get("kernels", [])
                )

            if self.settings.run_iq_blobs:
                self.results["iq_blobs"] = self._collect_iq_blobs_results(
                    entries_by_node.get("iq_blobs", [])
                )

            return self.results
        finally:
            self.settings.task_execution_mode = previous_mode

    def _entries_by_node(
        self,
        task_entries: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        entries_by_node: dict[str, list[dict[str, Any]]] = {}
        for entry in task_entries:
            entries_by_node.setdefault(str(entry.get("node", "")), []).append(entry)
        return entries_by_node

    def _collect_resonator_results(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.resonator_handlers = []
        resonator_data = {}

        entries_by_qubit = self._entries_by_single_qubit(entries)
        for qubit_name in self.qubit_names:
            entry = entries_by_qubit.get(qubit_name)
            if entry is None:
                continue
            handler = self._build_resonator_handler(qubit_name)
            self.resonator_handlers.append(handler)
            self.resonator_handler = handler
            result = self._wait_for_task(entry["task_id"], entry["task_key"])
            self._load_handler_result(handler, result)
            self._update_profile_from_resonator(handler)
            resonator_data.update(handler.data)

        return resonator_data

    def _collect_kernel_results(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._validate_kernel_states()
        self.kernel_handlers = []
        entry = entries[0] if entries else None
        if entry is None:
            return {}

        handler = self._build_kernel_handler()
        self.kernel_handlers.append(handler)
        self.kernel_handler = handler
        result = self._wait_for_task(entry["task_id"], entry["task_key"])
        self._load_handler_result(handler, result)

        return handler.data

    def _collect_iq_blobs_results(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        entry = entries[0] if entries else None
        if entry is None:
            return {}

        handler = self._build_iq_blobs_handler()
        self.iq_blobs_handler = handler
        result = self._wait_for_task(entry["task_id"], entry["task_key"])
        self._load_handler_result(handler, result)
        handler.export_data()
        return handler.data

    def _entries_by_single_qubit(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        entries_by_qubit = {}
        for entry in entries:
            qubit_names = entry.get("qubit_names", [])
            if len(qubit_names) == 1:
                entries_by_qubit[str(qubit_names[0])] = entry
        return entries_by_qubit

    def _load_handler_result(self, handler: ExperimentHandler, result: Any) -> None:
        """Convert task-manager result into the handler's normal data format.

        Implement this once in BaseExperimentHandler if possible.
        """
        with self._optional_output_suppression():
            handler.experiment_result = self._deserialize_laboneq_result(result.raw_data)
        self._analyze_handler_result(handler)

    def _deserialize_laboneq_result(self, raw_data: Any) -> Any:
        try:
            return from_json(raw_data)
        except TypeError:
            fixed_raw_data, fixed_count = self._fill_missing_acquired_data_parts(raw_data)
            if fixed_count == 0:
                raise

            try:
                return from_json(fixed_raw_data)
            except Exception as retry_error:
                raise RuntimeError(
                    "LabOneQ result deserialization failed after replacing "
                    f"{fixed_count} missing acquired-result real/imag component(s)."
                ) from retry_error

    def _fill_missing_acquired_data_parts(self, raw_data: Any) -> tuple[Any, int]:
        decoded_data = self._decode_raw_json(raw_data)
        fixed_count = self._fill_missing_acquired_data_parts_in_place(decoded_data)
        if fixed_count == 0:
            return raw_data, 0

        encoded_data = json.dumps(decoded_data)
        if isinstance(raw_data, bytes):
            return encoded_data.encode(), fixed_count
        return encoded_data, fixed_count

    def _decode_raw_json(self, raw_data: Any) -> Any:
        if isinstance(raw_data, memoryview):
            raw_data = raw_data.tobytes()
        return json.loads(raw_data)

    def _fill_missing_acquired_data_parts_in_place(self, value: Any) -> int:
        if isinstance(value, dict):
            fixed_count = self._fill_acquired_result_data_part(value)
            for item in value.values():
                fixed_count += self._fill_missing_acquired_data_parts_in_place(item)
            return fixed_count

        if isinstance(value, list):
            return sum(self._fill_missing_acquired_data_parts_in_place(item) for item in value)

        return 0

    def _fill_acquired_result_data_part(self, value: dict[str, Any]) -> int:
        if "data.real" not in value or "data.imag" not in value:
            return 0

        fixed_count = 0
        if value["data.real"] is None and value["data.imag"] is not None:
            value["data.real"] = self._zero_like_json_number_tree(value["data.imag"])
            fixed_count += 1

        if value["data.imag"] is None and value["data.real"] is not None:
            value["data.imag"] = self._zero_like_json_number_tree(value["data.real"])
            fixed_count += 1

        return fixed_count

    def _zero_like_json_number_tree(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._zero_like_json_number_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: self._zero_like_json_number_tree(item) for key, item in value.items()}
        return 0.0

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
            figures = [figure for figure in plot_result if isinstance(figure, Figure)]
        else:
            figures = self._handler_figures(handler)

        new_figures = set(plt.get_fignums()) - existing_figures
        figures.extend(plt.figure(number) for number in sorted(new_figures))
        return self._unique_figures(figures)

    def _handler_figures(self, handler: ExperimentHandler) -> list[Figure]:
        figures = []
        for attribute_name in ("workflow_figures", "figs", "figures"):
            figures.extend(self._extract_figures(getattr(handler, attribute_name, None)))

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
            self._timing_print(f"{name} failed after {self._format_duration(elapsed)}")
            raise

        elapsed = perf_counter() - start
        self.timings[name] = elapsed
        self._timing_print(f"{name} finished in {self._format_duration(elapsed)}")
        return result

    def _wait_for_task(self, task_id: Any, label: str) -> Any:
        wait_start = perf_counter()
        initial_status = self._task_status(task_id)
        if initial_status:
            self._timing_print(f"{label} status: {initial_status}")

        stop_event = threading.Event()
        poll_thread = self._start_status_polling(task_id, label, stop_event)
        try:
            return self.task_manager.wait_for_result(task_id)
        finally:
            stop_event.set()
            if poll_thread is not None:
                poll_thread.join(timeout=0.2)

            elapsed = perf_counter() - wait_start
            final_status = self._task_status(task_id)
            status_suffix = f" final status: {final_status}" if final_status else ""
            self._timing_print(
                f"{label} wait finished in {self._format_duration(elapsed)}{status_suffix}"
            )

    def _start_status_polling(
        self,
        task_id: Any,
        label: str,
        stop_event: threading.Event,
    ) -> threading.Thread | None:
        interval = float(self.settings.task_status_poll_interval)
        if interval <= 0:
            return None

        def poll_status() -> None:
            last_status = self._task_status(task_id)
            while not stop_event.wait(interval):
                status = self._task_status(task_id)
                if status and status != last_status:
                    self._timing_print(f"{label} status: {status}")
                    last_status = status
                else:
                    elapsed = self._format_duration(perf_counter() - wait_start)
                    self._timing_print(f"{label} waiting for {elapsed}")

        wait_start = perf_counter()
        thread = threading.Thread(target=poll_status, daemon=True)
        thread.start()
        return thread

    def _task_status(self, task_id: Any) -> str | None:
        for source in (task_id, self.task_manager):
            status = self._status_from_source(source, task_id)
            if status:
                return status
        return None

    def _status_from_source(self, source: Any, task_id: Any) -> str | None:
        for name in ("status", "state", "task_status", "task_state"):
            value = getattr(source, name, None)
            status = self._read_status_value(value, task_id)
            if status:
                return status

        for name in ("get_status", "get_task_status", "get_task_state"):
            method = getattr(source, name, None)
            if callable(method):
                for args in ((task_id,), ()):
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
        for qubit_name in handler.qubit_names:
            optimal_frequency = handler.data[qubit_name]["optimal_resonance_freq"]
            self.profile.qubits[qubit_name].readout_resonator_frequency.value = optimal_frequency

    def _build_resonator_handler(self, qubit_name: str):
        settings = ExperimentSettings(
            log_level=0,
            num_shots=300,
            exportation_method=ExportationMethod.NONE,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            update_params_method=UpdateParamsMethod.NONE,
            # configure_logging=self.settings.show_handler_output,
            do_emulation=True,
            # do_plotting=self.settings.do_plotting,
        )
        handler = ResonatorSpectroscopyHandler(
            x_resonator_frequency_arrays=[
                MidIntervalArray(mid_point=None, interval=150e6, num_points=120)
            ],
            long_drive_pulse=False,
            qubit_names=[qubit_name],
            settings=settings,
            profile=self.profile,
            session=self.session,
            states=self.settings.states,
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
            # configure_logging=self.settings.show_handler_output,
            do_emulation=True,
            # do_plotting=self.settings.do_plotting,
        )
        handler = KernelTracesCalculationHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            profile=self.profile,
            session=self.session,
            states=self.settings.states,
        )

        return handler

    def _build_iq_blobs_handler(self):
        settings = ExperimentSettings(
            num_shots=10000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            exportation_method=ExportationMethod.NONE,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            reset=self.settings.reset,
            do_emulation=True,
        )

        handler = IQBlobsHandler(
            qubit_names=self.qubit_names,
            settings=settings,
            states=self.settings.states,
        )
        handler.configuration_params = self.profile
        handler.platform = create_platform(self.profile)
        handler.device_setup = handler.platform.setup
        if self.session is not None:
            handler.session = self.session

        return handler


def main() -> None:
    from resources.load_profile import load_profile, load_task_manager

    profile_name = "main_asaf"
    qubit_names = ["q3"]

    states = ["g", "e"]

    do_emulation = False  # Set to True to run the workflow without a task manager
    show_plots = True

    profile = load_profile(profile_name)
    profile.ensure_pi_ef_pulse_for_all_qubits()

    task_manager = object() if do_emulation else load_task_manager()

    settings = ReadoutFidelityWorkflowSettings(
        profile_name=profile_name,
        do_emulation=do_emulation,
        run_resonator=False,
        run_kernels=True,
        run_iq_blobs=True,
        do_plotting=show_plots,
        show_handler_output=True,
        reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=5),
        states=states,
    )
    workflow = ReadoutFidelityWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
    )
    results = workflow.run()
    print(f"Readout workflow result keys: {', '.join(results)}")
    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
