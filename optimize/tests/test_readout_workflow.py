from __future__ import annotations

import json
from types import SimpleNamespace

from optimize.readout import readout_workflow
from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
    ReadoutFidelityWorkflowSettings,
)


def test_deserialize_laboneq_result_fills_missing_real_data(monkeypatch):
    workflow = object.__new__(ReadoutFidelityWorkflow)
    raw_data = json.dumps(
        {
            "__data__": {
                "acquired_results": {
                    "handle_q3": {
                        "data.real": None,
                        "data.imag": [[1.0, 2.0], [3.0, 4.0]],
                    }
                }
            }
        }
    ).encode()
    calls = []

    def fake_from_json(data):
        calls.append(json.loads(data))
        if calls[-1]["__data__"]["acquired_results"]["handle_q3"]["data.real"] is None:
            raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'complex'")
        return "decoded"

    monkeypatch.setattr(readout_workflow, "from_json", fake_from_json)

    result = workflow._deserialize_laboneq_result(raw_data)

    assert result == "decoded"
    assert len(calls) == 2
    assert calls[1]["__data__"]["acquired_results"]["handle_q3"]["data.real"] == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_deserialize_laboneq_result_reraises_unrelated_type_errors(monkeypatch):
    workflow = object.__new__(ReadoutFidelityWorkflow)

    def fake_from_json(data):
        raise TypeError("different failure")

    monkeypatch.setattr(readout_workflow, "from_json", fake_from_json)

    try:
        workflow._deserialize_laboneq_result("{}")
    except TypeError as error:
        assert str(error) == "different failure"
    else:
        raise AssertionError("expected TypeError")


def test_resonator_runs_per_qubit_kernel_runs_once_for_all_qubits(monkeypatch):
    calls = []

    class FakeProfile:
        def __init__(self, qubit_names):
            self.qubits = {
                qubit_name: SimpleNamespace(
                    readout_resonator_frequency=SimpleNamespace(value=None)
                )
                for qubit_name in qubit_names
            }

    class FakePlatform:
        setup = object()

    class FakeHandler:
        experiment_kind = "handler"

        def __init__(self, *, qubit_names, **kwargs):
            self.qubit_names = qubit_names
            self.data = {}
            self.workflow_figures = []

        @property
        def experiment_name(self):
            return self.experiment_kind

        def run(self):
            calls.append((self.experiment_kind, tuple(self.qubit_names)))
            self.data = {
                qubit_name: self._qubit_data(qubit_name) for qubit_name in self.qubit_names
            }

        def plot(self):
            return None

        def _qubit_data(self, qubit_name):
            return {}

    class FakeResonatorHandler(FakeHandler):
        experiment_kind = "resonator"

        def _qubit_data(self, qubit_name):
            return {"optimal_resonance_freq": 7.0e9 + int(qubit_name[1:])}

    class FakeKernelHandler(FakeHandler):
        experiment_kind = "kernel"

        def _qubit_data(self, qubit_name):
            return {"kernel": f"kernel-{qubit_name}"}

    class FakeIQBlobsHandler(FakeHandler):
        experiment_kind = "iq_blobs"

        def _qubit_data(self, qubit_name):
            return {"readout_fidelity": 0.9}

        def export_data(self):
            calls.append(("export_iq_blobs", tuple(self.qubit_names)))

    monkeypatch.setattr(
        readout_workflow,
        "ResonatorSpectroscopyHandler",
        FakeResonatorHandler,
    )
    monkeypatch.setattr(
        readout_workflow,
        "KernelTracesCalculationHandler",
        FakeKernelHandler,
    )
    monkeypatch.setattr(readout_workflow, "IQBlobsHandler", FakeIQBlobsHandler)
    monkeypatch.setattr(readout_workflow, "create_platform", lambda profile: FakePlatform())

    qubit_names = ["q1", "q3"]
    profile = FakeProfile(qubit_names)
    workflow = ReadoutFidelityWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=object(),
        settings=ReadoutFidelityWorkflowSettings(
            do_emulation=True,
            run_resonator=True,
            run_kernels=True,
            run_iq_blobs=True,
            show_handler_output=False,
            report_timing=False,
        ),
    )

    result = workflow.run()

    assert calls == [
        ("resonator", ("q1",)),
        ("resonator", ("q3",)),
        ("kernel", ("q1", "q3")),
        ("iq_blobs", ("q1", "q3")),
    ]
    assert result["resonator"].keys() == {"q1", "q3"}
    assert result["kernels"].keys() == {"q1", "q3"}
    assert result["iq_blobs"].keys() == {"q1", "q3"}
    assert [handler.qubit_names for handler in workflow.resonator_handlers] == [
        ["q1"],
        ["q3"],
    ]
    assert [handler.qubit_names for handler in workflow.kernel_handlers] == [
        ["q1", "q3"],
    ]
    assert workflow.iq_blobs_handler.qubit_names == ["q1", "q3"]
    assert profile.qubits["q1"].readout_resonator_frequency.value == 7.0e9 + 1
    assert profile.qubits["q3"].readout_resonator_frequency.value == 7.0e9 + 3


def test_submit_only_records_task_id_without_waiting():
    class FakeTaskManager:
        def __init__(self):
            self.submitted = []
            self.waited = []

        def submit_compiled_experiment(self, *args, **kwargs):
            self.submitted.append((args, kwargs))
            return "task-123"

        def wait_for_result(self, task_id):
            self.waited.append(task_id)
            raise AssertionError("submit-only workflow should not wait for results")

    class FakeHandler:
        experiment_name = "iq_blobs"
        qubit_names = ["q1", "q2"]

        def get_compiled_experiment(self):
            return "compiled"

    workflow = ReadoutFidelityWorkflow(
        qubit_names=["q1", "q2"],
        profile=object(),
        task_manager=FakeTaskManager(),
        settings=ReadoutFidelityWorkflowSettings(
            task_execution_mode="submit_only",
            report_timing=False,
        ),
    )

    result = workflow._submit_handler(FakeHandler())

    assert result["task_id"] == "task-123"
    assert result["node"] == "iq_blobs"
    assert result["result_status"] == "pending"
    assert workflow.submitted_tasks == [result]
    assert workflow.task_manager.waited == []


def test_submit_compiled_experiment_passes_low_priority_to_task_manager():
    class FakeTaskManager:
        def __init__(self):
            self.kwargs = None

        def submit_compiled_experiment(self, *args, **kwargs):
            self.kwargs = kwargs
            return "task-low"

        def wait_for_result(self, task_id):
            return SimpleNamespace(raw_data="{}")

    class FakeHandler:
        experiment_name = "iq_blobs"
        qubit_names = ["q1"]

    task_manager = FakeTaskManager()
    workflow = ReadoutFidelityWorkflow(
        qubit_names=["q1"],
        profile=object(),
        task_manager=task_manager,
        settings=ReadoutFidelityWorkflowSettings(
            low_priority_tasks=True,
            report_timing=False,
            task_status_poll_interval=0,
        ),
    )

    workflow._submit_compiled_experiment(FakeHandler(), "compiled")

    assert task_manager.kwargs["low_priority"] is True
    assert workflow.submitted_tasks[0]["low_priority"] is True
