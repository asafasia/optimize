import json
from types import SimpleNamespace

import pytest
from matplotlib import pyplot as plt

from optimize.readout.optimizer import amplitude_sweep as optimizer_mod
from optimize.readout.optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.optimizer.plotter import ReadoutAmplitudeSweepPlotter
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES


class FakeProfile:
    def __init__(self, qubit_names=("q1", "q2")):
        self.ensure_pi_ef_calls = []
        self.qubits = {}
        for index, qubit_name in enumerate(qubit_names, start=1):
            readout = SimpleNamespace(
                readout_amplitude=0.01 * index,
                readout_duration=1e-6,
            )
            self.qubits[qubit_name] = SimpleNamespace(
                pulses={
                    SUPPORTED_PULSE_TYPES.readout: {
                        SUPPORTED_PULSE_SHAPES.const: readout,
                    },
                },
                readout_resonator_frequency=SimpleNamespace(
                    value=7e9 + index * 1e6,
                ),
            )

    def ensure_pi_ef_pulse(self, qubit_name, overwrite=False):
        self.ensure_pi_ef_calls.append((qubit_name, overwrite))


def make_optimizer(
    *,
    amplitudes=(0.02,),
    qubit_names=("q1",),
    profile=None,
    continue_on_measurement_error=False,
    states=None,
):
    workflow_settings = ReadoutFidelityWorkflowSettings()
    if states is not None:
        workflow_settings.states = states

    settings = ReadoutAmplitudeSweepSettings(
        amplitudes=list(amplitudes),
        continue_on_measurement_error=continue_on_measurement_error,
        auto_save_results=False,
        use_live_html_plotter=False,
        show_progress=False,
        workflow_settings=workflow_settings,
    )
    return ReadoutAmplitudeSweepWorkflow(
        qubit_names=list(qubit_names),
        profile=profile or FakeProfile(qubit_names),
        task_manager=object(),
        settings=settings,
    )


def workflow_result(qubit_names, fidelity=0.91):
    return {
        "iq_blobs": {
            qubit_name: {
                "readout_fidelity": fidelity,
                "readout_fidelity_std": 0.01,
                "separation": 3.5,
                "average_roundness": 0.8,
            }
            for qubit_name in qubit_names
        }
    }


def test_settings_fail_fast_on_measurement_errors_by_default():
    settings = ReadoutAmplitudeSweepSettings(amplitudes=[0.01])

    assert settings.continue_on_measurement_error is False


def test_workflow_and_optimizer_default_to_two_states():
    workflow_settings = ReadoutFidelityWorkflowSettings()
    optimizer_settings = ReadoutAmplitudeSweepSettings(amplitudes=[0.01])

    assert workflow_settings.states == ["g", "e"]
    assert optimizer_settings.workflow_settings.states == ["g", "e"]


def test_prepare_profile_skips_pi_ef_for_default_two_state_workflow():
    profile = FakeProfile(("q1", "q2"))
    optimizer = make_optimizer(profile=profile, qubit_names=("q2",))

    optimizer._prepare_profile_for_states()

    assert profile.ensure_pi_ef_calls == []


def test_prepare_profile_adds_pi_ef_for_explicit_three_state_workflow():
    profile = FakeProfile(("q1", "q2"))
    optimizer = make_optimizer(
        profile=profile,
        qubit_names=("q2",),
        states=["g", "e", "f"],
    )

    optimizer._prepare_profile_for_states()

    assert profile.ensure_pi_ef_calls == [("q2", False)]


def test_sweep_runs_workflow_and_records_metrics(monkeypatch):
    created_workflows = []

    class FakeWorkflow:
        resonator_handler = None
        kernel_handler = None
        iq_blobs_handler = None

        def __init__(self, qubit_names, profile, task_manager, settings):
            self.qubit_names = qubit_names
            created_workflows.append(self)

        def run(self):
            return workflow_result(self.qubit_names)

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    optimizer = make_optimizer(amplitudes=(0.02, 0.04))

    results = optimizer.run()

    assert len(created_workflows) == 2
    assert optimizer.measured_amplitudes == [0.02, 0.04]
    assert optimizer.fidelities["q1"] == [0.91, 0.91]
    assert optimizer.fidelity_errors["q1"] == [0.01, 0.01]
    assert optimizer.separations["q1"] == [3.5, 3.5]
    assert optimizer.roundnesses["q1"] == [0.8, 0.8]
    assert set(results) == {0.02, 0.04}
    readout = optimizer.profile.qubits["q1"].pulses[
        SUPPORTED_PULSE_TYPES.readout
    ][SUPPORTED_PULSE_SHAPES.const]
    assert readout.readout_amplitude == 0.04


def test_run_auto_saves_completed_results(monkeypatch, tmp_path):
    class FakeWorkflow:
        resonator_handler = None
        kernel_handler = None
        iq_blobs_handler = None

        def __init__(self, qubit_names, profile, task_manager, settings):
            self.qubit_names = qubit_names

        def run(self):
            return workflow_result(self.qubit_names)

    calls = []
    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    optimizer = make_optimizer(amplitudes=(0.02,))
    optimizer.settings.auto_save_results = True
    optimizer.settings.close_auto_saved_figure = False
    optimizer.settings.live_html_output_dir = tmp_path
    monkeypatch.setattr(optimizer, "plot", lambda: "figure")

    def fake_save_results(output_dir, figure):
        calls.append((output_dir, figure))
        optimizer.run_dir = tmp_path / "saved"
        return str(optimizer.run_dir)

    monkeypatch.setattr(optimizer, "save_results", fake_save_results)

    optimizer.run()

    assert optimizer.figure == "figure"
    assert calls == [(tmp_path, "figure")]


def test_plotter_uses_separate_panels_for_multiple_qubits():
    plotter = ReadoutAmplitudeSweepPlotter(
        qubit_names=["q1", "q2"],
        amplitudes=[0.02, 0.04],
        fidelities={
            "q1": [0.85, 0.91],
            "q2": [0.89, 0.87],
        },
    )

    figure = plotter.plot()

    try:
        assert len(figure.axes) == 4
        assert figure.axes[0].get_title() == "Qubit q1"
        assert figure.axes[2].get_title() == "Qubit q2"
    finally:
        plt.close(figure)


def test_measurement_error_is_raised_by_default(monkeypatch):
    class FailingWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise RuntimeError("compile failed")

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FailingWorkflow)
    optimizer = make_optimizer()

    with pytest.raises(RuntimeError, match="compile failed"):
        optimizer.run()

    assert optimizer.measured_amplitudes == []
    assert optimizer.measurement_errors == {}


def test_continue_on_measurement_error_records_placeholder(monkeypatch):
    class FailingWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise RuntimeError("compile failed")

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FailingWorkflow)
    optimizer = make_optimizer(continue_on_measurement_error=True)

    optimizer.run()

    assert optimizer.measured_amplitudes == [0.02]
    assert optimizer.fidelities["q1"] == [0.5]
    assert optimizer.measurement_errors == {0.02: "RuntimeError: compile failed"}


def test_duplicate_amplitude_reuses_existing_result(monkeypatch):
    run_count = 0

    class FakeWorkflow:
        resonator_handler = None
        kernel_handler = None
        iq_blobs_handler = None

        def __init__(self, qubit_names, **kwargs):
            self.qubit_names = qubit_names

        def run(self):
            nonlocal run_count
            run_count += 1
            return workflow_result(self.qubit_names, fidelity=0.87)

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    optimizer = make_optimizer(amplitudes=(0.02, 0.02))

    optimizer.run()

    assert run_count == 1
    assert optimizer.measured_amplitudes == [0.02]
    assert optimizer.fidelities["q1"] == [0.87]


def test_interrupt_pads_unfinished_amplitudes(monkeypatch):
    class InterruptingWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", InterruptingWorkflow)
    optimizer = make_optimizer(amplitudes=(0.02, 0.04))

    optimizer.run()

    assert optimizer.interrupted is True
    assert optimizer.interrupt_reason == "KeyboardInterrupt"
    assert optimizer.measured_amplitudes == [0.02, 0.04]
    assert optimizer.fidelities["q1"] == [0.5, 0.5]


def test_submit_only_sweep_saves_pending_manifest(monkeypatch, tmp_path):
    class FakeWorkflow:
        def __init__(self, qubit_names, profile, task_manager, settings):
            self.qubit_names = qubit_names
            self.settings = settings
            self.submitted_tasks = []

        def run(self):
            assert self.settings.task_execution_mode == "submit_only"
            amplitude = len(submitted_amplitudes)
            submitted_amplitudes.append(amplitude)
            self.submitted_tasks = [
                {
                    "task_id": f"task-{amplitude}-kernel-q1-q2",
                    "task_key": "kernel/q1+q2",
                    "experiment_name": "kernel",
                    "qubit_names": ["q1", "q2"],
                    "node": "kernels",
                    "result_status": "pending",
                    "result_path": None,
                },
                {
                    "task_id": f"task-{amplitude}-iq-q1-q2",
                    "task_key": "iq_blobs/q1+q2",
                    "experiment_name": "iq_blobs",
                    "qubit_names": ["q1", "q2"],
                    "node": "iq_blobs",
                    "result_status": "pending",
                    "result_path": None,
                },
            ]
            return {"submitted": True}

    submitted_amplitudes = []
    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    optimizer = make_optimizer(amplitudes=(0.02, 0.04), qubit_names=("q1", "q2"))
    optimizer.settings.submit_only = True
    optimizer.settings.workflow_settings.run_kernels = True
    optimizer.settings.workflow_settings.run_iq_blobs = True
    optimizer.settings.live_html_output_dir = tmp_path

    optimizer.run()

    assert optimizer.measured_amplitudes == []
    assert optimizer.submitted_amplitudes == [0.02, 0.04]
    manifest_path = optimizer.run_dir / "task_manifest.json"
    metadata_path = optimizer.run_dir / "metadata.json"
    assert manifest_path.exists()
    assert metadata_path.exists()
    assert (optimizer.run_dir / "results").is_dir()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert manifest["run_status"] == "submitted_pending_results"
    assert manifest["run_key"] == optimizer.run_dir.name
    assert metadata["run_key"] == optimizer.run_dir.name
    assert metadata["run_dir"] == str(optimizer.run_dir)
    assert manifest["amplitudes"] == [0.02, 0.04]
    assert manifest["task_count"] == 4
    assert manifest["tasks"][0]["sweep_parameters"] == {"readout_amplitude": 0.02}
    assert manifest["tasks"][0]["task_key"].startswith(
        "sweep/0000/readout_amplitude=0.02/kernels/q1+q2/"
    )
    assert manifest["tasks"][1]["task_key"].startswith(
        "sweep/0000/readout_amplitude=0.02/iq_blobs/q1+q2/"
    )


def test_submit_only_allows_kernels_and_iq_blobs_together(monkeypatch, tmp_path):
    class FakeWorkflow:
        def __init__(self, qubit_names, profile, task_manager, settings):
            self.settings = settings
            self.submitted_tasks = []

        def run(self):
            assert self.settings.task_execution_mode == "submit_only"
            assert self.settings.run_kernels is True
            assert self.settings.run_iq_blobs is True
            self.submitted_tasks = [
                {
                    "task_id": "kernel-task",
                    "task_key": "kernel/q1+q2",
                    "experiment_name": "kernel",
                    "qubit_names": ["q1", "q2"],
                    "node": "kernels",
                    "result_status": "pending",
                    "result_path": None,
                },
                {
                    "task_id": "iq-task",
                    "task_key": "iq_blobs/q1+q2",
                    "experiment_name": "iq_blobs",
                    "qubit_names": ["q1", "q2"],
                    "node": "iq_blobs",
                    "result_status": "pending",
                    "result_path": None,
                },
            ]
            return {"submitted": True}

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    optimizer = make_optimizer(amplitudes=(0.02, 0.04))
    optimizer.settings.submit_only = True
    optimizer.settings.workflow_settings.run_kernels = True
    optimizer.settings.workflow_settings.run_iq_blobs = True
    optimizer.settings.live_html_output_dir = tmp_path

    optimizer.run()

    assert optimizer.submitted_amplitudes == [0.02, 0.04]
    assert [task["node"] for task in optimizer.submitted_task_entries] == [
        "kernels",
        "iq_blobs",
        "kernels",
        "iq_blobs",
    ]


def test_submit_only_rejects_non_sweep_mode():
    optimizer = make_optimizer(amplitudes=(0.02, 0.04))
    optimizer.settings.submit_only = True
    optimizer.settings.workflow_settings.run_iq_blobs = False
    optimizer.settings.method = "gradient"

    with pytest.raises(ValueError, match="submit_only is supported only for sweep mode"):
        optimizer.run()


def test_collect_submitted_results_uses_manifest_tasks(monkeypatch, tmp_path):
    collected = []

    class FakeWorkflow:
        resonator_handler = None
        kernel_handler = None
        iq_blobs_handler = None
        resonator_handlers = []
        kernel_handlers = []

        def __init__(self, qubit_names, profile, task_manager, settings):
            self.qubit_names = qubit_names

        def collect_submitted_results(self, task_entries):
            collected.append([entry["task_id"] for entry in task_entries])
            return workflow_result(self.qubit_names, fidelity=0.88)

    monkeypatch.setattr(optimizer_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    run_dir = tmp_path / "pending"
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "run_status": "submitted_pending_results",
        "tasks": [
            {"task_id": "task-a", "amplitude": 0.04, "node": "iq_blobs"},
            {"task_id": "task-b", "amplitude": 0.02, "node": "iq_blobs"},
        ],
    }
    (run_dir / "task_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    optimizer = make_optimizer(amplitudes=(0.02, 0.04))

    results = optimizer.collect_submitted_results(run_dir, save_results=False)

    assert set(results) == {0.02, 0.04}
    assert collected == [["task-b"], ["task-a"]]
    assert optimizer.measured_amplitudes == [0.02, 0.04]
    assert optimizer.fidelities["q1"] == [0.88, 0.88]
    updated_manifest = json.loads((run_dir / "task_manifest.json").read_text())
    assert updated_manifest["run_status"] == "complete"
    assert [task["result_status"] for task in updated_manifest["tasks"]] == [
        "collected",
        "collected",
    ]


def test_check_submitted_results_reports_pending_tasks(tmp_path):
    class FakeTaskManager:
        def get_status(self, task_id):
            return {"task-a": "completed", "task-b": "running"}[task_id]

    run_dir = tmp_path / "pending"
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "run_status": "submitted_pending_results",
        "tasks": [
            {
                "task_id": "task-a",
                "task_key": "sweep/0000/readout_amplitude=0.02/iq_blobs/q1/00",
                "amplitude": 0.02,
            },
            {
                "task_id": "task-b",
                "task_key": "sweep/0001/readout_amplitude=0.04/iq_blobs/q1/00",
                "amplitude": 0.04,
            },
        ],
    }
    (run_dir / "task_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    optimizer = make_optimizer(amplitudes=(0.02, 0.04))
    optimizer.task_manager = FakeTaskManager()

    summary = optimizer.check_submitted_results(run_dir)

    assert summary["ready_to_collect"] is False
    assert summary["completed"] == 1
    assert summary["pending"] == 1
    assert "wait before collecting" in summary["message"]
    assert summary["pending_task_keys"] == [
        "sweep/0001/readout_amplitude=0.04/iq_blobs/q1/00"
    ]
    updated_manifest = json.loads((run_dir / "task_manifest.json").read_text())
    assert updated_manifest["run_status"] == "submitted_pending_results"
    assert [task["result_status"] for task in updated_manifest["tasks"]] == [
        "ready",
        "pending",
    ]


def test_collect_submitted_results_wait_false_checks_without_blocking(tmp_path):
    class FakeTaskManager:
        def get_status(self, task_id):
            return "completed"

        def wait_for_result(self, task_id):
            raise AssertionError("wait=False should not wait for results")

    run_dir = tmp_path / "pending"
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "run_status": "submitted_pending_results",
        "tasks": [
            {
                "task_id": "task-a",
                "task_key": "sweep/0000/readout_amplitude=0.02/iq_blobs/q1/00",
                "amplitude": 0.02,
            },
        ],
    }
    (run_dir / "task_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    optimizer = make_optimizer(amplitudes=(0.02,))
    optimizer.task_manager = FakeTaskManager()

    summary = optimizer.collect_submitted_results(run_dir, wait=False)

    assert summary["ready_to_collect"] is True
    assert summary["message"] == (
        "All submitted readout tasks are complete; results are ready to collect."
    )
    updated_manifest = json.loads((run_dir / "task_manifest.json").read_text())
    assert updated_manifest["run_status"] == "ready_to_collect"
    assert updated_manifest["tasks"][0]["task_status"] == "completed"
