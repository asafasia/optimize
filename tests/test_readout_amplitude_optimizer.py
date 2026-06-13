from types import SimpleNamespace

import pytest

from optimize.readout import readout_amplitude_optimizer as optimizer_mod
from optimize.readout.readout_amplitude_optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
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
