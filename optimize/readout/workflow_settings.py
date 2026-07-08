from __future__ import annotations

from dataclasses import dataclass, field

from qratena.system.components_params.reset_settings import ResetSettings


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
