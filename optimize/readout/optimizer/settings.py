from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.optimizer.scan_types import ReadoutScanMethod


@dataclass(slots=True)
class ReadoutAmplitudeSweepSettings:
    amplitudes: Any
    method: ReadoutScanMethod | str = ReadoutScanMethod.SWEEP
    zoom_in_iterations: int = 3
    zoom_in_shrink_factor: float = 0.5
    gradient_max_iterations: int = 5
    gradient_initial_step: float | None = None
    gradient_min_step: float = 0.001
    gradient_fidelity_tolerance: float = 0.01
    golden_section_max_iterations: int = 8
    golden_section_interval_tolerance: float = 0.001
    fill_unfinished_on_interrupt: bool = True
    unfinished_fidelity: float = 0.5
    continue_on_measurement_error: bool = False
    failed_measurement_fidelity: float = 0.5
    profile_path: str | Path | None = None
    show_progress: bool = True
    use_live_html_plotter: bool = True
    live_html_output_dir: str | Path = Path("data") / "readout_optimize"
    live_html_refresh_seconds: float = 1.0
    live_html_open_browser: bool = False
    auto_save_results: bool = True
    close_auto_saved_figure: bool = True
    submit_only: bool = False
    workflow_settings: ReadoutFidelityWorkflowSettings = field(
        default_factory=ReadoutFidelityWorkflowSettings
    )
