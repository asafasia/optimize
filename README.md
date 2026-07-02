# Readout Amplitude Optimizer

Tools for optimizing qubit readout amplitude using Qratena/Qigeon experiment
workflows.

## All-Qubit HTML Report

`readout/run_all_qubits_report.py` is the class-based replacement for the
ad-hoc `run_all_qubits.py` script. It runs resonator spectroscopy, kernel
calculation, and IQ blobs for each selected qubit, followed by a passive-reset
IQ comparison. It saves a standalone HTML dashboard containing:

- readout amplitude, pulse length, and resonator frequency
- active/passive fidelity comparison across qubits
- pulse and resonator parameter sweeps across qubits
- resonator, kernel, and IQ experiment figures for each qubit
- CSV, JSON, raw-result, settings, error, and profile artifacts

```bash
python readout/run_all_qubits_report.py --qubits q5 q6 q9
```

Use `--skip-passive-comparison`, `--skip-resonator`, or `--skip-kernels` for a
shorter run. Results are written under `data/readout_all_qubits_report`.

This optimizer expects the Qratena repository to be on:

```text
feature/readout-experiment-refactor
```

The optimizer runs readout fidelity measurements at selected amplitudes,
analyzes the resulting fidelity metrics, plots the optimization curve, and saves
all run artifacts in a structured folder under `data/readout_optimize`.

## What It Does

For each selected readout amplitude, the optimizer can run:

1. Resonator spectroscopy
2. Kernel traces calculation
3. IQ blobs

It then extracts readout metrics from the IQ blobs result:

- readout fidelity
- readout fidelity error/std, when available
- IQ/readout separation, when available

The final plot contains:

- fidelity vs amplitude
- fidelity error band/error bars
- separation vs amplitude in a second subplot
- the original profile amplitude
- the best measured amplitude
- qubit name, readout pulse length, and active reset status

## Scan Methods

The scan method is selected with `ReadoutScanMethod`.

### Sweep

Measures every configured amplitude.

```python
method=ReadoutScanMethod.SWEEP
```

### Zoom In

Runs the configured amplitude vector as a rough sweep, then repeatedly sweeps
around the best point with a narrower interval. Each iteration keeps the same
number of points as the initial vector and remains within its original bounds.

Defaults:

```python
method=ReadoutScanMethod.ZOOM_IN
zoom_in_iterations=3
zoom_in_shrink_factor=0.75
```

`zoom_in_iterations` includes the initial rough sweep. A shrink factor of `0.75`
makes each new interval 75% as wide as the preceding interval.

### Gradient

A primitive gradient-ascent style scan.

- starts from amplitude `0.0`
- tests neighboring points
- moves toward the better fidelity
- stops on low improvement, small step size, or max run count

```python
method=ReadoutScanMethod.GRADIENT
```

More detail is in
[`readout/utils/readout_gradient_method.md`](readout/utils/readout_gradient_method.md).

### Golden Section

A coarse-to-fine interval search based on golden-section search.

- starts from the lower and upper amplitude bounds
- probes inside the interval
- narrows around the best measured region

```python
method=ReadoutScanMethod.GOLDEN_SECTION
```

## Basic Usage

Before running, make sure your local Qratena checkout is on:

```bash
git checkout feature/readout-experiment-refactor
```

```python
import numpy as np

from optimize.readout.readout_amplitude_optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType
from resources.load_profile import load_profile, load_task_manager


profile = load_profile()
task_manager = load_task_manager()

workflow_settings = ReadoutFidelityWorkflowSettings(
    profile_name="main",
    do_emulation=False,
    run_resonator=False,
    run_kernels=True,
    run_iq_blobs=True,
    do_plotting=False,
    show_handler_output=False,
    report_timing=True,
    task_status_poll_interval=10.0,
    states=["g", "e", "f"],
    reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
)

optimizer_settings = ReadoutAmplitudeSweepSettings(
    amplitudes=np.linspace(0.005, 0.15, 20),
    method=ReadoutScanMethod.SWEEP,
    use_live_html_plotter=True,
    workflow_settings=workflow_settings,
)

optimizer = ReadoutAmplitudeSweepWorkflow(
    qubit_names=["q5"],
    profile=profile,
    task_manager=task_manager,
    settings=optimizer_settings,
)

optimizer.run()
fig = optimizer.plot()
run_dir = optimizer.save_results(figure=fig)
```

Kernel traces run as one experiment with one compiled experiment and one result.
Supported kernel state lists are `["g", "e"]` and `["g", "e", "f"]`.

The experimental live HTML monitor opens by default while the optimizer runs.
Set `use_live_html_plotter=False` to disable it. The monitor writes files under
the same dated run folder used by `save_results()`. It has tabs for resonator,
kernel, and IQ blob figures beside the current fidelity-vs-amplitude plot. The figure panel
includes a slider for moving through acquired figures from earlier amplitudes,
and the HTML file can be opened again after the run. After `save_results()`
finishes, `live_readout_optimizer.html` is standalone: the plot images are
embedded in the HTML file, so it can be shared without the image folders.

Set `report_timing=True` in `ReadoutFidelityWorkflowSettings` to print elapsed
time for each workflow node. When running through the task manager, the workflow
also polls for task status every `task_status_poll_interval` seconds if the task
manager exposes a status API; otherwise it still prints how long it has been
waiting.

## Saved Output

Runs are saved by date and time:

```text
data/readout_optimize/
  2026-05-28/
    14-32-08_sweep_q5/
      data.npz
      fidelities.csv
      summary.json
      report.md
      live_readout_optimizer.html
      plot.png
      profile.json
      iq_blobs/
      kernels/
      resonator/
```

Files:

- `data.npz`: raw workflow results and measured arrays
- `fidelities.csv`: amplitude, fidelity, error, separation, roundness, resonator frequency, and mean fidelity
- `summary.json`: machine-readable analysis summary
- `report.md`: human-readable report including each qubit's readout frequency
- `live_readout_optimizer.html`: live/final browser view of the run
- `plot.png`: main optimizer plot
- `profile.json`: copied or best-effort serialized profile snapshot
- `iq_blobs/`: IQ blobs figures for each measured amplitude
- `kernels/`: kernel figures for each measured amplitude
- `resonator/`: resonator figures for each measured amplitude

## Interrupting a Run

If the run is stopped with `Ctrl+C` or `EOFError`, the optimizer can keep the
partial data usable for plotting and saving.

By default:

```python
fill_unfinished_on_interrupt=True
unfinished_fidelity=0.5
```

Unfinished configured amplitudes are appended with fidelity `0.5`, so the plot
and saved files are still produced. Set `fill_unfinished_on_interrupt=False` to
raise the interrupt normally.

## Failed Measurements

By default, a failed amplitude measurement aborts the optimization and exposes the original traceback:

```python
continue_on_measurement_error=False
failed_measurement_fidelity=0.5
```

Set `continue_on_measurement_error=True` to record the error, assign fidelity `0.5`
for that point, and continue to the next amplitude or scan step. Failed points
are marked in `fidelities.csv`, `summary.json`, `report.md`, and `data.npz`.

This continuation mode is useful for unattended scans, but it can hide systematic configuration failures.

## Important Modules

```text
readout/readout_amplitude_optimizer.py  # main optimizer workflow
readout/readout_workflow.py             # single-amplitude fidelity workflow
readout/utils/readout_scan_methods.py   # sweep / zoom-in / gradient / golden-section scans
readout/utils/readout_scan_types.py     # scan method enum
readout/utils/readout_sweep_analysis.py # analysis summary
readout/utils/readout_sweep_plotter.py  # plotting
readout/utils/readout_sweep_artifacts.py # saving artifacts
```

## Notes

- `amplitudes` are the actual sweep points for `SWEEP` and the initial rough
  sweep points and bounds for `ZOOM_IN`.
- In `GRADIENT` and `GOLDEN_SECTION`, `amplitudes` define the search bounds and
  maximum number of measured amplitudes.
- Fidelity scoring currently uses the mean readout fidelity across the selected
  qubits.
- The optimizer is intended for iterative lab use and still favors explicit,
  inspectable behavior over a complicated optimizer.
