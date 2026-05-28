# Readout Amplitude Optimizer

Tools for optimizing qubit readout amplitude using Qratena/Qigeon experiment
workflows.

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
    display_handler_plots=False,
    suppress_handler_output=True,
    reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
)

optimizer_settings = ReadoutAmplitudeSweepSettings(
    amplitudes=np.linspace(0.005, 0.15, 20),
    method=ReadoutScanMethod.SWEEP,
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
      plot.png
      profile.json
      iq_blobs/
```

Files:

- `data.npz`: raw workflow results and measured arrays
- `fidelities.csv`: amplitude, fidelity, error, separation, and mean fidelity
- `summary.json`: machine-readable analysis summary
- `report.md`: human-readable report
- `plot.png`: main optimizer plot
- `profile.json`: copied or best-effort serialized profile snapshot
- `iq_blobs/`: IQ blobs figures for each measured amplitude

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

## Important Modules

```text
readout/readout_amplitude_optimizer.py  # main optimizer workflow
readout/readout_workflow.py             # single-amplitude fidelity workflow
readout/utils/readout_scan_methods.py   # sweep / gradient / golden-section scans
readout/utils/readout_scan_types.py     # scan method enum
readout/utils/readout_sweep_analysis.py # analysis summary
readout/utils/readout_sweep_plotter.py  # plotting
readout/utils/readout_sweep_artifacts.py # saving artifacts
```

## Notes

- `amplitudes` are the actual sweep points for `SWEEP`.
- In `GRADIENT` and `GOLDEN_SECTION`, `amplitudes` define the search bounds and
  maximum number of measured amplitudes.
- Fidelity scoring currently uses the mean readout fidelity across the selected
  qubits.
- The optimizer is intended for iterative lab use and still favors explicit,
  inspectable behavior over a complicated optimizer.
