# Submit Readout Sweeps Now, Collect Results Later

Submit-only mode is for large readout amplitude sweeps where you want to submit
all task-manager experiments, stop the local process, and collect acquired
results later from saved task IDs.

Submit-only mode works only with:

- `ReadoutScanMethod.SWEEP`
- a real task manager, not `do_emulation=True`

The optimizer submits the enabled workflow nodes for each amplitude in one pass.
Kernel traces are submitted as one multi-qubit task per amplitude, followed by
the IQ blobs task for the same qubit set.

## Submit From Python

```python
import numpy as np

from optimize.readout.readout_amplitude_optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType
from resources.load_profile import load_profile, load_task_manager

profile = load_profile("main")
task_manager = load_task_manager()

workflow_settings = ReadoutFidelityWorkflowSettings(
    profile_name="main",
    do_emulation=False,
    run_resonator=False,
    run_kernels=True,
    run_iq_blobs=True,
    do_plotting=False,
    show_handler_output=False,
    low_priority_tasks=True,
    states=["g", "e"],
    reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
)

optimizer_settings = ReadoutAmplitudeSweepSettings(
    amplitudes=np.linspace(0.005, 0.15, 20),
    method=ReadoutScanMethod.SWEEP,
    submit_only=True,
    live_html_output_dir="data/readout_optimize",
    workflow_settings=workflow_settings,
)

optimizer = ReadoutAmplitudeSweepWorkflow(
    qubit_names=["q5", "q6", "q7"],
    profile=profile,
    task_manager=task_manager,
    settings=optimizer_settings,
)

optimizer.run()
print(optimizer.run_dir)
```

Do not call `optimizer.plot()` after a submit-only run; no acquired results
exist yet, so there is nothing to plot until collection finishes.

## Submit From The CLI

```bash
.venv/bin/python optimize_readout.py \
  --qubits q5 q6 q7 \
  --amplitudes 0.005 0.01 0.015 0.02 \
  --profile-branch main \
  --profile-name main \
  --method sweep \
  --low-priority-tasks \
  --submit-only
```

The command prints the pending run folder path.

## Collect Results

Use `scripts/load_optimizer_results.py` when you already have the optimizer run
folder name or path and want the script to check, download, analyze, and save
the results.

Edit the constants at the top of the script:

```python
RUN_KEY = "14-32-08_sweep_q5_q6_q7"  # folder name or full run folder path
OUTPUT_ROOT = Path("data") / "readout_optimize"
PROFILE_BRANCH = None  # None means use metadata profile_name
WAIT_FOR_RESULTS = False
```

Then run it:

```bash
.venv/bin/python optimize/readout/scripts/load_optimizer_results.py
```

With `WAIT_FOR_RESULTS = False`, the script checks the saved task IDs and exits
if any work is still queued or running. With `WAIT_FOR_RESULTS = True`, it blocks
for task-manager results and saves the normal optimizer artifacts.

## Pending Folder Contents

The pending folder has the same general shape as a normal optimizer output
folder, but it does not contain acquired result arrays until collection.

```text
data/readout_optimize/
  2026-07-07/
    14-32-08_sweep_q5_q6_q7/
      task_manifest.json
      metadata.json
      report.md
      profile.json
      results/
      iq_blobs/
      kernels/
      resonator/
```

Important files:

- `task_manifest.json`: source of truth for submitted task IDs.
- `metadata.json`: sweep, qubit, optimizer, workflow settings, and `run_key`.
- `profile.json`: profile snapshot copied or serialized at submission time.
- `results/`, `iq_blobs/`, `kernels/`, `resonator/`: initially empty placeholders.

Each task in `task_manifest.json` includes:

- `task_id`
- `task_key`
- `experiment_name`
- `node`
- `qubit_names`
- `low_priority`
- `amplitude`
- `sweep_index`
- `sweep_parameters`
- `result_status`

Example task key:

```text
sweep/0003/readout_amplitude=0.035/kernels/q5+q6+q7/00
```

## Python Collection

Later, recreate the optimizer with the same profile and task manager context,
then point it at the pending folder:

```python
from pathlib import Path

pending_run_dir = Path("data/readout_optimize/2026-07-07/14-32-08_sweep_q5_q6_q7")

profile = load_profile("main")
task_manager = load_task_manager()

workflow_settings = ReadoutFidelityWorkflowSettings(
    profile_name="main",
    do_emulation=False,
    run_resonator=False,
    run_kernels=True,
    run_iq_blobs=True,
    do_plotting=False,
    show_handler_output=False,
    low_priority_tasks=True,
    states=["g", "e"],
    reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
)

optimizer_settings = ReadoutAmplitudeSweepSettings(
    amplitudes=[],
    method=ReadoutScanMethod.SWEEP,
    workflow_settings=workflow_settings,
)

optimizer = ReadoutAmplitudeSweepWorkflow(
    qubit_names=["q5", "q6", "q7"],
    profile=profile,
    task_manager=task_manager,
    settings=optimizer_settings,
)
```

Check whether the tasks are ready without blocking:

```python
summary = optimizer.check_submitted_results(pending_run_dir)
print(summary["message"])
print(summary["counts"])
```

Collect once every task is ready:

```python
optimizer.collect_submitted_results(pending_run_dir)
```

## Notes

- Keep the pending folder. It is the durable link between sweep points and task IDs.
- Use the same `qubit_names`, workflow nodes, `states`, reset settings, profile name,
  and task manager backend when collecting.
- Use `low_priority_tasks=True` when you want qigeon to mark submitted jobs as
  low priority. This flag is stored in `task_manifest.json` for each task.
- Submit-only does not run adaptive optimizer methods because those need completed
  results before choosing the next amplitude.
