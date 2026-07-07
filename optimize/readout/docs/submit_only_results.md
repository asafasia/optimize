# Submit Readout Sweeps Now, Collect Results Later

This flow is for large readout amplitude sweeps where you want to submit all
experiments to the task manager, stop waiting in the local process, and collect
the acquired results later from saved task IDs.

Submit-only mode works only with:

- `ReadoutScanMethod.SWEEP`
- a real task manager, not `do_emulation=True`

It creates the normal dated optimizer run folder immediately, but the folder is
metadata-only until results are collected.

## Submit All Sweep Experiments

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
    qubit_names=["q5"],
    profile=profile,
    task_manager=task_manager,
    settings=optimizer_settings,
)

optimizer.run()

print(optimizer.run_dir)
```

The printed `run_dir` is the folder you need later for collection.
Do not call `optimizer.plot()` after a submit-only run; no acquired results exist
yet, so there is nothing to plot until collection finishes.

## Submit From The CLI

```bash
.venv/bin/python optimize_readout.py \
  --qubits q5 \
  --amplitudes 0.005 0.01 0.015 0.02 \
  --profile-branch main \
  --profile-name main \
  --method sweep \
  --skip-kernels \
  --low-priority-tasks \
  --submit-only
```

The command prints the pending run folder path.

## Load Results From A Pending Run Folder

Use `scripts/load_optimizer_results.py` when you already have the optimizer run
folder name or path and want the script to check, download, analyze, and save
the results.

Edit the constants at the top of the script:

```python
RUN_KEY = "14-32-08_sweep_q5"  # folder name or full run folder path
OUTPUT_ROOT = Path("data") / "readout_optimize"
PROFILE_BRANCH = None  # None means use metadata profile_name
WAIT_FOR_RESULTS = False
```

Then run it:

```bash
.venv/bin/python optimize/readout/scripts/load_optimizer_results.py
```

With `WAIT_FOR_RESULTS = False`, it checks every saved `task_id`, prints pending
`task_key`s if anything is still queued/running, and exits without blocking. If
every task is complete, it downloads, analyzes, and writes the normal optimizer
outputs. Set `WAIT_FOR_RESULTS = True` to block until qigeon returns unfinished
results.

## Pending Folder Contents

The pending folder has the same general shape as a normal optimizer output
folder, but it does not contain acquired result arrays yet.

```text
data/readout_optimize/
  2026-07-07/
    14-32-08_sweep_q5/
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

The value to paste into `RUN_KEY` is stored in `metadata.json`:

```json
{
  "run_key": "14-32-08_sweep_q5"
}
```

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
sweep/0003/readout_amplitude=0.035/kernels/q5/00
```

This makes large runs easier to debug because task IDs remain tied to amplitude,
experiment node, and qubit names.

## Collect Results Later

Later, recreate the optimizer with the same profile and task manager context,
then point it at the pending folder:

```python
from pathlib import Path

pending_run_dir = Path("data/readout_optimize/2026-07-07/14-32-08_sweep_q5")

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
    qubit_names=["q5"],
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

If any tasks are still queued or running, the message tells you to wait before
collecting. The check updates `task_manifest.json` with `task_status`,
`checked_at`, and `result_status` for every saved task.

You can also use collection in non-blocking mode:

```python
summary = optimizer.collect_submitted_results(pending_run_dir, wait=False)
```

When every task is complete, collect the acquired results:

```python
optimizer.collect_submitted_results(pending_run_dir)
```

Collection does the normal result processing:

- loads `task_manifest.json`
- waits for each saved `task_id`
- rebuilds handlers for each workflow node
- analyzes returned LabOneQ results
- writes the usual optimizer artifacts into the same folder
- marks the manifest `run_status` as `complete`

## Notes

- Keep the pending folder. It is the durable link between sweep points and task IDs.
- Use the same `qubit_names`, workflow nodes, `states`, reset settings, profile name,
  and task manager backend when collecting.
- Use `low_priority_tasks=True` when you want qigeon to mark submitted jobs as
  low priority. This flag is stored in `task_manifest.json` for each task.
- Use `check_submitted_results(...)` before collection when you do not want the
  process to block on unfinished qigeon tasks.
- Submit-only does not run adaptive optimizer methods because those need completed
  results before choosing the next amplitude.
- If the task backend still has pending tasks, collection will wait according to
  the existing task-manager behavior.
