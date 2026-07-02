# Measure Resonator Thermal Population

Goal: run a modified T1-style experiment that can choose the prepared initial
state before the decay wait. The first useful states are `e` and `g`; later the
same interface should support `f`.

## Recommended Code Strategy

Keep this as a local workbench experiment first. Do not edit `../qratena`
until the sequence and analysis are stable.

Suggested files:

- `modified_t1.py`: local experiment and handler classes.
- `run_modified_t1.py`: executable script that loads the profile, compiles,
  submits, analyzes, and plots.
- `analysis.py`: optional thermal-population-specific analysis once the raw
  T1 curves are working.

## Experiment Shape

The existing `qratena.experiments.t1.T1` sequence is:

1. reset
2. excite qubit with pi pulse
3. wait for swept decay time
4. readout

For this experiment, make step 2 conditional:

- `initial_state="e"`: apply `ge` pi pulse, same as normal T1.
- `initial_state="g"`: apply no excitation pulse, then wait and readout.
- `initial_state="f"`: later, apply `ge` pi pulse followed by `ef` pi pulse.

This keeps one handler API:

```python
handler = ModifiedT1Handler(
    qubit_names=["q3"],
    initial_state="g",
    decay_time_sweep_interval_length=200e-6,
    num_sweep_points=101,
    settings=settings,
    configuration_params=profile,
)
```

## Why This Structure

Starting from `g` is not a normal T1 decay; it measures thermal excitation and
readout drift/background over the same wait sweep. Keeping the sequence identical
except for the preparation pulse makes the `g` and `e` curves directly
comparable.

For thermal population, run both states back-to-back using the same reset,
shots, sweep, acquisition type, and readout calibration:

- `e` curve: normal decay reference.
- `g` curve: thermal/background curve.
- optional interleaved version later: sweep over both `initial_state` and decay
  time inside one compiled experiment to reduce slow drift.

Start with separate compiled runs because it is easier to debug and uses the
existing `T1Handler` analysis almost unchanged. Move to an interleaved 2D
sequence only after the basic signal is validated.

