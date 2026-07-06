# Resonator Photon Number Measurement Plan

This plan follows the method in `2310.16312v1.pdf`, "Measurement of small photon
numbers in circuit QED resonators", and maps it onto the local Hahn echo and CPMG
experiments in this folder.

## Goal

Measure small average photon populations in a readout resonator by using a qubit
as a dephasing sensor. Photons in the resonator shift the qubit frequency through
the dispersive interaction, so photon-number fluctuations add qubit dephasing.
The measured dephasing rate can be converted to an intracavity photon number
after calibrating or fitting against the paper's CPMG dephasing formulas.

## Physical Picture

The paper's sequence is:

```text
resonator drive reaches steady state before t = 0

pi/2 - [CPMG pi pulses separated by Delta t] - pi/2 - qubit readout

resonator drive is on continuously during the qubit sequence
```

For coherent photons, the added resonator tone is a constant microwave drive near
the resonator frequency. For thermal photons, the added drive is broadband noise
whose correlation time is short compared with the resonator lifetime. The first
implementation should use the coherent constant resonator pulse because it is
simpler and directly matches the requested next step.

## Current State

- `cpmg.py` runs local CPMG-like sequences with swept interpulse delay and fixed
  `N`.
- `hahn_echo.py` runs the single-refocusing-pulse echo sequence.
- `run_cpmg_sweep.py` runs many CPMG experiments for `N = [0, 1, 2, 4, 8, 16]`
  and saves each run into a new output directory.
- The current analysis fits the acquired signal versus sequence time and reports
  a fitted T2-like decay time.

The current experiments do not yet drive the resonator during the free evolution
window. That is the next required code step.

## Immediate Code Step

Add an optional continuous resonator drive to both CPMG and Hahn echo:

```text
resonator pre-drive / fill time
pi/2
continuous resonator drive during all CPMG or Hahn echo evolution
final pi/2
normal readout
```

Implementation details:

1. Add settings/handler arguments:
   - `resonator_drive_enabled: bool`
   - `resonator_drive_amplitude: float`
   - `resonator_drive_frequency: float | None`
   - `resonator_drive_phase: float = 0.0`
   - `resonator_fill_time: float`
   - `resonator_drive_duration_margin: float`
2. Use a const pulse on the resonator/measure signal, not the qubit drive signal.
   The signal should be `measure_{qubit_name}` or the equivalent readout-resonator
   signal used by the device setup.
3. Start the resonator drive before the first qubit `pi/2` by several resonator
   lifetimes, ideally at least `5 / kappa`.
4. Keep the resonator drive on during the full CPMG/Hahn echo sequence.
5. Turn it off before the final readout pulse, or keep it separate from the
   readout pulse so the measurement pulse remains unchanged.
6. Save drive metadata with every result:
   - drive amplitude
   - drive frequency and detuning from resonator
   - fill time
   - sequence type
   - `N`, `Delta t`, and total evolution time

For the first code pass, implement coherent drive only. Add thermal/noise drive
later after the coherent workflow is validated.

## Measurement Workflow

### 1. Characterize Baselines

Measure without added resonator drive:

- T1 for each qubit used.
- Hahn echo baseline.
- CPMG baseline over several `Delta t` values and `N` values.

Use these to estimate the non-photon dephasing floor:

```text
Gamma_phi_baseline = 1 / T2_baseline - 1 / (2 T1)
```

This baseline must be subtracted or included as a fit pedestal.

### 2. Validate Constant Resonator Drive

For each qubit/resonator pair:

1. Pick a low resonator drive amplitude.
2. Run Hahn echo with the drive on during the echo window.
3. Sweep drive amplitude and confirm that the extracted dephasing rate increases
   approximately linearly with resonator drive power in the small-photon regime.
4. Keep amplitudes small enough that the qubit coherence remains fit-quality
   usable over the sequence duration.

Hahn echo is the right first calibration because the paper uses low-frequency
spin-echo-style measurements to calibrate added photon number versus applied
drive power.

### 3. CPMG Spectral Measurement

Run CPMG with the resonator drive enabled:

- Fix an interpulse period `Delta t`.
- Sweep `N` to extract an exponential decay rate versus total time:

```text
C(t_cpmg) ~ exp(-Gamma_cpmg(Delta t) * t_cpmg)
```

- Repeat for many `Delta t` values.
- Plot `Gamma_cpmg` versus CPMG frequency:

```text
f_s = 1 / (2 Delta t)
```

This is the paper's main measurement: the shape of `Gamma_cpmg(Delta t)` tells
whether the photons look coherent or thermal, and the scale gives the photon
number.

### 4. Convert Dephasing to Photon Number

For the first-pass coherent-drive calibration, use the low-frequency echo limit:

```text
Gamma_phi_coh ~= 8 chi^2 n_coh / (kappa * (1 + (2 chi / kappa)^2))
```

For thermal photons in the low-frequency limit:

```text
Gamma_phi_th ~= 4 chi^2 n_th / (kappa * (1 + (2 chi / kappa)^2))
```

Here:

- `2 chi` is the qubit frequency shift per photon.
- `kappa` is the resonator energy decay rate.
- `Gamma_phi` is the pure dephasing rate after subtracting `1/(2 T1)` and the
  baseline pedestal.

For the full CPMG measurement, fit `Gamma_cpmg(Delta t)` to the non-Gaussian
formulas in the paper:

- Thermal photons: main text Eqs. 13-14.
- Coherent photons: main text Eqs. 16-17 for resonant drive.

Do not rely only on the Gaussian filter-function approximation unless
`|2 chi| << kappa`; the paper emphasizes that this can be inaccurate in the
moderate/strong dispersive regime.

## Analysis Improvements Needed

The paper extracts coherence by scanning the final `pi/2` phase and fitting the
qubit response versus phase. The current local scripts use a single final
`pi/2` phase and fit the measured signal directly. For robust photon-number
metrology, add:

1. Final `pi/2` phase sweep, ideally six phases over `0..2pi`.
2. Sinusoidal fit at each sequence time to extract visibility/coherence `C`.
3. Exponential fit of `C` versus evolution time to extract `Gamma`.
4. Fit pedestal and T1 correction.
5. Per-qubit summary plots:
   - coherence decay for each `Delta t`
   - extracted `Gamma_cpmg` versus `f_s`
   - inferred photon number versus resonator drive power

## Proposed Implementation Order

1. Add continuous coherent resonator drive support to `hahn_echo.py`.
2. Add the same drive support to `cpmg.py`.
3. Add a small runner for Hahn echo versus resonator drive amplitude.
4. Extend `run_cpmg_sweep.py` with drive arguments and save drive metadata.
5. Add final `pi/2` phase sweep and coherence extraction.
6. Add analysis functions for:
   - `Gamma_phi` extraction
   - low-frequency photon-number estimate
   - full `Gamma_cpmg(Delta t)` model fit
7. Only after coherent-drive calibration works, add thermal/noise drive support.

## Acceptance Criteria

- With resonator drive disabled, CPMG/Hahn echo results match the current
  baseline behavior.
- With a weak coherent resonator drive enabled, echo/CPMG dephasing increases
  monotonically with drive power.
- Run artifacts include all pulse and drive parameters needed to reproduce the
  inferred photon number.
- The analysis produces a photon-number estimate with explicit assumptions:
  coherent vs thermal, `chi`, `kappa`, T1 correction, and baseline dephasing.

