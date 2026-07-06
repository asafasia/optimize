"""Robust decay fit helpers for local thermal-population experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


SECONDS_TO_MICROSECONDS = 1e6


def fit_exponential_decay(
    x_time_data_points: NDArray[np.floating],
    y_values: NDArray[np.floating],
) -> dict[str, float | NDArray[np.floating]]:
    return _fit_decay_with_power(
        x_time_data_points=x_time_data_points,
        y_values=y_values,
        exponent_power=1.0,
    )


def fit_gaussian_decay(
    x_time_data_points: NDArray[np.floating],
    y_values: NDArray[np.floating],
) -> dict[str, float | NDArray[np.floating]]:
    return _fit_decay_with_power(
        x_time_data_points=x_time_data_points,
        y_values=y_values,
        exponent_power=2.0,
    )


def _fit_decay_with_power(
    x_time_data_points: NDArray[np.floating],
    y_values: NDArray[np.floating],
    exponent_power: float,
) -> dict[str, float | NDArray[np.floating]]:
    x_full = np.asarray(x_time_data_points, dtype=float)
    y_full = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_full) & np.isfinite(y_full)
    x = x_full[valid]
    y = y_full[valid]

    if x.size < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0 or exponent_power <= 0:
        return _empty_decay_fit()

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    tail_count = max(5, int(np.ceil(0.35 * y_sorted.size)))
    tail_y = y_sorted[-tail_count:]

    y_span = float(np.ptp(y_sorted))
    epsilon = max(y_span * 1e-9, 1e-15)
    tail_baseline = float(np.median(tail_y))
    tail_noise = _robust_scale(tail_y)
    noise_floor = max(tail_noise, y_span * 0.015, epsilon)
    signal_threshold = max(3.0 * noise_floor, y_span * 0.04)

    offset_radius = max(5.0 * noise_floor, y_span * 0.18)
    offset_candidates = np.linspace(
        tail_baseline - offset_radius,
        tail_baseline + 0.5 * offset_radius,
        240,
    )

    best: dict[str, float | NDArray[np.floating]] | None = None
    for offset in offset_candidates:
        fit_mask = y > offset + signal_threshold
        if np.count_nonzero(fit_mask) < 3 or np.ptp(x[fit_mask]) <= 0:
            continue

        shifted_fit_y = y[fit_mask] - offset
        if np.any(shifted_fit_y <= 0):
            continue

        weights = np.sqrt(shifted_fit_y / np.max(shifted_fit_y))
        x_fit_powered = np.power(np.clip(x[fit_mask], 0.0, None), exponent_power)
        if np.ptp(x_fit_powered) <= 0:
            continue

        slope, intercept = np.polyfit(
            x_fit_powered,
            np.log(shifted_fit_y),
            1,
            w=weights,
        )
        slope_stderr = _weighted_linear_slope_stderr(
            x_values=x_fit_powered,
            y_values=np.log(shifted_fit_y),
            weights=weights,
            slope=float(slope),
            intercept=float(intercept),
        )
        if slope >= 0:
            continue

        tau = np.power(-1.0 / slope, 1.0 / exponent_power)
        amplitude = float(np.exp(intercept))
        if not np.isfinite(tau) or not np.isfinite(amplitude) or amplitude <= 0:
            continue

        fitted = offset + amplitude * _decay_envelope(x, tau, exponent_power)
        fitted_full = np.full_like(x_full, np.nan, dtype=float)
        fitted_full[valid] = fitted
        residual = y - fitted
        score = _weighted_huber_score(
            residual=residual,
            y_values=y,
            offset=offset,
            threshold=signal_threshold,
            noise_floor=noise_floor,
        )
        tail_fit = offset + amplitude * _decay_envelope(
            x_sorted[-tail_count:],
            tau,
            exponent_power,
        )
        tail_penalty = float(((np.median(tail_fit) - tail_baseline) / noise_floor) ** 2)
        score += 0.2 * tail_penalty

        total = float(np.sum((y - np.mean(y)) ** 2))
        rss = float(np.sum(residual**2))
        r2 = 1.0 - rss / total if total > 0 else 0.0

        if best is None or score < float(best["score"]):
            best = {
                "fitted": fitted_full,
                "tau": float(tau),
                "tau_stderr": _tau_stderr_from_slope_stderr(
                    tau=float(tau),
                    slope=float(slope),
                    slope_stderr=slope_stderr,
                    exponent_power=exponent_power,
                ),
                "amplitude": amplitude,
                "offset": float(offset),
                "r2": float(r2),
                "rss": rss,
                "score": float(score),
                "fit_points": int(np.count_nonzero(fit_mask)),
            }

    if best is None:
        return _empty_decay_fit()
    return best


def _decay_envelope(
    x_values: NDArray[np.floating],
    tau: float,
    exponent_power: float,
) -> NDArray[np.floating]:
    return np.exp(-np.power(np.clip(x_values, 0.0, None) / tau, exponent_power))


def _weighted_linear_slope_stderr(
    x_values: NDArray[np.floating],
    y_values: NDArray[np.floating],
    weights: NDArray[np.floating],
    slope: float,
    intercept: float,
) -> float:
    if x_values.size < 3:
        return float("nan")

    design = np.column_stack((x_values, np.ones_like(x_values)))
    weighted_design = design * weights[:, None]
    weighted_residual = weights * (y_values - (slope * x_values + intercept))
    degrees_of_freedom = x_values.size - 2
    if degrees_of_freedom <= 0:
        return float("nan")

    try:
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
    except np.linalg.LinAlgError:
        return float("nan")

    residual_variance = float(np.sum(weighted_residual**2) / degrees_of_freedom)
    slope_variance = float(covariance[0, 0] * residual_variance)
    if not np.isfinite(slope_variance) or slope_variance < 0:
        return float("nan")
    return float(np.sqrt(slope_variance))


def _tau_stderr_from_slope_stderr(
    tau: float,
    slope: float,
    slope_stderr: float,
    exponent_power: float,
) -> float:
    if (
        not np.isfinite(tau)
        or tau <= 0
        or not np.isfinite(slope)
        or slope >= 0
        or not np.isfinite(slope_stderr)
        or slope_stderr < 0
        or exponent_power <= 0
    ):
        return float("nan")
    return float(tau * slope_stderr / (exponent_power * abs(slope)))


def _robust_scale(values: NDArray[np.floating]) -> float:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


def _weighted_huber_score(
    residual: NDArray[np.floating],
    y_values: NDArray[np.floating],
    offset: float,
    threshold: float,
    noise_floor: float,
) -> float:
    normalized_residual = residual / noise_floor
    abs_residual = np.abs(normalized_residual)
    huber_loss = np.where(
        abs_residual <= 1.5,
        0.5 * normalized_residual**2,
        1.5 * abs_residual - 1.125,
    )
    signal_weight = np.where(y_values > offset + threshold, 1.0, 0.25)
    return float(np.mean(signal_weight * huber_loss))


def _empty_decay_fit() -> dict[str, float | NDArray[np.floating]]:
    return {
        "fitted": np.array([]),
        "tau": 0.0,
        "tau_stderr": float("nan"),
        "amplitude": 0.0,
        "offset": 0.0,
        "r2": 0.0,
        "rss": float("inf"),
        "score": float("inf"),
        "fit_points": 0,
    }
