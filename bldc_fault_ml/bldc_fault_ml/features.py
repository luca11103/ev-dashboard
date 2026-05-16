from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .signal import bandpass_fft, frequency_domain_features, safe_array, slope, time_domain_features


SCALAR_COLUMNS = (
    "temp_c",
    "rpm",
    "battery_voltage_v",
    "battery_current_a",
    "phase_u_back_emf_v",
    "phase_v_back_emf_v",
    "phase_w_back_emf_v",
    "phase_u_current_a",
    "phase_v_current_a",
    "phase_w_current_a",
)

VIBRATION_COLUMNS = ("vibration_x_g", "vibration_y_g", "vibration_z_g")

DEFAULT_BANDS = (
    (2.0, 10.0),
    (10.0, 30.0),
    (30.0, 60.0),
    (60.0, 120.0),
    (120.0, 200.0),
    (200.0, 400.0),
)


@dataclass
class FeatureFrame:
    x: np.ndarray
    feature_names: list[str]
    labels: list[str]
    rul_minutes: np.ndarray
    groups: list[str]
    raw_windows: list[dict[str, np.ndarray]]


def _arr(window: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in window:
        return np.zeros(1, dtype=float)
    return safe_array(window[name])


def _scalar_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = safe_array(values)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_range": float(np.ptp(arr)),
        f"{prefix}_slope": slope(arr),
    }


def _phase_imbalance(prefix: str, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> dict[str, float]:
    phases = np.vstack([safe_array(u), safe_array(v), safe_array(w)])
    means = np.mean(phases, axis=1)
    abs_mean = float(np.mean(np.abs(means)))
    spread = float(np.max(means) - np.min(means))
    rms = np.sqrt(np.mean(phases**2, axis=1))
    rms_mean = float(np.mean(rms))
    rms_spread = float(np.max(rms) - np.min(rms))
    return {
        f"{prefix}_mean_u": float(means[0]),
        f"{prefix}_mean_v": float(means[1]),
        f"{prefix}_mean_w": float(means[2]),
        f"{prefix}_spread": spread,
        f"{prefix}_imbalance_ratio": spread / max(abs_mean, 1e-9),
        f"{prefix}_rms_spread": rms_spread,
        f"{prefix}_rms_imbalance_ratio": rms_spread / max(rms_mean, 1e-9),
        f"{prefix}_uv_delta": float(means[0] - means[1]),
        f"{prefix}_vw_delta": float(means[1] - means[2]),
        f"{prefix}_wu_delta": float(means[2] - means[0]),
    }


def extract_window_features(
    window: Mapping[str, np.ndarray],
    sample_rate_hz: float,
    bandpass_low_hz: float = 5.0,
    bandpass_high_hz: float = 180.0,
) -> dict[str, float]:
    features: dict[str, float] = {}

    for column in SCALAR_COLUMNS:
        features.update(_scalar_stats(column, _arr(window, column)))

    phase_current_u = _arr(window, "phase_u_current_a")
    phase_current_v = _arr(window, "phase_v_current_a")
    phase_current_w = _arr(window, "phase_w_current_a")
    phase_bemf_u = _arr(window, "phase_u_back_emf_v")
    phase_bemf_v = _arr(window, "phase_v_back_emf_v")
    phase_bemf_w = _arr(window, "phase_w_back_emf_v")
    features.update(_phase_imbalance("phase_current", phase_current_u, phase_current_v, phase_current_w))
    features.update(_phase_imbalance("back_emf", phase_bemf_u, phase_bemf_v, phase_bemf_w))

    voltage = _arr(window, "battery_voltage_v")
    current = _arr(window, "battery_current_a")
    rpm = _arr(window, "rpm")
    temp = _arr(window, "temp_c")
    power = voltage * current
    features.update(_scalar_stats("power_w", power))
    features["current_per_krpm"] = float(np.mean(current) / max(float(np.mean(rpm)) / 1000.0, 1e-9))
    features["temp_per_kw"] = float(np.mean(temp) / max(float(np.mean(power)) / 1000.0, 1e-9))

    filtered_axes = []
    rpm_mean = float(np.mean(rpm))
    high = min(bandpass_high_hz, max(sample_rate_hz / 2.0 - 1.0, bandpass_low_hz))
    bands = tuple((low, min(high_band, sample_rate_hz / 2.0)) for low, high_band in DEFAULT_BANDS if low < sample_rate_hz / 2.0)

    for column in VIBRATION_COLUMNS:
        axis = _arr(window, column)
        filtered = bandpass_fft(axis, sample_rate_hz, bandpass_low_hz, high)
        filtered_axes.append(filtered)
        features.update(time_domain_features(column, axis))
        features.update(time_domain_features(f"{column}_bp", filtered))
        features.update(frequency_domain_features(column, filtered, sample_rate_hz, bands, rpm=rpm_mean))

    length = min(axis.size for axis in filtered_axes)
    xyz = np.vstack([axis[:length] for axis in filtered_axes])
    magnitude = np.sqrt(np.sum(xyz**2, axis=0))
    features.update(time_domain_features("vibration_mag_bp", magnitude))
    features.update(frequency_domain_features("vibration_mag_bp", magnitude, sample_rate_hz, bands, rpm=rpm_mean))

    features["vibration_xyz_rms_sum"] = (
        features["vibration_x_g_bp_rms"] + features["vibration_y_g_bp_rms"] + features["vibration_z_g_bp_rms"]
    )
    features["vibration_xy_ratio"] = features["vibration_x_g_bp_rms"] / max(features["vibration_y_g_bp_rms"], 1e-9)
    features["vibration_xz_ratio"] = features["vibration_x_g_bp_rms"] / max(features["vibration_z_g_bp_rms"], 1e-9)
    features["electrical_mechanical_stress"] = (
        features["phase_current_rms_imbalance_ratio"]
        * features["back_emf_rms_imbalance_ratio"]
        * max(features["vibration_mag_bp_rms"], 1e-9)
    )

    return {key: float(value) if np.isfinite(value) else 0.0 for key, value in features.items()}


def build_feature_frame(
    windows: list[dict[str, np.ndarray]],
    labels: list[str],
    rul_minutes: list[float],
    groups: list[str],
    sample_rate_hz: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
) -> FeatureFrame:
    rows = [
        extract_window_features(window, sample_rate_hz, bandpass_low_hz, bandpass_high_hz)
        for window in windows
    ]
    feature_names = sorted({name for row in rows for name in row})
    matrix = np.array([[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=float)
    return FeatureFrame(
        x=matrix,
        feature_names=feature_names,
        labels=labels,
        rul_minutes=np.asarray(rul_minutes, dtype=float),
        groups=groups,
        raw_windows=windows,
    )
