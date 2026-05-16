from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-12


def safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.zeros(1, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return np.zeros_like(arr, dtype=float)
    median = float(np.nanmedian(arr))
    return np.where(np.isnan(arr), median, arr)


def bandpass_fft(values: Iterable[float], sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    signal = safe_array(values)
    n = signal.size
    if n < 4 or sample_rate_hz <= 0:
        return signal - float(np.mean(signal))

    nyquist = sample_rate_hz / 2.0
    low = max(0.0, min(low_hz, nyquist))
    high = max(low, min(high_hz, nyquist))
    centered = signal - float(np.mean(signal))
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mask = (freqs >= low) & (freqs <= high)
    spectrum = spectrum * mask
    return np.fft.irfft(spectrum, n=n)


def fft_spectrum(values: Iterable[float], sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    signal = safe_array(values)
    n = signal.size
    if n < 4 or sample_rate_hz <= 0:
        return np.zeros(1), np.zeros(1)

    window = np.hanning(n)
    centered = signal - float(np.mean(signal))
    spectrum = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    amp = (2.0 / max(np.sum(window), EPS)) * np.abs(spectrum)
    return freqs, amp


def slope(values: Iterable[float]) -> float:
    arr = safe_array(values)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    x = x - float(np.mean(x))
    y = arr - float(np.mean(arr))
    denom = float(np.sum(x * x))
    if denom <= EPS:
        return 0.0
    return float(np.sum(x * y) / denom)


def skewness(values: Iterable[float]) -> float:
    arr = safe_array(values)
    std = float(np.std(arr))
    if std <= EPS:
        return 0.0
    z = (arr - float(np.mean(arr))) / std
    return float(np.mean(z**3))


def kurtosis(values: Iterable[float]) -> float:
    arr = safe_array(values)
    std = float(np.std(arr))
    if std <= EPS:
        return 0.0
    z = (arr - float(np.mean(arr))) / std
    return float(np.mean(z**4))


def time_domain_features(prefix: str, values: Iterable[float]) -> dict[str, float]:
    arr = safe_array(values)
    abs_arr = np.abs(arr)
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(abs_arr))
    mean_abs = float(np.mean(abs_arr))
    std = float(np.std(arr))
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_ptp": float(np.ptp(arr)),
        f"{prefix}_crest_factor": peak / max(rms, EPS),
        f"{prefix}_impulse_factor": peak / max(mean_abs, EPS),
        f"{prefix}_shape_factor": rms / max(mean_abs, EPS),
        f"{prefix}_skew": skewness(arr),
        f"{prefix}_kurtosis": kurtosis(arr),
        f"{prefix}_slope": slope(arr),
    }


def frequency_domain_features(
    prefix: str,
    values: Iterable[float],
    sample_rate_hz: float,
    bands: tuple[tuple[float, float], ...],
    rpm: float | None = None,
) -> dict[str, float]:
    freqs, amp = fft_spectrum(values, sample_rate_hz)
    power = amp**2
    total_power = float(np.sum(power)) + EPS
    non_dc = freqs > 0.0

    if np.any(non_dc):
        idx = int(np.argmax(amp[non_dc]))
        non_dc_freqs = freqs[non_dc]
        non_dc_amp = amp[non_dc]
        dominant_freq = float(non_dc_freqs[idx])
        dominant_amp = float(non_dc_amp[idx])
    else:
        dominant_freq = 0.0
        dominant_amp = 0.0

    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    rolloff_threshold = 0.85 * total_power
    cumsum = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumsum, rolloff_threshold, side="left"))
    rolloff_idx = min(rolloff_idx, freqs.size - 1)

    result = {
        f"{prefix}_fft_dominant_hz": dominant_freq,
        f"{prefix}_fft_dominant_amp": dominant_amp,
        f"{prefix}_fft_centroid_hz": centroid,
        f"{prefix}_fft_bandwidth_hz": bandwidth,
        f"{prefix}_fft_rolloff85_hz": float(freqs[rolloff_idx]),
        f"{prefix}_fft_total_power": float(total_power),
    }

    for low, high in bands:
        if high <= low:
            band_power = 0.0
        else:
            mask = (freqs >= low) & (freqs < high)
            band_power = float(np.sum(power[mask]))
        tag = f"{int(low)}_{int(high)}hz"
        result[f"{prefix}_bandpower_{tag}"] = band_power
        result[f"{prefix}_bandpower_ratio_{tag}"] = band_power / total_power

    if rpm and rpm > 0:
        shaft_hz = rpm / 60.0
        harmonic_power = 0.0
        for harmonic in (1, 2, 3, 4):
            target = harmonic * shaft_hz
            width = max(1.5, target * 0.04)
            mask = (freqs >= target - width) & (freqs <= target + width)
            harmonic_power += float(np.sum(power[mask]))
        result[f"{prefix}_shaft_harmonic_power"] = harmonic_power
        result[f"{prefix}_shaft_harmonic_ratio"] = harmonic_power / total_power
        result[f"{prefix}_dominant_to_shaft_ratio"] = dominant_freq / max(shaft_hz, EPS)
    else:
        result[f"{prefix}_shaft_harmonic_power"] = 0.0
        result[f"{prefix}_shaft_harmonic_ratio"] = 0.0
        result[f"{prefix}_dominant_to_shaft_ratio"] = 0.0

    if math.isnan(result[f"{prefix}_fft_total_power"]):
        result[f"{prefix}_fft_total_power"] = 0.0
    return result
