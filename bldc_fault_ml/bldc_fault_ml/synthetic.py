from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


FAULT_CLASSES = (
    "normal",
    "bearing_wear",
    "rotor_imbalance",
    "misalignment",
    "phase_loss",
    "winding_short",
    "back_emf_anomaly",
    "demagnetization",
)

DEFAULT_SYNTHETIC_PROFILE = "ev_72v"

SYNTHETIC_PROFILES = {
    "ev_72v": {
        "title": "EV-scale synthetic BLDC demo",
        "battery_voltage_v": "approximately 72 V",
        "battery_current_a": "approximately 18-110 A",
        "rpm": "1800-5200 RPM",
        "temperature_c": "load and fault dependent",
        "note": "Original broad demo envelope.",
    },
    "a2212_12v_2a": {
        "title": "A2212 instrumented 12 V / 2 A test envelope",
        "battery_voltage_v": "10.4-12.6 V synthetic bus samples",
        "battery_current_a": "0.05-2.00 A source-limited samples",
        "rpm": "1200-7200 RPM under the limited-current test envelope",
        "temperature_c": "30-55 C",
        "note": "Back EMF and signed phase-current waveforms assume calibrated per-phase sensing.",
    },
}


RAW_COLUMNS = (
    "timestamp_s",
    "motor_id",
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
    "vibration_x_g",
    "vibration_y_g",
    "vibration_z_g",
    "fault_type",
    "rul_minutes",
)


def _sin(t: np.ndarray, hz: float, phase: float = 0.0) -> np.ndarray:
    return np.sin((2.0 * np.pi * hz * t) + phase)


def _rul_minutes(label: str, severity: float, rng: np.random.Generator) -> float:
    if label == "normal":
        return float(rng.uniform(5000.0, 12000.0))
    base = {
        "bearing_wear": 900.0,
        "rotor_imbalance": 1500.0,
        "misalignment": 1800.0,
        "phase_loss": 250.0,
        "winding_short": 180.0,
        "back_emf_anomaly": 500.0,
        "demagnetization": 650.0,
    }[label]
    noise = rng.normal(0.0, base * 0.04)
    return float(max(5.0, base * ((1.0 - severity) ** 1.65) + noise))


def simulate_window(
    label: str,
    severity: float,
    sample_rate_hz: float,
    window_seconds: float,
    rng: np.random.Generator,
    window_index: int,
    profile: str = DEFAULT_SYNTHETIC_PROFILE,
) -> tuple[dict[str, np.ndarray], float, str]:
    if profile == "a2212_12v_2a":
        return _simulate_a2212_window(label, severity, sample_rate_hz, window_seconds, rng, window_index)
    if profile != "ev_72v":
        raise ValueError(f"Unknown synthetic profile: {profile}")

    n = int(sample_rate_hz * window_seconds)
    t = np.arange(n, dtype=float) / sample_rate_hz
    load = rng.uniform(0.25, 1.0)
    rpm_base = rng.uniform(1800.0, 5200.0)
    shaft_hz = rpm_base / 60.0
    pole_pairs = rng.choice([4.0, 5.0, 7.0])
    electrical_hz = shaft_hz * pole_pairs

    speed_ripple = 12.0 * _sin(t, 0.8, rng.uniform(0.0, np.pi))
    rpm_fault_ripple = np.zeros_like(t)
    if label in ("misalignment", "rotor_imbalance"):
        rpm_fault_ripple += severity * 22.0 * _sin(t, shaft_hz)
    if label in ("phase_loss", "back_emf_anomaly"):
        rpm_fault_ripple += severity * 55.0 * _sin(t, 2.0 * shaft_hz)
    if label == "demagnetization":
        rpm_fault_ripple += severity * 40.0 * _sin(t, 3.0 * shaft_hz)
    rpm = rpm_base + speed_ripple + rpm_fault_ripple + rng.normal(0.0, 6.0, size=n)

    nominal_current = 18.0 + 58.0 * load + rng.normal(0.0, 0.7, size=n)
    current_fault = np.zeros_like(t)
    if label == "bearing_wear":
        current_fault += severity * 7.0
    if label == "phase_loss":
        current_fault += severity * 18.0
    if label == "winding_short":
        current_fault += severity * 32.0
    if label == "back_emf_anomaly":
        current_fault += severity * 10.0 * (1.0 + 0.2 * _sin(t, electrical_hz / 4.0))
    if label == "demagnetization":
        current_fault += severity * 14.0
    battery_current = nominal_current + current_fault

    voltage_nominal = 72.0 - (battery_current * 0.045) - (load * 1.8)
    if label in ("phase_loss", "winding_short"):
        voltage_nominal -= severity * 2.0
    battery_voltage = voltage_nominal + rng.normal(0.0, 0.12, size=n)

    temp_base = 35.0 + (battery_current * 0.35) + (load * 9.0)
    temp_fault = {
        "normal": 0.0,
        "bearing_wear": 8.0 * severity,
        "rotor_imbalance": 4.0 * severity,
        "misalignment": 5.0 * severity,
        "phase_loss": 13.0 * severity,
        "winding_short": 26.0 * severity,
        "back_emf_anomaly": 7.0 * severity,
        "demagnetization": 9.0 * severity,
    }[label]
    temp = temp_base + temp_fault + 0.25 * _sin(t, 0.2) + rng.normal(0.0, 0.08, size=n)

    phase_amp = 0.45 * battery_current + 4.0
    current_u = phase_amp * _sin(t, electrical_hz)
    current_v = phase_amp * _sin(t, electrical_hz, -2.0 * np.pi / 3.0)
    current_w = phase_amp * _sin(t, electrical_hz, 2.0 * np.pi / 3.0)

    bemf_amp = np.maximum(4.0, rpm / 100.0)
    bemf_u = bemf_amp * _sin(t, electrical_hz)
    bemf_v = bemf_amp * _sin(t, electrical_hz, -2.0 * np.pi / 3.0)
    bemf_w = bemf_amp * _sin(t, electrical_hz, 2.0 * np.pi / 3.0)

    if label == "phase_loss":
        current_u *= max(0.05, 1.0 - severity * 0.9)
        current_v *= 1.0 + severity * 0.26
        current_w *= 1.0 + severity * 0.18
        bemf_u *= 1.0 - severity * 0.55
    elif label == "winding_short":
        current_u *= 1.0 + severity * 0.65
        current_v *= 1.0 + severity * 0.18
        bemf_u *= 1.0 - severity * 0.35
        bemf_v *= 1.0 + severity * 0.1
    elif label == "back_emf_anomaly":
        distortion = severity * 0.22 * _sin(t, electrical_hz * 3.0)
        bemf_u = bemf_u + bemf_amp * distortion
        bemf_v = bemf_v - bemf_amp * severity * 0.18
        bemf_w = bemf_w * (1.0 - severity * 0.25)
    elif label == "demagnetization":
        collapse = max(0.28, 1.0 - severity * 0.58)
        bemf_u *= collapse
        bemf_v *= collapse * (1.0 - severity * 0.08)
        bemf_w *= collapse * (1.0 + severity * 0.05)

    phase_noise = rng.normal(0.0, 0.25 + severity * 0.1, size=(3, n))
    current_u += phase_noise[0]
    current_v += phase_noise[1]
    current_w += phase_noise[2]
    bemf_u += rng.normal(0.0, 0.18, size=n)
    bemf_v += rng.normal(0.0, 0.18, size=n)
    bemf_w += rng.normal(0.0, 0.18, size=n)

    vib_noise = rng.normal(0.0, 0.018 + 0.012 * load, size=(3, n))
    vib_x = 0.045 * _sin(t, shaft_hz) + 0.012 * _sin(t, electrical_hz) + vib_noise[0]
    vib_y = 0.035 * _sin(t, shaft_hz, 0.9) + 0.012 * _sin(t, electrical_hz) + vib_noise[1]
    vib_z = 0.025 * _sin(t, 2.0 * shaft_hz) + 0.008 * _sin(t, electrical_hz) + vib_noise[2]

    if label == "bearing_wear":
        bearing_low = min(70.0, max(8.0, sample_rate_hz / 4.0))
        bearing_high = max(bearing_low + 1.0, min(180.0, sample_rate_hz / 2.0 - 5.0))
        bearing_hz = rng.uniform(bearing_low, bearing_high)
        impulses = (np.sin(2.0 * np.pi * bearing_hz * t) > 0.985).astype(float)
        impulse_train = severity * 0.45 * impulses
        vib_x += severity * 0.18 * _sin(t, bearing_hz) + impulse_train
        vib_y += severity * 0.13 * _sin(t, bearing_hz * 1.5)
        vib_z += severity * 0.10 * _sin(t, bearing_hz * 0.5) + impulse_train * 0.5
    elif label == "rotor_imbalance":
        vib_x += severity * 0.55 * _sin(t, shaft_hz)
        vib_y += severity * 0.35 * _sin(t, shaft_hz, np.pi / 2.0)
    elif label == "misalignment":
        vib_x += severity * 0.20 * _sin(t, shaft_hz)
        vib_y += severity * 0.23 * _sin(t, 2.0 * shaft_hz)
        vib_z += severity * 0.42 * _sin(t, 2.0 * shaft_hz)
    elif label == "phase_loss":
        vib_x += severity * 0.18 * _sin(t, electrical_hz / 2.0)
        vib_y += severity * 0.14 * _sin(t, electrical_hz / 2.0, 1.2)
    elif label == "winding_short":
        vib_x += severity * 0.24 * _sin(t, electrical_hz)
        vib_y += severity * 0.16 * _sin(t, electrical_hz * 2.0)
        vib_z += severity * 0.11 * _sin(t, electrical_hz)
    elif label == "back_emf_anomaly":
        vib_x += severity * 0.13 * _sin(t, 3.0 * shaft_hz)
        vib_z += severity * 0.10 * _sin(t, electrical_hz / 3.0)
    elif label == "demagnetization":
        vib_x += severity * 0.12 * _sin(t, 2.0 * shaft_hz)
        vib_y += severity * 0.09 * _sin(t, electrical_hz / 3.0)

    rul = _rul_minutes(label, severity, rng)
    motor_id = f"motor_{int(window_index % 4) + 1}"
    offset = window_index * window_seconds
    window = {
        "timestamp_s": t + offset,
        "temp_c": temp,
        "rpm": rpm,
        "battery_voltage_v": battery_voltage,
        "battery_current_a": battery_current,
        "phase_u_back_emf_v": bemf_u,
        "phase_v_back_emf_v": bemf_v,
        "phase_w_back_emf_v": bemf_w,
        "phase_u_current_a": current_u,
        "phase_v_current_a": current_v,
        "phase_w_current_a": current_w,
        "vibration_x_g": vib_x,
        "vibration_y_g": vib_y,
        "vibration_z_g": vib_z,
    }
    return window, rul, motor_id


def _simulate_a2212_window(
    label: str,
    severity: float,
    sample_rate_hz: float,
    window_seconds: float,
    rng: np.random.Generator,
    window_index: int,
) -> tuple[dict[str, np.ndarray], float, str]:
    n = int(sample_rate_hz * window_seconds)
    t = np.arange(n, dtype=float) / sample_rate_hz
    load = rng.uniform(0.08, 1.0)
    rpm_base = rng.uniform(1200.0, 7200.0) * (1.0 - 0.08 * load)
    shaft_hz = rpm_base / 60.0
    electrical_hz = shaft_hz * 7.0

    rpm_fault = np.zeros_like(t)
    if label in ("misalignment", "rotor_imbalance"):
        rpm_fault += severity * 34.0 * _sin(t, shaft_hz)
    if label in ("phase_loss", "back_emf_anomaly"):
        rpm_fault += severity * 70.0 * _sin(t, 2.0 * shaft_hz)
    if label == "demagnetization":
        rpm_fault -= severity * 280.0
        rpm_fault += severity * 48.0 * _sin(t, 3.0 * shaft_hz)
    rpm = (
        rpm_base
        + 10.0 * _sin(t, 0.7, rng.uniform(0.0, np.pi))
        + rpm_fault
        + rng.normal(0.0, 4.0, size=n)
    )
    rpm = np.clip(rpm, 300.0, 7600.0)

    nominal_current = 0.18 + 1.42 * load + rng.normal(0.0, 0.018, size=n)
    current_fault = np.zeros_like(t)
    if label == "bearing_wear":
        current_fault += severity * 0.16
    if label == "phase_loss":
        current_fault += severity * 0.38
    if label == "winding_short":
        current_fault += severity * 0.62
    if label == "back_emf_anomaly":
        current_fault += severity * 0.22 * (1.0 + 0.16 * _sin(t, electrical_hz / 4.0))
    if label == "demagnetization":
        current_fault += severity * 0.28
    battery_current = np.clip(nominal_current + current_fault, 0.05, 2.0)

    battery_voltage = 12.25 - 0.16 * load - 0.42 * battery_current
    if label in ("phase_loss", "winding_short"):
        battery_voltage -= severity * 0.18
    battery_voltage += rng.normal(0.0, 0.025, size=n)
    battery_voltage = np.clip(battery_voltage, 10.4, 12.6)

    ambient = rng.uniform(29.5, 33.5)
    temp_fault = {
        "normal": 0.0,
        "bearing_wear": 2.8 * severity,
        "rotor_imbalance": 1.5 * severity,
        "misalignment": 2.0 * severity,
        "phase_loss": 4.8 * severity,
        "winding_short": 8.2 * severity,
        "back_emf_anomaly": 2.7 * severity,
        "demagnetization": 3.6 * severity,
    }[label]
    temp = ambient + (5.1 * load) + (3.8 * battery_current) + temp_fault
    temp += 0.18 * _sin(t, 0.18) + rng.normal(0.0, 0.045, size=n)
    temp = np.clip(temp, 30.0, 55.0)

    phase_amp = 0.09 + (0.82 * battery_current)
    current_u = phase_amp * _sin(t, electrical_hz)
    current_v = phase_amp * _sin(t, electrical_hz, -2.0 * np.pi / 3.0)
    current_w = phase_amp * _sin(t, electrical_hz, 2.0 * np.pi / 3.0)

    bemf_amp = np.clip(0.55 + (rpm * 0.0011), 0.55, np.minimum(10.8, battery_voltage * 0.94))
    bemf_u = bemf_amp * _sin(t, electrical_hz)
    bemf_v = bemf_amp * _sin(t, electrical_hz, -2.0 * np.pi / 3.0)
    bemf_w = bemf_amp * _sin(t, electrical_hz, 2.0 * np.pi / 3.0)

    if label == "phase_loss":
        current_u *= max(0.04, 1.0 - severity * 0.94)
        current_v *= 1.0 + severity * 0.18
        current_w *= 1.0 + severity * 0.12
        bemf_u *= 1.0 - severity * 0.62
    elif label == "winding_short":
        current_u *= 1.0 + severity * 0.72
        current_v *= 1.0 + severity * 0.12
        bemf_u *= 1.0 - severity * 0.44
        bemf_v *= 1.0 + severity * 0.08
    elif label == "back_emf_anomaly":
        distortion = severity * 0.28 * _sin(t, electrical_hz * 3.0)
        bemf_u = bemf_u + bemf_amp * distortion
        bemf_v = bemf_v - bemf_amp * severity * 0.22
        bemf_w *= 1.0 - severity * 0.30
    elif label == "demagnetization":
        collapse = max(0.24, 1.0 - severity * 0.63)
        bemf_u *= collapse
        bemf_v *= collapse * (1.0 - severity * 0.06)
        bemf_w *= collapse * (1.0 + severity * 0.05)

    phase_noise = rng.normal(0.0, 0.015 + severity * 0.012, size=(3, n))
    current_u = np.clip(current_u + phase_noise[0], -2.25, 2.25)
    current_v = np.clip(current_v + phase_noise[1], -2.25, 2.25)
    current_w = np.clip(current_w + phase_noise[2], -2.25, 2.25)
    bemf_u += rng.normal(0.0, 0.035, size=n)
    bemf_v += rng.normal(0.0, 0.035, size=n)
    bemf_w += rng.normal(0.0, 0.035, size=n)

    vib_noise = rng.normal(0.0, 0.009 + 0.008 * load, size=(3, n))
    vib_x = 0.018 * _sin(t, shaft_hz) + 0.005 * _sin(t, electrical_hz) + vib_noise[0]
    vib_y = 0.015 * _sin(t, shaft_hz, 0.8) + 0.004 * _sin(t, electrical_hz) + vib_noise[1]
    vib_z = 0.012 * _sin(t, 2.0 * shaft_hz) + 0.004 * _sin(t, electrical_hz) + vib_noise[2]

    if label == "bearing_wear":
        bearing_hz = rng.uniform(32.0, min(180.0, sample_rate_hz / 2.0 - 5.0))
        impulses = (np.sin(2.0 * np.pi * bearing_hz * t) > 0.985).astype(float)
        impulse_train = severity * 0.24 * impulses
        vib_x += severity * 0.11 * _sin(t, bearing_hz) + impulse_train
        vib_y += severity * 0.08 * _sin(t, bearing_hz * 1.35)
        vib_z += severity * 0.06 * _sin(t, bearing_hz * 0.6) + impulse_train * 0.45
    elif label == "rotor_imbalance":
        vib_x += severity * 0.34 * _sin(t, shaft_hz)
        vib_y += severity * 0.22 * _sin(t, shaft_hz, np.pi / 2.0)
    elif label == "misalignment":
        vib_x += severity * 0.11 * _sin(t, shaft_hz)
        vib_y += severity * 0.15 * _sin(t, 2.0 * shaft_hz)
        vib_z += severity * 0.24 * _sin(t, 2.0 * shaft_hz)
    elif label == "phase_loss":
        vib_x += severity * 0.12 * _sin(t, electrical_hz / 2.0)
        vib_y += severity * 0.09 * _sin(t, electrical_hz / 2.0, 1.1)
    elif label == "winding_short":
        vib_x += severity * 0.14 * _sin(t, electrical_hz)
        vib_y += severity * 0.10 * _sin(t, electrical_hz * 2.0)
        vib_z += severity * 0.07 * _sin(t, electrical_hz)
    elif label == "back_emf_anomaly":
        vib_x += severity * 0.08 * _sin(t, 3.0 * shaft_hz)
        vib_z += severity * 0.07 * _sin(t, electrical_hz / 3.0)
    elif label == "demagnetization":
        vib_x += severity * 0.07 * _sin(t, 2.0 * shaft_hz)
        vib_y += severity * 0.06 * _sin(t, electrical_hz / 3.0)

    rul = _rul_minutes(label, severity, rng)
    motor_id = f"a2212_rig_{int(window_index % 4) + 1}"
    offset = window_index * window_seconds
    window = {
        "timestamp_s": t + offset,
        "temp_c": temp,
        "rpm": rpm,
        "battery_voltage_v": battery_voltage,
        "battery_current_a": battery_current,
        "phase_u_back_emf_v": bemf_u,
        "phase_v_back_emf_v": bemf_v,
        "phase_w_back_emf_v": bemf_w,
        "phase_u_current_a": current_u,
        "phase_v_current_a": current_v,
        "phase_w_current_a": current_w,
        "vibration_x_g": vib_x,
        "vibration_y_g": vib_y,
        "vibration_z_g": vib_z,
    }
    return window, rul, motor_id


def generate_synthetic_windows(
    windows_per_class: int,
    sample_rate_hz: float,
    window_seconds: float,
    seed: int,
    profile: str = DEFAULT_SYNTHETIC_PROFILE,
) -> tuple[list[dict[str, np.ndarray]], list[str], list[float], list[str]]:
    rng = np.random.default_rng(seed)
    windows: list[dict[str, np.ndarray]] = []
    labels: list[str] = []
    rul: list[float] = []
    groups: list[str] = []
    index = 0

    for label in FAULT_CLASSES:
        for _ in range(windows_per_class):
            severity = 0.0 if label == "normal" else float(rng.uniform(0.18, 0.98))
            window, one_rul, motor_id = simulate_window(
                label,
                severity,
                sample_rate_hz,
                window_seconds,
                rng,
                index,
                profile=profile,
            )
            windows.append(window)
            labels.append(label)
            rul.append(one_rul)
            groups.append(motor_id)
            index += 1

    order = rng.permutation(len(windows))
    return (
        [windows[i] for i in order],
        [labels[i] for i in order],
        [rul[i] for i in order],
        [groups[i] for i in order],
    )


def write_raw_csv(
    path: Path,
    windows: list[dict[str, np.ndarray]],
    labels: list[str],
    rul_minutes: list[float],
    groups: list[str],
    max_windows: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(windows) if max_windows is None else min(max_windows, len(windows))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for idx in range(count):
            window = windows[idx]
            n = len(window["timestamp_s"])
            for sample_idx in range(n):
                writer.writerow(
                    {
                        "timestamp_s": f"{window['timestamp_s'][sample_idx]:.6f}",
                        "motor_id": groups[idx],
                        "temp_c": f"{window['temp_c'][sample_idx]:.5f}",
                        "rpm": f"{window['rpm'][sample_idx]:.5f}",
                        "battery_voltage_v": f"{window['battery_voltage_v'][sample_idx]:.5f}",
                        "battery_current_a": f"{window['battery_current_a'][sample_idx]:.5f}",
                        "phase_u_back_emf_v": f"{window['phase_u_back_emf_v'][sample_idx]:.5f}",
                        "phase_v_back_emf_v": f"{window['phase_v_back_emf_v'][sample_idx]:.5f}",
                        "phase_w_back_emf_v": f"{window['phase_w_back_emf_v'][sample_idx]:.5f}",
                        "phase_u_current_a": f"{window['phase_u_current_a'][sample_idx]:.5f}",
                        "phase_v_current_a": f"{window['phase_v_current_a'][sample_idx]:.5f}",
                        "phase_w_current_a": f"{window['phase_w_current_a'][sample_idx]:.5f}",
                        "vibration_x_g": f"{window['vibration_x_g'][sample_idx]:.7f}",
                        "vibration_y_g": f"{window['vibration_y_g'][sample_idx]:.7f}",
                        "vibration_z_g": f"{window['vibration_z_g'][sample_idx]:.7f}",
                        "fault_type": labels[idx],
                        "rul_minutes": f"{rul_minutes[idx]:.3f}",
                    }
                )
