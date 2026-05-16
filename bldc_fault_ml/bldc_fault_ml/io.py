from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "time_s", "time", "t"),
    "motor_id": ("motor_id", "motorId", "motor", "drive"),
    "temp_c": ("temp_c", "temp", "temperature", "motorTemp", "motor_temp", "temperature_c"),
    "rpm": ("rpm", "speedRpm", "speed", "motor_rpm"),
    "battery_voltage_v": ("battery_voltage_v", "inputVoltage", "voltage", "busVoltage", "batteryVoltage"),
    "battery_current_a": ("battery_current_a", "currentDraw", "current", "totalCurrent", "batteryCurrent"),
    "phase_u_back_emf_v": ("phase_u_back_emf_v", "phaseUBackEmf", "backEmfU", "bemf_u", "bemfU"),
    "phase_v_back_emf_v": ("phase_v_back_emf_v", "phaseVBackEmf", "backEmfV", "bemf_v", "bemfV"),
    "phase_w_back_emf_v": ("phase_w_back_emf_v", "phaseWBackEmf", "backEmfW", "bemf_w", "bemfW"),
    "phase_u_current_a": ("phase_u_current_a", "phaseUCurrent", "phaseCurrentU", "phaseU", "u"),
    "phase_v_current_a": ("phase_v_current_a", "phaseVCurrent", "phaseCurrentV", "phaseV", "v"),
    "phase_w_current_a": ("phase_w_current_a", "phaseWCurrent", "phaseCurrentW", "phaseW", "w"),
    "vibration_x_g": ("vibration_x_g", "vibrationX", "accelX", "imuX", "x"),
    "vibration_y_g": ("vibration_y_g", "vibrationY", "accelY", "imuY", "y"),
    "vibration_z_g": ("vibration_z_g", "vibrationZ", "accelZ", "imuZ", "z"),
    "fault_type": ("fault_type", "fault", "label", "class"),
    "rul_minutes": ("rul_minutes", "rul", "runtime_left_min", "time_to_failure_min"),
}

NUMERIC_COLUMNS = (
    "timestamp_s",
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
)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if np.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return default


def _canonicalize(row: dict[str, str]) -> dict[str, Any]:
    lower_lookup = {key.lower(): key for key in row}
    out: dict[str, Any] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        value = None
        for alias in aliases:
            key = lower_lookup.get(alias.lower())
            if key is not None:
                value = row.get(key)
                break
        if value is not None:
            out[canonical] = value

    generic_bemf_key = lower_lookup.get("backemf") or lower_lookup.get("bemf")
    if generic_bemf_key is not None:
        if "phase_u_back_emf_v" not in out:
            out["phase_u_back_emf_v"] = row[generic_bemf_key]
        if "phase_v_back_emf_v" not in out:
            out["phase_v_back_emf_v"] = row[generic_bemf_key]
        if "phase_w_back_emf_v" not in out:
            out["phase_w_back_emf_v"] = row[generic_bemf_key]

    for column in NUMERIC_COLUMNS:
        out[column] = _to_float(out.get(column, 0.0))
    out["motor_id"] = str(out.get("motor_id", "motor_1"))
    out["fault_type"] = str(out.get("fault_type", "unknown"))
    out["rul_minutes"] = _to_float(out.get("rul_minutes", np.nan), default=np.nan)
    return out


def load_raw_csv_windows(
    path: Path,
    sample_rate_hz: float,
    window_seconds: float,
) -> tuple[list[dict[str, np.ndarray]], list[str], list[float], list[str]]:
    samples_per_window = max(1, int(sample_rate_hz * window_seconds))
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [_canonicalize(row) for row in reader]

    if not rows:
        raise ValueError(f"No telemetry rows found in {path}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["motor_id"]].append(row)

    windows: list[dict[str, np.ndarray]] = []
    labels: list[str] = []
    rul: list[float] = []
    groups: list[str] = []
    for motor_id, motor_rows in grouped.items():
        motor_rows.sort(key=lambda item: item["timestamp_s"])
        for start in range(0, len(motor_rows), samples_per_window):
            chunk = motor_rows[start : start + samples_per_window]
            if len(chunk) < max(4, samples_per_window // 4):
                continue
            window = {
                column: np.asarray([row[column] for row in chunk], dtype=float)
                for column in NUMERIC_COLUMNS
                if column != "timestamp_s"
            }
            window["timestamp_s"] = np.asarray([row["timestamp_s"] for row in chunk], dtype=float)
            label_counts: dict[str, int] = defaultdict(int)
            for row in chunk:
                label_counts[row["fault_type"]] += 1
            label = max(label_counts.items(), key=lambda item: item[1])[0]
            rul_values = [row["rul_minutes"] for row in chunk if np.isfinite(row["rul_minutes"])]
            windows.append(window)
            labels.append(label)
            rul.append(float(np.mean(rul_values)) if rul_values else float("nan"))
            groups.append(motor_id)

    if not windows:
        raise ValueError("Telemetry CSV did not contain enough rows to build one window")
    return windows, labels, rul, groups


def write_feature_csv(path: Path, x: np.ndarray, feature_names: list[str], labels: list[str], rul: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fault_type", "rul_minutes", *feature_names])
        for i in range(x.shape[0]):
            writer.writerow([labels[i], f"{rul[i]:.6f}", *[f"{value:.8g}" for value in x[i]]])


def write_feature_importance_csv(path: Path, names: list[str], importance: np.ndarray) -> None:
    order = np.argsort(importance)[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for idx in order:
            writer.writerow([names[idx], f"{importance[idx]:.10f}"])
