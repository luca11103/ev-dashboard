#!/usr/bin/env python3
"""Read STM32 serial telemetry and post it to the EV dashboard API."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_CSV_FIELDS = [
    "rpm",
    "temp",
    "inputVoltage",
    "currentDraw",
    "phaseU",
    "phaseV",
    "phaseW",
    "vibrationX",
    "vibrationY",
    "vibrationZ",
    "backEmf",
    "health",
]

FIELD_ALIASES = {
    "rpm": "rpm",
    "speed": "rpm",
    "speedrpm": "rpm",
    "temp": "temp",
    "temperature": "temp",
    "motortemp": "temp",
    "inputvoltage": "inputVoltage",
    "voltage": "inputVoltage",
    "busvoltage": "inputVoltage",
    "current": "currentDraw",
    "currentdraw": "currentDraw",
    "totalcurrent": "currentDraw",
    "phaseu": "phaseU",
    "u": "phaseU",
    "phasecurrentu": "phaseU",
    "phasev": "phaseV",
    "v": "phaseV",
    "phasecurrentv": "phaseV",
    "phasew": "phaseW",
    "w": "phaseW",
    "phasecurrentw": "phaseW",
    "phaseubackemf": "phaseUBackEmf",
    "backemfu": "phaseUBackEmf",
    "bemfu": "phaseUBackEmf",
    "phasevbackemf": "phaseVBackEmf",
    "backemfv": "phaseVBackEmf",
    "bemfv": "phaseVBackEmf",
    "phasewbackemf": "phaseWBackEmf",
    "backemfw": "phaseWBackEmf",
    "bemfw": "phaseWBackEmf",
    "vibrationx": "vibrationX",
    "vibx": "vibrationX",
    "accelx": "vibrationX",
    "imux": "vibrationX",
    "x": "vibrationX",
    "vibrationy": "vibrationY",
    "viby": "vibrationY",
    "accely": "vibrationY",
    "imuy": "vibrationY",
    "y": "vibrationY",
    "vibrationz": "vibrationZ",
    "vibz": "vibrationZ",
    "accelz": "vibrationZ",
    "imuz": "vibrationZ",
    "az": "vibrationZ",
    "z": "vibrationZ",
    "backemf": "backEmf",
    "bemf": "backEmf",
    "tempv": "tempSensorVoltage",
    "tempvoltage": "tempSensorVoltage",
    "temperaturevoltage": "tempSensorVoltage",
    "hall": "rpm",
    "health": "health",
}

TOP_LEVEL_ALIASES = {
    "mlstatus": "mlStatus",
    "controllermode": "controllerMode",
}

TORQUE_ALIASES = {
    "fronttorque": ("torqueSplit", "front"),
    "torquefront": ("torqueSplit", "front"),
    "reartorque": ("torqueSplit", "rear"),
    "torquerear": ("torqueSplit", "rear"),
}


def compact_key(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def parse_value(value: str) -> Any:
    stripped = value.strip()
    if stripped == "":
        return stripped

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        return stripped


def split_motor_key(raw_key: str) -> tuple[str | None, str]:
    key = raw_key.strip()
    normalized = key.lower().replace(".", "_").replace("-", "_").replace(" ", "_")
    for prefix, motor in (
        ("front_", "front"),
        ("rear_", "rear"),
        ("f_", "front"),
        ("r_", "rear"),
    ):
        if normalized.startswith(prefix):
            return motor, normalized[len(prefix):]

    compact = compact_key(key)
    for prefix, motor in (("front", "front"), ("rear", "rear")):
        if compact.startswith(prefix):
            return motor, compact[len(prefix):]

    return None, key


def apply_flat_fields(fields: dict[str, Any], default_motor: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": "stm32-laptop-bridge",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for raw_key, value in fields.items():
        motor, local_key = split_motor_key(raw_key)
        compact = compact_key(local_key)

        torque_target = TORQUE_ALIASES.get(compact_key(raw_key))
        if torque_target:
            group, side = torque_target
            payload.setdefault(group, {})[side] = value
            continue

        top_level = TOP_LEVEL_ALIASES.get(compact_key(raw_key))
        if top_level:
            payload[top_level] = value
            continue

        field = FIELD_ALIASES.get(compact)
        if not field:
            continue

        targets = [motor] if motor else (["front", "rear"] if default_motor == "both" else [default_motor])
        for target in targets:
            payload.setdefault(target, {})[field] = value

    return payload


def normalize_json_payload(data: dict[str, Any], default_motor: str) -> dict[str, Any]:
    if any(key in data for key in ("front", "rear", "frontMotor", "rearMotor")):
        payload = dict(data)
        payload.setdefault("source", "stm32-laptop-bridge")
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        return payload

    return apply_flat_fields(data, default_motor)


def parse_key_value_line(line: str, default_motor: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for token in re.split(r"[,;\s]+", line):
        if not token:
            continue
        separator = "=" if "=" in token else ":" if ":" in token else None
        if not separator:
            continue
        key, value = token.split(separator, 1)
        fields[key] = parse_value(value)

    if not fields:
        raise ValueError("line did not contain key=value telemetry")

    return apply_flat_fields(fields, default_motor)


def parse_csv_line(line: str, default_motor: str, csv_fields: list[str]) -> dict[str, Any]:
    values = [part.strip() for part in line.split(",")]
    if len(values) < 2:
        raise ValueError("CSV telemetry needs at least two values")

    fields = {
        field: parse_value(value)
        for field, value in zip(csv_fields, values, strict=False)
        if value != ""
    }
    return apply_flat_fields(fields, default_motor)


def parse_serial_line(line: str, default_motor: str, csv_fields: list[str]) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        raise ValueError("empty line")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        return normalize_json_payload(data, default_motor)
    if data is not None:
        raise ValueError("JSON telemetry must be an object")

    if "=" in stripped or ":" in stripped:
        return parse_key_value_line(stripped, default_motor)

    return parse_csv_line(stripped, default_motor, csv_fields)


ML_COLUMNS = (
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

ML_FIELD_LOOKUP = {
    "temp_c": ("temp", "temperature", "motorTemp"),
    "rpm": ("rpm", "speedRpm", "speed"),
    "battery_voltage_v": ("inputVoltage", "batteryVoltage", "busVoltage", "voltage"),
    "battery_current_a": ("currentDraw", "batteryCurrent", "totalCurrent", "current"),
    "phase_u_back_emf_v": ("phaseUBackEmf", "backEmfU", "bemfU", "backEmf"),
    "phase_v_back_emf_v": ("phaseVBackEmf", "backEmfV", "bemfV", "backEmf"),
    "phase_w_back_emf_v": ("phaseWBackEmf", "backEmfW", "bemfW", "backEmf"),
    "phase_u_current_a": ("phaseU", "phaseUCurrent", "phaseCurrentU", "u"),
    "phase_v_current_a": ("phaseV", "phaseVCurrent", "phaseCurrentV", "v"),
    "phase_w_current_a": ("phaseW", "phaseWCurrent", "phaseCurrentW", "w"),
    "vibration_x_g": ("vibrationX", "accelX", "imuX", "x"),
    "vibration_y_g": ("vibrationY", "accelY", "imuY", "y"),
    "vibration_z_g": ("vibrationZ", "accelZ", "imuZ", "z"),
}


def read_number(mapping: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = mapping.get(alias)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed not in (float("inf"), float("-inf")):
            return parsed
    return None


def downsample_points(freqs: Any, amplitudes: Any, max_points: int) -> list[dict[str, float]]:
    if len(freqs) == 0:
        return []

    if len(freqs) <= max_points:
        indexes = range(len(freqs))
    else:
        step = max(1, len(freqs) // max_points)
        indexes = range(0, len(freqs), step)

    return [
        {"hz": round(float(freqs[index]), 3), "amp": round(float(amplitudes[index]), 6)}
        for index in indexes
    ][:max_points]


def health_index(label: str, confidence: float) -> int:
    if label == "normal":
        return int(round(90 + min(10.0, confidence * 10.0)))
    return int(round(max(12.0, 82.0 - confidence * 62.0)))


class LiveFaultScorer:
    def __init__(
        self,
        model_path: Path,
        sample_rate_hz: float | None,
        window_seconds: float | None,
        prediction_interval_ms: int,
        spectrum_points: int,
        allow_partial_ml: bool,
    ) -> None:
        ml_root = Path(__file__).resolve().parents[1] / "bldc_fault_ml"
        if str(ml_root) not in sys.path:
            sys.path.insert(0, str(ml_root))

        import numpy as np
        from bldc_fault_ml.features import build_feature_frame
        from bldc_fault_ml.signal import bandpass_fft, fft_spectrum

        with model_path.open("rb") as model_file:
            self.bundle = pickle.load(model_file)

        self.np = np
        self.build_feature_frame = build_feature_frame
        self.bandpass_fft = bandpass_fft
        self.fft_spectrum = fft_spectrum
        self.sample_rate_hz = sample_rate_hz or float(self.bundle.sample_rate_hz)
        self.window_seconds = window_seconds or float(self.bundle.window_seconds)
        self.samples_per_window = max(4, int(round(self.sample_rate_hz * self.window_seconds)))
        self.prediction_interval_s = max(0, prediction_interval_ms) / 1000.0
        self.spectrum_points = max(24, spectrum_points)
        self.allow_partial_ml = allow_partial_ml
        self.buffers: dict[str, deque[dict[str, float | None]]] = {}
        self.last_scored_at: dict[str, float] = {}
        self.latest: dict[str, dict[str, Any]] = {}

    def enrich_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        for motor_id in ("front", "rear"):
            motor = payload.get(motor_id)
            if not isinstance(motor, dict):
                continue

            buffer = self.buffers.setdefault(motor_id, deque(maxlen=self.samples_per_window))
            buffer.append(self._sample_from_motor(motor))
            self._score_if_ready(motor_id, buffer)

            latest = self.latest.get(motor_id)
            if latest:
                motor["ml"] = latest["ml"]
                motor["signal"] = latest["signal"]
                if "health" not in motor:
                    motor["health"] = latest["ml"]["healthIndex"]
            else:
                motor["ml"] = self._collecting_status(buffer)
        return payload

    def _sample_from_motor(self, motor: dict[str, Any]) -> dict[str, float | None]:
        return {
            column: read_number(motor, aliases)
            for column, aliases in ML_FIELD_LOOKUP.items()
        }

    def _collecting_status(self, buffer: deque[dict[str, float | None]]) -> dict[str, Any]:
        return {
            "status": "collecting",
            "sampleCount": len(buffer),
            "requiredSamples": self.samples_per_window,
            "sampleRateHz": self.sample_rate_hz,
            "windowSeconds": self.window_seconds,
        }

    def _score_if_ready(self, motor_id: str, buffer: deque[dict[str, float | None]]) -> None:
        if len(buffer) < self.samples_per_window:
            return

        now = time.monotonic()
        if now - self.last_scored_at.get(motor_id, 0.0) < self.prediction_interval_s:
            return
        self.last_scored_at[motor_id] = now

        missing = sorted(
            column
            for column in ML_COLUMNS
            if any(sample[column] is None for sample in buffer)
        )
        signal = self._signal_summary(buffer)
        if missing and not self.allow_partial_ml:
            self.latest[motor_id] = {
                "ml": {
                    "status": "waiting_for_channels",
                    "missingFields": missing,
                    "sampleCount": len(buffer),
                    "requiredSamples": self.samples_per_window,
                    "sampleRateHz": self.sample_rate_hz,
                    "windowSeconds": self.window_seconds,
                },
                "signal": signal,
            }
            return

        np = self.np
        window = {
            column: np.asarray(
                [sample[column] if sample[column] is not None else 0.0 for sample in buffer],
                dtype=float,
            )
            for column in ML_COLUMNS
        }
        frame = self.build_feature_frame(
            windows=[window],
            labels=["unknown"],
            rul_minutes=[0.0],
            groups=[motor_id],
            sample_rate_hz=self.sample_rate_hz,
            bandpass_low_hz=self.bundle.bandpass_low_hz,
            bandpass_high_hz=self.bundle.bandpass_high_hz,
        )
        features = self._align_features(frame.feature_names, frame.x, self.bundle.feature_names)
        scaled = self.bundle.transform_features(features)
        probabilities = self.bundle.classifier().predict_proba(scaled)[0]
        label_index = int(np.argmax(probabilities))
        label = self.bundle.label_encoder.classes_[label_index]
        confidence = float(probabilities[label_index])
        rul_prediction = self.bundle.regressor().predict(scaled)
        if getattr(self.bundle, "rul_target_log", True):
            rul_prediction = np.expm1(rul_prediction)
        rul_minutes = max(0.0, float(rul_prediction[0]))
        top_indexes = np.argsort(probabilities)[::-1][:4]

        self.latest[motor_id] = {
            "ml": {
                "status": "ready",
                "predictedFault": label,
                "confidence": round(confidence, 5),
                "rulMinutes": round(rul_minutes, 2),
                "healthIndex": health_index(label, confidence),
                "topFaultProbabilities": {
                    self.bundle.label_encoder.classes_[int(index)]: round(float(probabilities[index]), 5)
                    for index in top_indexes
                },
                "sampleCount": len(buffer),
                "sampleRateHz": self.sample_rate_hz,
                "windowSeconds": self.window_seconds,
                "partialInput": bool(missing),
                "missingFields": missing,
            },
            "signal": signal,
        }

    def _signal_summary(self, buffer: deque[dict[str, float | None]]) -> dict[str, Any]:
        np = self.np
        filtered_axes = []
        for axis in ("vibration_x_g", "vibration_y_g", "vibration_z_g"):
            values = np.asarray(
                [sample[axis] if sample[axis] is not None else 0.0 for sample in buffer],
                dtype=float,
            )
            filtered_axes.append(
                self.bandpass_fft(
                    values,
                    self.sample_rate_hz,
                    self.bundle.bandpass_low_hz,
                    self.bundle.bandpass_high_hz,
                )
            )

        magnitude = np.sqrt(np.sum(np.vstack(filtered_axes) ** 2, axis=0))
        freqs, amplitudes = self.fft_spectrum(magnitude, self.sample_rate_hz)
        in_band = freqs <= min(self.bundle.bandpass_high_hz, self.sample_rate_hz / 2.0)
        plotted_freqs = freqs[in_band]
        plotted_amplitudes = amplitudes[in_band]
        dominant_index = int(np.argmax(plotted_amplitudes)) if len(plotted_amplitudes) else 0
        dominant_hz = float(plotted_freqs[dominant_index]) if len(plotted_freqs) else 0.0
        peak_amp = float(plotted_amplitudes[dominant_index]) if len(plotted_amplitudes) else 0.0

        return {
            "fft": {
                "axis": "bandpass_vibration_magnitude",
                "bandpassHz": [self.bundle.bandpass_low_hz, self.bundle.bandpass_high_hz],
                "dominantHz": round(dominant_hz, 3),
                "peakAmplitude": round(peak_amp, 6),
                "points": downsample_points(plotted_freqs, plotted_amplitudes, self.spectrum_points),
            }
        }

    def _align_features(self, frame_names: list[str], x: Any, target_names: list[str]) -> Any:
        lookup = {name: index for index, name in enumerate(frame_names)}
        aligned = self.np.zeros((x.shape[0], len(target_names)), dtype=float)
        for target_index, name in enumerate(target_names):
            source_index = lookup.get(name)
            if source_index is not None:
                aligned[:, target_index] = x[:, source_index]
        return aligned


def post_payload(url: str, payload: dict[str, Any], api_key: str | None, key_header: str, timeout: float) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers[key_header] = api_key

    api_request = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(api_request, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge STM32 serial telemetry to the EV dashboard.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5 on Windows.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--url", required=True, help="Telemetry API URL, for example https://your-site.vercel.app/api/telemetry.")
    parser.add_argument("--key", default=None, help="Optional telemetry API key.")
    parser.add_argument("--key-header", default="x-telemetry-key", help="Header used for the optional API key.")
    parser.add_argument("--motor", choices=("front", "rear", "both"), default="front", help="Where unprefixed values should be applied.")
    parser.add_argument("--csv-fields", default=",".join(DEFAULT_CSV_FIELDS), help="Comma-separated field order for raw CSV serial lines.")
    parser.add_argument("--min-interval-ms", type=int, default=100, help="Minimum delay between API posts.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds.")
    parser.add_argument("--model", type=Path, help="Optional model.pkl used for live fault/RUL windows.")
    parser.add_argument("--sample-rate", type=float, help="Raw UART sample rate in Hz for ML windows.")
    parser.add_argument("--window-seconds", type=float, help="Override model window duration for live ML.")
    parser.add_argument("--prediction-interval-ms", type=int, default=500, help="Minimum delay between ML window scores.")
    parser.add_argument("--spectrum-points", type=int, default=96, help="Maximum FFT points sent to the dashboard.")
    parser.add_argument(
        "--allow-partial-ml",
        action="store_true",
        help="Score windows with missing electrical channels filled with zero.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print parsed payloads without posting.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_fields = [field.strip() for field in args.csv_fields.split(",") if field.strip()]
    scorer = None

    if args.model:
        try:
            scorer = LiveFaultScorer(
                model_path=args.model,
                sample_rate_hz=args.sample_rate,
                window_seconds=args.window_seconds,
                prediction_interval_ms=args.prediction_interval_ms,
                spectrum_points=args.spectrum_points,
                allow_partial_ml=args.allow_partial_ml,
            )
        except Exception as exc:
            print(f"Could not load live ML model: {exc}", file=sys.stderr)
            return 1

    try:
        import serial
    except ModuleNotFoundError:
        print("pyserial is not installed. Run: python -m pip install -r tools/requirements.txt", file=sys.stderr)
        return 1

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open {args.port}: {exc}", file=sys.stderr)
        return 1

    print(f"Reading STM32 telemetry from {args.port} at {args.baud} baud")
    print(f"Posting to {args.url}" if not args.dry_run else "Dry run mode: not posting")
    if scorer:
        print(
            "Live ML enabled: "
            f"{scorer.samples_per_window} samples/window at {scorer.sample_rate_hz:g} Hz"
        )

    last_posted = 0.0
    min_interval = max(0, args.min_interval_ms) / 1000

    try:
        while True:
            raw = serial_port.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                payload = parse_serial_line(line, args.motor, csv_fields)
            except ValueError as exc:
                print(f"Skipped line: {exc}: {line}", file=sys.stderr)
                continue

            if scorer:
                payload = scorer.enrich_payload(payload)

            now = time.monotonic()
            if now - last_posted < min_interval:
                continue
            last_posted = now

            if args.dry_run:
                print(json.dumps(payload, indent=2))
                continue

            try:
                post_payload(args.url, payload, args.key, args.key_header, args.timeout)
                print(f"Posted telemetry at {datetime.now().strftime('%H:%M:%S')}")
            except (error.URLError, TimeoutError, RuntimeError) as exc:
                print(f"Post failed: {exc}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nBridge stopped")
        return 0
    finally:
        serial_port.close()


if __name__ == "__main__":
    raise SystemExit(main())
