#!/usr/bin/env python3
"""Read STM32 serial telemetry and post it to the EV dashboard API."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
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
    "vibrationx": "vibrationX",
    "vibx": "vibrationX",
    "imux": "vibrationX",
    "x": "vibrationX",
    "vibrationy": "vibrationY",
    "viby": "vibrationY",
    "imuy": "vibrationY",
    "y": "vibrationY",
    "vibrationz": "vibrationZ",
    "vibz": "vibrationZ",
    "imuz": "vibrationZ",
    "z": "vibrationZ",
    "backemf": "backEmf",
    "bemf": "backEmf",
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
    parser.add_argument("--min-interval-ms", type=int, default=200, help="Minimum delay between API posts.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed payloads without posting.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_fields = [field.strip() for field in args.csv_fields.split(",") if field.strip()]

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
