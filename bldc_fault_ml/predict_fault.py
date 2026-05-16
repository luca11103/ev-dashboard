from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from bldc_fault_ml.features import build_feature_frame
from bldc_fault_ml.io import load_raw_csv_windows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLDC fault prediction on raw telemetry CSV windows.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.pkl from train_fault_model.py.")
    parser.add_argument("--csv", type=Path, required=True, help="Raw telemetry CSV to score.")
    parser.add_argument("--sample-rate", type=float, help="Override sample rate in Hz.")
    parser.add_argument("--window-seconds", type=float, help="Override window size in seconds.")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Prediction CSV output.")
    return parser.parse_args()


def _align_features(frame_names: list[str], x: np.ndarray, target_names: list[str]) -> np.ndarray:
    lookup = {name: idx for idx, name in enumerate(frame_names)}
    aligned = np.zeros((x.shape[0], len(target_names)), dtype=float)
    for target_idx, name in enumerate(target_names):
        source_idx = lookup.get(name)
        if source_idx is not None:
            aligned[:, target_idx] = x[:, source_idx]
    return aligned


def main() -> None:
    args = _parse_args()
    with args.model.open("rb") as f:
        bundle = pickle.load(f)

    sample_rate = args.sample_rate or bundle.sample_rate_hz
    window_seconds = args.window_seconds or bundle.window_seconds
    windows, labels, rul, groups = load_raw_csv_windows(args.csv, sample_rate, window_seconds)
    safe_rul = [value if np.isfinite(value) else 0.0 for value in rul]
    frame = build_feature_frame(
        windows=windows,
        labels=labels,
        rul_minutes=safe_rul,
        groups=groups,
        sample_rate_hz=sample_rate,
        bandpass_low_hz=bundle.bandpass_low_hz,
        bandpass_high_hz=bundle.bandpass_high_hz,
    )
    x = _align_features(frame.feature_names, frame.x, bundle.feature_names)
    x_scaled = bundle.transform_features(x)

    probs = bundle.classifier().predict_proba(x_scaled)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = bundle.label_encoder.inverse_transform(pred_idx)
    rul_pred = bundle.regressor().predict(x_scaled)
    if getattr(bundle, "rul_target_log", True):
        rul_pred = np.expm1(rul_pred)
    rul_pred = np.maximum(rul_pred, 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window",
                "motor_id",
                "predicted_fault",
                "fault_confidence",
                "estimated_rul_minutes",
                "top_fault_probabilities_json",
            ],
        )
        writer.writeheader()
        for i, label in enumerate(pred_labels):
            top = np.argsort(probs[i])[::-1][:3]
            top_probs = {
                bundle.label_encoder.classes_[int(idx)]: round(float(probs[i, idx]), 5)
                for idx in top
            }
            writer.writerow(
                {
                    "window": i,
                    "motor_id": groups[i],
                    "predicted_fault": label,
                    "fault_confidence": f"{float(np.max(probs[i])):.5f}",
                    "estimated_rul_minutes": f"{float(rul_pred[i]):.2f}",
                    "top_fault_probabilities_json": json.dumps(top_probs),
                }
            )

    print(f"Scored {len(pred_labels)} windows.")
    print(f"Predictions written to: {args.output}")
    for i, label in enumerate(pred_labels[:10]):
        print(
            f"window={i} motor={groups[i]} fault={label} "
            f"confidence={float(np.max(probs[i])):.3f} rul_min={float(rul_pred[i]):.1f}"
        )


if __name__ == "__main__":
    main()
