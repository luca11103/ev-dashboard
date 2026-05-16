from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from bldc_fault_ml.features import build_feature_frame
from bldc_fault_ml.io import (
    load_raw_csv_windows,
    write_feature_csv,
    write_feature_importance_csv,
)
from bldc_fault_ml.models import (
    BaggedTreesClassifier,
    BaggedTreesRegressor,
    FaultModelBundle,
    GaussianNB,
    KNNClassifier,
    KNNRegressor,
    LabelEncoder,
    Preprocessor,
    RidgeRegressor,
    WeightedClassifierEnsemble,
    WeightedRegressorEnsemble,
    accuracy_score,
    confusion_matrix,
    macro_f1_score,
    regression_metrics,
    stratified_split,
)
from bldc_fault_ml.reporting import write_html_report
from bldc_fault_ml.synthetic import FAULT_CLASSES, generate_synthetic_windows, write_raw_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BLDC fault classification and RUL models.")
    parser.add_argument("--csv", type=Path, help="Raw telemetry CSV with labels and RUL.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic BLDC fault data.")
    parser.add_argument("--sample-rate", type=float, default=400.0, help="ADXL sample rate in Hz.")
    parser.add_argument("--window-seconds", type=float, default=2.0, help="Feature window length in seconds.")
    parser.add_argument("--bandpass-low", type=float, default=5.0, help="Vibration band-pass low cutoff in Hz.")
    parser.add_argument("--bandpass-high", type=float, default=180.0, help="Vibration band-pass high cutoff in Hz.")
    parser.add_argument("--windows-per-class", type=int, default=36, help="Demo windows per fault class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--export-demo-windows",
        type=int,
        default=28,
        help="How many synthetic raw windows to export to demo_dataset.csv.",
    )
    return parser.parse_args()


def _normalise_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in raw.values())
    if total <= 1e-12:
        count = max(1, len(raw))
        return {name: 1.0 / count for name in raw}
    return {name: max(0.0, value) / total for name, value in raw.items()}


def _classifier_model_report(
    models: dict[str, object],
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    raw_weights: dict[str, float] = {}
    for name, model in models.items():
        pred = model.predict(x_val)
        acc = accuracy_score(y_val, pred)
        f1 = macro_f1_score(y_val, pred, n_classes)
        report[name] = {"accuracy": acc, "macro_f1": f1}
        raw_weights[name] = max(0.01, (0.55 * acc + 0.45 * f1) ** 2)
    return report, _normalise_weights(raw_weights)


def _regressor_model_report(
    models: dict[str, object],
    x_val: np.ndarray,
    y_val_minutes: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    raw_weights: dict[str, float] = {}
    for name, model in models.items():
        pred_minutes = np.expm1(model.predict(x_val))
        pred_minutes = np.maximum(pred_minutes, 0.0)
        metrics = regression_metrics(y_val_minutes, pred_minutes)
        report[name] = metrics
        raw_weights[name] = 1.0 / max(metrics["mae"], 1.0)
    return report, _normalise_weights(raw_weights)


def _class_counts(labels: list[str], class_labels: list[str]) -> np.ndarray:
    return np.asarray([sum(1 for label in labels if label == cls) for cls in class_labels], dtype=float)


def main() -> None:
    args = _parse_args()
    if not args.csv and not args.demo:
        args.demo = True

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("Generating synthetic BLDC telemetry windows...")
        windows, labels, rul_minutes, groups = generate_synthetic_windows(
            windows_per_class=args.windows_per_class,
            sample_rate_hz=args.sample_rate,
            window_seconds=args.window_seconds,
            seed=args.seed,
        )
        write_raw_csv(
            out_dir / "demo_dataset.csv",
            windows,
            labels,
            rul_minutes,
            groups,
            max_windows=args.export_demo_windows,
        )
        source = "synthetic_demo"
    else:
        if not args.csv:
            raise ValueError("Pass --csv or use --demo")
        print(f"Loading telemetry CSV: {args.csv}")
        windows, labels, rul_minutes, groups = load_raw_csv_windows(
            args.csv,
            sample_rate_hz=args.sample_rate,
            window_seconds=args.window_seconds,
        )
        source = str(args.csv)

    if any(label == "unknown" for label in labels):
        raise ValueError("Training requires a fault_type/label column. Use predict_fault.py for unlabeled data.")
    if not np.all(np.isfinite(np.asarray(rul_minutes, dtype=float))):
        raise ValueError("Training requires finite rul_minutes values for every window.")

    print("Extracting vibration FFT, time-domain, electrical, phase, and thermal features...")
    frame = build_feature_frame(
        windows=windows,
        labels=labels,
        rul_minutes=rul_minutes,
        groups=groups,
        sample_rate_hz=args.sample_rate,
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
    )
    write_feature_csv(out_dir / "window_features.csv", frame.x, frame.feature_names, labels, frame.rul_minutes)

    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    train_idx, val_idx, test_idx = stratified_split(y, seed=args.seed)
    if test_idx.size == 0:
        test_idx = val_idx.copy()
    if val_idx.size == 0:
        val_idx = train_idx.copy()

    preprocessor = Preprocessor()
    x_train = preprocessor.fit_transform(frame.x[train_idx])
    x_val = preprocessor.transform(frame.x[val_idx])
    x_test = preprocessor.transform(frame.x[test_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print("Training multi-model fault classifier...")
    n_classes = len(encoder.classes_)
    classifier_models = {
        "gaussian_nb": GaussianNB().fit(x_train, y_train),
        "knn": KNNClassifier(k=7, n_classes=n_classes).fit(x_train, y_train),
        "bagged_trees": BaggedTreesClassifier(
            n_estimators=31,
            max_depth=9,
            min_samples_leaf=3,
            random_state=args.seed,
            n_classes=n_classes,
        ).fit(x_train, y_train),
    }
    classifier_val_report, classifier_weights = _classifier_model_report(
        classifier_models,
        x_val,
        y_val,
        n_classes,
    )
    classifier = WeightedClassifierEnsemble(classifier_models, classifier_weights)

    print("Training RUL ensemble regressors...")
    y_train_rul = frame.rul_minutes[train_idx]
    y_val_rul = frame.rul_minutes[val_idx]
    y_test_rul = frame.rul_minutes[test_idx]
    y_train_log = np.log1p(np.maximum(y_train_rul, 0.0))
    regressor_models = {
        "ridge_log_rul": RidgeRegressor(alpha=2.5).fit(x_train, y_train_log),
        "knn_log_rul": KNNRegressor(k=7).fit(x_train, y_train_log),
        "bagged_tree_log_rul": BaggedTreesRegressor(
            n_estimators=31,
            max_depth=9,
            min_samples_leaf=3,
            random_state=args.seed + 1000,
        ).fit(x_train, y_train_log),
    }
    regressor_val_report, regressor_weights = _regressor_model_report(regressor_models, x_val, y_val_rul)
    regressor = WeightedRegressorEnsemble(regressor_models, regressor_weights)

    pred_test = classifier.predict(x_test)
    prob_test = classifier.predict_proba(x_test)
    pred_test_labels = encoder.inverse_transform(pred_test)
    rul_pred_test = np.maximum(np.expm1(regressor.predict(x_test)), 0.0)

    classifier_metrics = {
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_macro_f1": macro_f1_score(y_test, pred_test, n_classes),
        "mean_prediction_confidence": float(np.mean(np.max(prob_test, axis=1))),
        "test_windows": int(test_idx.size),
    }
    regressor_test_metrics = {"ensemble_log_rul": regression_metrics(y_test_rul, rul_pred_test)}
    regressor_all_metrics = {**regressor_val_report, **regressor_test_metrics}
    confusion = confusion_matrix(y_test, pred_test, n_classes)

    tree_importance = classifier_models["bagged_trees"].feature_importances_
    reg_importance = regressor_models["bagged_tree_log_rul"].feature_importances_
    if tree_importance is None:
        tree_importance = np.zeros(len(frame.feature_names))
    if reg_importance is None:
        reg_importance = np.zeros(len(frame.feature_names))
    feature_importance = (0.7 * tree_importance) + (0.3 * reg_importance)
    total_importance = float(np.sum(feature_importance))
    if total_importance > 0:
        feature_importance = feature_importance / total_importance

    metadata = {
        "source": source,
        "windows": len(labels),
        "features": len(frame.feature_names),
        "sample_rate_hz": args.sample_rate,
        "window_seconds": args.window_seconds,
        "bandpass_hz": [args.bandpass_low, args.bandpass_high],
        "fault_classes": encoder.classes_,
        "classifier_validation": classifier_val_report,
        "classifier_weights": classifier_weights,
        "regressor_validation": regressor_val_report,
        "regressor_weights": regressor_weights,
        "note": "Synthetic demo metrics prove the pipeline; final accuracy requires real labeled BLDC data.",
    }

    metrics = {
        "metadata": metadata,
        "classifier_test": classifier_metrics,
        "regressor_test": regressor_test_metrics,
        "confusion_matrix": confusion.tolist(),
        "test_predictions": [
            {
                "actual_fault": labels[int(idx)],
                "predicted_fault": pred_test_labels[row_idx],
                "actual_rul_minutes": float(frame.rul_minutes[int(idx)]),
                "predicted_rul_minutes": float(rul_pred_test[row_idx]),
                "confidence": float(np.max(prob_test[row_idx])),
            }
            for row_idx, idx in enumerate(test_idx[:30])
        ],
    }

    bundle = FaultModelBundle(
        feature_names=frame.feature_names,
        label_encoder=encoder,
        preprocessor=preprocessor,
        classifier_models=classifier_models,
        classifier_weights=classifier_weights,
        regressor_models=regressor_models,
        regressor_weights=regressor_weights,
        sample_rate_hz=args.sample_rate,
        window_seconds=args.window_seconds,
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
        rul_target_log=True,
    )

    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_feature_importance_csv(out_dir / "feature_importance.csv", frame.feature_names, feature_importance)

    sample_idx = int(test_idx[0]) if test_idx.size else 0
    write_html_report(
        out_dir / "report.html",
        title="BLDC Fault Detection ML Report",
        class_labels=encoder.classes_,
        class_counts=_class_counts(labels, encoder.classes_),
        classifier_metrics=classifier_metrics,
        regressor_metrics_map=regressor_all_metrics,
        confusion=confusion,
        y_rul_true=y_test_rul,
        y_rul_pred=rul_pred_test,
        feature_names=frame.feature_names,
        feature_importance=feature_importance,
        sample_window=frame.raw_windows[sample_idx],
        sample_rate_hz=args.sample_rate,
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
        metadata=metadata,
    )

    print()
    print("Training complete.")
    print(f"Output folder: {out_dir}")
    print(f"Fault classifier accuracy: {classifier_metrics['test_accuracy']:.3f}")
    print(f"Fault classifier macro F1: {classifier_metrics['test_macro_f1']:.3f}")
    print(f"RUL MAE minutes: {regressor_test_metrics['ensemble_log_rul']['mae']:.2f}")
    print("Top features:")
    for idx in np.argsort(feature_importance)[::-1][:10]:
        print(f"  {frame.feature_names[idx]}: {feature_importance[idx]:.4f}")


if __name__ == "__main__":
    main()
