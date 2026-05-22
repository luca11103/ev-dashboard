from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an existing ML run into the dashboard public report folder.")
    parser.add_argument("--run", type=Path, required=True, help="Run folder containing metrics.json and report.html.")
    parser.add_argument("--target", type=Path, required=True, help="Dashboard public report folder.")
    return parser.parse_args()


def _top_features(path: Path, limit: int = 18) -> list[dict[str, float | str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            {"feature": row["feature"], "importance": float(row["importance"])}
            for _, row in zip(range(limit), rows)
        ]


def _summary(metrics: dict[str, Any], features_path: Path) -> dict[str, Any]:
    return {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "metadata": metrics["metadata"],
        "classifierTest": metrics["classifier_test"],
        "regressorTest": metrics["regressor_test"],
        "confusionMatrix": metrics["confusion_matrix"],
        "topFeatures": _top_features(features_path),
        "rulScatter": [
            {
                "actualMinutes": prediction["actual_rul_minutes"],
                "predictedMinutes": prediction["predicted_rul_minutes"],
                "fault": prediction["actual_fault"],
            }
            for prediction in metrics.get("test_predictions", [])
        ],
        "reportHref": "/ml-report/report.html",
    }


def main() -> None:
    args = _parse_args()
    run_dir = args.run.resolve()
    target = args.target.resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    (target / "latest.json").write_text(
        json.dumps(_summary(metrics, run_dir / "feature_importance.csv"), indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(run_dir / "report.html", target / "report.html")
    print(f"Dashboard report exported to: {target}")


if __name__ == "__main__":
    main()
