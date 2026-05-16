from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .signal import bandpass_fft, fft_spectrum


PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4f46e5"]


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return np.full(values.shape, (low + high) / 2.0)
    return low + (values - vmin) * (high - low) / (vmax - vmin)


def _polyline(x: np.ndarray, y: np.ndarray) -> str:
    return " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(x, y))


def _bar_chart(title: str, labels: list[str], values: np.ndarray, width: int = 760, height: int = 320) -> str:
    margin_left = 180
    margin_right = 30
    margin_top = 36
    bar_h = max(18, int((height - margin_top - 30) / max(len(labels), 1) * 0.58))
    gap = max(8, int(bar_h * 0.55))
    plot_w = width - margin_left - margin_right
    max_v = float(np.max(values)) if values.size else 1.0
    rows = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + i * (bar_h + gap)
        w = (float(value) / max(max_v, 1e-12)) * plot_w
        rows.append(
            f'<text x="{margin_left - 10}" y="{y + bar_h * 0.7:.1f}" text-anchor="end">{html.escape(label)}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{PALETTE[i % len(PALETTE)]}"/>'
            f'<text x="{margin_left + w + 8:.1f}" y="{y + bar_h * 0.7:.1f}">{float(value):.3g}</text>'
        )
    svg_h = max(height, margin_top + len(labels) * (bar_h + gap) + 20)
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {svg_h}" role="img">'
        f'<rect width="{width}" height="{svg_h}" fill="#ffffff"/>'
        f'{"".join(rows)}'
        "</svg>"
    )


def _confusion_matrix(title: str, labels: list[str], matrix: np.ndarray) -> str:
    size = 42
    left = 170
    top = 60
    width = left + size * len(labels) + 40
    height = top + size * len(labels) + 120
    max_v = max(float(np.max(matrix)), 1.0)
    cells = []
    for i, actual in enumerate(labels):
        cells.append(f'<text x="{left - 10}" y="{top + i * size + 27}" text-anchor="end">{html.escape(actual)}</text>')
        for j, pred in enumerate(labels):
            value = int(matrix[i, j])
            intensity = value / max_v
            color = f"rgba(37, 99, 235, {0.12 + 0.82 * intensity:.3f})"
            text_color = "#ffffff" if intensity > 0.45 else "#0f172a"
            x = left + j * size
            y = top + i * size
            cells.append(
                f'<rect x="{x}" y="{y}" width="{size - 2}" height="{size - 2}" rx="5" fill="{color}"/>'
                f'<text x="{x + size / 2}" y="{y + 27}" text-anchor="middle" fill="{text_color}">{value}</text>'
            )
    for j, pred in enumerate(labels):
        x = left + j * size + 22
        cells.append(f'<text transform="translate({x},{top - 12}) rotate(-45)" text-anchor="start">{html.escape(pred)}</text>')
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<text x="{left}" y="28" class="axis-title">Predicted fault type</text>'
        f'<text x="20" y="{top + len(labels) * size / 2}" transform="rotate(-90,20,{top + len(labels) * size / 2})" class="axis-title">Actual fault type</text>'
        f'{"".join(cells)}'
        "</svg>"
    )


def _scatter(title: str, actual: np.ndarray, pred: np.ndarray, width: int = 760, height: int = 360) -> str:
    margin = 58
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    if actual.size == 0:
        return f"<h2>{html.escape(title)}</h2><p>No finite RUL values available.</p>"
    limit = float(max(np.max(actual), np.max(pred), 1.0))
    x = margin + (actual / limit) * (width - margin * 2)
    y = height - margin - (pred / limit) * (height - margin * 2)
    points = "".join(
        f'<circle cx="{float(a):.1f}" cy="{float(b):.1f}" r="3.2" fill="#2563eb" opacity="0.72"/>'
        for a, b in zip(x, y)
    )
    line = f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{margin}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 5"/>'
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#64748b"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#64748b"/>'
        f'{line}{points}'
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle">Actual RUL minutes</text>'
        f'<text x="18" y="{height / 2}" transform="rotate(-90,18,{height / 2})" text-anchor="middle">Predicted RUL minutes</text>'
        "</svg>"
    )


def _line_chart(title: str, values: np.ndarray, sample_rate_hz: float, width: int = 760, height: int = 300) -> str:
    values = np.asarray(values, dtype=float)
    max_points = min(values.size, 600)
    if max_points < 2:
        return f"<h2>{html.escape(title)}</h2><p>Not enough samples.</p>"
    values = values[:max_points]
    t = np.arange(values.size) / sample_rate_hz
    x = _scale(t, 52, width - 24)
    y = height - _scale(values, 44, height - 42)
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<line x1="52" y1="{height - 42}" x2="{width - 24}" y2="{height - 42}" stroke="#64748b"/>'
        f'<line x1="52" y1="32" x2="52" y2="{height - 42}" stroke="#64748b"/>'
        f'<polyline points="{_polyline(x, y)}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle">Time seconds</text>'
        f'<text x="15" y="{height / 2}" transform="rotate(-90,15,{height / 2})" text-anchor="middle">Vibration g</text>'
        "</svg>"
    )


def _spectrum_chart(title: str, values: np.ndarray, sample_rate_hz: float, width: int = 760, height: int = 300) -> str:
    freqs, amp = fft_spectrum(values, sample_rate_hz)
    mask = freqs <= min(sample_rate_hz / 2.0, 250.0)
    freqs = freqs[mask]
    amp = amp[mask]
    if freqs.size < 2:
        return f"<h2>{html.escape(title)}</h2><p>Not enough samples.</p>"
    x = _scale(freqs, 52, width - 24)
    y = height - _scale(amp, 44, height - 42)
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<line x1="52" y1="{height - 42}" x2="{width - 24}" y2="{height - 42}" stroke="#64748b"/>'
        f'<line x1="52" y1="32" x2="52" y2="{height - 42}" stroke="#64748b"/>'
        f'<polyline points="{_polyline(x, y)}" fill="none" stroke="#16a34a" stroke-width="2"/>'
        f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle">Frequency Hz</text>'
        f'<text x="15" y="{height / 2}" transform="rotate(-90,15,{height / 2})" text-anchor="middle">Amplitude</text>'
        "</svg>"
    )


def _metrics_table(title: str, metrics: dict[str, float | int | str]) -> str:
    rows = []
    for key, value in metrics.items():
        display = f"{value:.4f}" if isinstance(value, float) else str(value)
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(display)}</td></tr>")
    return f"<h2>{html.escape(title)}</h2><table>{''.join(rows)}</table>"


def write_html_report(
    path: Path,
    *,
    title: str,
    class_labels: list[str],
    class_counts: np.ndarray,
    classifier_metrics: dict[str, float],
    regressor_metrics_map: dict[str, dict[str, float]],
    confusion: np.ndarray,
    y_rul_true: np.ndarray,
    y_rul_pred: np.ndarray,
    feature_names: list[str],
    feature_importance: np.ndarray,
    sample_window: dict[str, np.ndarray],
    sample_rate_hz: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_idx = np.argsort(feature_importance)[::-1][:22]
    top_labels = [feature_names[i] for i in top_idx]
    top_values = feature_importance[top_idx]

    vib_raw = np.asarray(sample_window["vibration_x_g"], dtype=float)
    vib_filtered = bandpass_fft(vib_raw, sample_rate_hz, bandpass_low_hz, bandpass_high_hz)

    reg_tables = "".join(
        _metrics_table(f"RUL Regression: {name}", values)
        for name, values in regressor_metrics_map.items()
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #0f172a;
      background: #f8fafc;
      line-height: 1.45;
    }}
    header {{
      background: #0f172a;
      color: white;
      padding: 28px 36px;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px;
    }}
    section {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 22px;
      margin: 0 0 20px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    h2 {{
      font-size: 18px;
      color: #1e293b;
    }}
    svg {{
      max-width: 100%;
      height: auto;
      font-size: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 8px;
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      width: 42%;
      color: #475569;
      font-weight: 700;
    }}
    code, pre {{
      background: #f1f5f9;
      border-radius: 6px;
    }}
    pre {{
      padding: 14px;
      overflow: auto;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .axis-title {{
      font-weight: 700;
      fill: #334155;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>BLDC motor fault classification and predictive maintenance report.</p>
  </header>
  <main>
    <section>
      <h2>Run Metadata</h2>
      <pre>{html.escape(json.dumps(metadata, indent=2))}</pre>
    </section>
    <section class="grid">
      <div>{_metrics_table("Fault Classification", classifier_metrics)}</div>
      <div>{reg_tables}</div>
    </section>
    <section>{_bar_chart("Class Distribution", class_labels, class_counts)}</section>
    <section>{_confusion_matrix("Fault Confusion Matrix", class_labels, confusion)}</section>
    <section>{_scatter("RUL Prediction: Actual vs Predicted", y_rul_true, y_rul_pred)}</section>
    <section>{_bar_chart("Top Feature Importance", top_labels, top_values)}</section>
    <section class="grid">
      <div>{_line_chart("ADXL X Axis Raw Time Domain", vib_raw, sample_rate_hz)}</div>
      <div>{_line_chart("ADXL X Axis Band-Pass Time Domain", vib_filtered, sample_rate_hz)}</div>
    </section>
    <section class="grid">
      <div>{_spectrum_chart("ADXL X Axis Raw FFT", vib_raw, sample_rate_hz)}</div>
      <div>{_spectrum_chart("ADXL X Axis Band-Pass FFT", vib_filtered, sample_rate_hz)}</div>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
