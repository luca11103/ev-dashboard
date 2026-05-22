# BLDC Motor Fault ML Lab

Local predictive-maintenance pipeline for BLDC motor telemetry.

It uses:

- ADXL vibration time-domain features: RMS, crest factor, kurtosis, skew, peak-to-peak, slopes.
- ADXL frequency-domain features: FFT, configurable band-pass filtering, band power, spectral centroid, dominant frequency, harmonic energy.
- Electrical features: battery voltage/current, phase U/V/W currents, phase U/V/W back EMF, phase imbalance, power proxy.
- Thermal and speed features: motor temperature, RPM trends and variability.
- Multiple local models:
  - Gaussian Naive Bayes classifier.
  - k-nearest-neighbor classifier.
  - Bagged decision-tree classifier.
  - Weighted voting fault classifier.
  - Ridge regression RUL model.
  - k-nearest-neighbor RUL model.
  - Bagged decision-tree RUL model.
  - Weighted RUL ensemble.

The code intentionally runs with only `numpy`, so it can run on your current machine without installing a heavy ML stack. Later, if you want, this folder can be upgraded to scikit-learn/XGBoost/PyTorch after real labeled data is collected.

## Quick Start

From the project root:

```powershell
cd bldc_fault_ml
python train_fault_model.py --demo
```

The run creates:

- `outputs/<run-id>/model.pkl`
- `outputs/<run-id>/metrics.json`
- `outputs/<run-id>/feature_importance.csv`
- `outputs/<run-id>/report.html`
- `outputs/<run-id>/demo_dataset.csv`

Open `report.html` to see the graphs.

## A2212 12 V Report Run

This command creates a larger synthetic run constrained to the instrumented A2212 lab envelope in this project: 12 V input, battery current limited to 2 A, temperature from 30 C to 55 C, phase U/V/W back EMF, phase U/V/W current, vibration X/Y/Z, RPM, and battery measurements.

```powershell
cd bldc_fault_ml
python train_fault_model.py --demo --profile a2212_12v_2a --windows-per-class 180 --export-demo-windows 0 --dashboard-export ..\public\ml-report
```

The output folder printed at the end contains:

- `demo_dataset.csv`: all raw synthetic training windows when `--export-demo-windows 0` is used.
- `window_features.csv`: extracted window features used by the models.
- `metrics.json`: classifier and RUL metrics.
- `feature_importance.csv`: ranked ML features.
- `model.pkl`: trained model bundle used by the bridge.
- `report.html`: report-ready HTML graphs.

The dashboard export writes `public/ml-report/report.html` and `public/ml-report/latest.json`, so the current model report appears inside the deployed website after the frontend is rebuilt and redeployed.

Export a run again later without retraining:

```powershell
python export_dashboard_report.py --run outputs\<run-id> --target ..\public\ml-report
```

## Real Telemetry CSV Format

Each row should be one raw sensor sample. For FFT, vibration must be sampled much faster than the dashboard polling rate; 200 Hz to 1000 Hz is a good starting range for ADXL data.

Required columns:

```csv
timestamp_s,motor_id,temp_c,rpm,battery_voltage_v,battery_current_a,phase_u_back_emf_v,phase_v_back_emf_v,phase_w_back_emf_v,phase_u_current_a,phase_v_current_a,phase_w_current_a,vibration_x_g,vibration_y_g,vibration_z_g
```

For supervised training, add:

```csv
fault_type,rul_minutes
```

Supported demo fault classes:

- `normal`
- `bearing_wear`
- `rotor_imbalance`
- `misalignment`
- `phase_loss`
- `winding_short`
- `back_emf_anomaly`
- `demagnetization`

Train on your own CSV:

```powershell
python train_fault_model.py --csv data\motor_log.csv --sample-rate 400 --window-seconds 2
```

Run inference on a CSV after training:

```powershell
python predict_fault.py --model outputs\<run-id>\model.pkl --csv data\latest_window.csv --sample-rate 400 --window-seconds 2
```

## Important

The demo dataset is synthetic. It is useful for proving that the pipeline, signal processing, reports, and local model flow work. It does not replace labeled A2212 measurements collected from normal operation and controlled fault experiments.
