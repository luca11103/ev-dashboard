# EV Telemetry Hub

Dashboard for live STM32 BLDC motor telemetry. The STM32 sends data over USB serial to the laptop, and the laptop bridge posts each packet to the website's telemetry API. The bridge can also buffer high-rate raw samples, run the local BLDC fault model, and send fault/RUL/FFT output to the dashboard.

## Data Flow

```text
STM32 USB serial -> laptop bridge script -> live ML windowing -> /api/telemetry -> dashboard UI
```

The browser reads the latest packet with `GET /api/telemetry`. The laptop sends packets with `POST /api/telemetry`.

## Run The Dashboard

```powershell
npm install
npm run dev
```

For the hosted site, deploy the repo and use the deployed API URL in the bridge, for example:

```text
https://your-site.vercel.app/api/telemetry
```

When running the dashboard locally but posting to the hosted API, create `.env.local`:

```text
VITE_TELEMETRY_API_URL=https://your-site.vercel.app/api/telemetry
```

## Laptop STM32 Bridge

Install the serial dependency:

```powershell
python -m pip install -r tools/requirements.txt
```

Run the bridge for telemetry only:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry
```

Run it with the bundled local ML model:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --model bldc_fault_ml\outputs\<run-id>\model.pkl --sample-rate 400
```

ML scoring stays in `waiting_for_channels` until the UART stream contains the electrical, thermal, RPM, and vibration channels expected by the model. `--allow-partial-ml` exists for bring-up, but zero-filled phase or battery channels should not be treated as a validated maintenance prediction.

If you set a secret on the deployment, set `TELEMETRY_API_KEY` in Vercel and pass the same value to the bridge:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --key YOUR_SECRET
```

Dry run mode is useful for checking parsing before posting:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --dry-run
```

## STM32 Serial Formats

Best format: send one JSON object per raw sample:

```json
{"front":{"rpm":2100,"temp":42.5,"inputVoltage":48.2,"currentDraw":13.4,"phaseU":4.1,"phaseV":4.0,"phaseW":4.2,"phaseUBackEmf":21.8,"phaseVBackEmf":21.6,"phaseWBackEmf":21.9,"vibrationX":0.12,"vibrationY":0.08,"vibrationZ":0.10},"rear":{"rpm":2050,"temp":43.1,"inputVoltage":48.1,"currentDraw":12.9,"phaseU":3.9,"phaseV":4.1,"phaseW":4.0,"phaseUBackEmf":21.1,"phaseVBackEmf":21.0,"phaseWBackEmf":21.3,"vibrationX":0.11,"vibrationY":0.09,"vibrationZ":0.12}}
```

Key-value lines also work:

```text
front_rpm=2100,front_temp=42.5,front_voltage=48.2,front_current=13.4,rear_rpm=2050,rear_temp=43.1,rear_voltage=48.1,rear_current=12.9
```

For one motor, unprefixed values are applied to the motor selected with `--motor`:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --motor front
```

```text
rpm=2100,temp=42.5,voltage=48.2,current=13.4,phaseU=4.1,phaseV=4.0,phaseW=4.2,vibrationX=0.12,vibrationY=0.08,vibrationZ=0.10,backEmf=22.4,health=98
```

Raw CSV works too. By default the order is:

```text
rpm,temp,inputVoltage,currentDraw,phaseU,phaseV,phaseW,vibrationX,vibrationY,vibrationZ,backEmf,health
```

You can override that with `--csv-fields`.

For FFT and fault classification, send raw vibration samples at the sample rate passed to the bridge. A slow dashboard packet rate is fine; a slow raw sensor stream is not. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for the firmware packet contract and the next production steps.

## ML Report On The Website

Train and export the A2212 synthetic report from the ML folder:

```powershell
cd bldc_fault_ml
python train_fault_model.py --demo --profile a2212_12v_2a --windows-per-class 180 --export-demo-windows 0 --dashboard-export ..\public\ml-report
```

That run creates the full local report under `bldc_fault_ml\outputs\<run-id>\report.html` and exports the latest report assets under `public\ml-report` for Vite/Vercel.

To publish an already-trained run again:

```powershell
cd bldc_fault_ml
python export_dashboard_report.py --run outputs\<run-id> --target ..\public\ml-report
```

The live console keeps exported ML graphs under the header `ML` view. Its `Generated Reports` panel is separate: it saves telemetry snapshots when an explainable abnormality rule fires, such as high vibration with a speed rise, high temperature/current, phase spread, low RUL/health, or a confident non-normal live ML window.
