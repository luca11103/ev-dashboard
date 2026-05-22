# BLDC Health Console Setup Guide

## What Is Wired Now

```text
STM32 UART raw samples
  -> tools/stm32_serial_bridge.py
  -> optional local model.pkl window scoring
  -> /api/telemetry latest-packet API
  -> React dual-motor dashboard
```

The React app is a visualization surface. The Python bridge keeps the sample buffer needed for band-pass vibration processing, FFT, fault classification, and remaining useful life output.

## Firmware Packet Contract

Prefer one JSON object per raw sample. Key-value UART lines also work, but JSON is easier to extend safely.

```json
{
  "front": {
    "rpm": 2120,
    "temp": 44.6,
    "inputVoltage": 48.1,
    "currentDraw": 13.2,
    "phaseU": 4.1,
    "phaseV": 4.0,
    "phaseW": 4.2,
    "phaseUBackEmf": 21.8,
    "phaseVBackEmf": 21.6,
    "phaseWBackEmf": 21.9,
    "vibrationX": 0.12,
    "vibrationY": 0.08,
    "vibrationZ": 0.10
  },
  "rear": {
    "rpm": 2070,
    "temp": 45.0,
    "inputVoltage": 48.0,
    "currentDraw": 12.7,
    "phaseU": 3.9,
    "phaseV": 4.1,
    "phaseW": 4.0,
    "phaseUBackEmf": 21.2,
    "phaseVBackEmf": 21.1,
    "phaseWBackEmf": 21.4,
    "vibrationX": 0.11,
    "vibrationY": 0.09,
    "vibrationZ": 0.12
  }
}
```

For the current STM32 printout, the bridge accepts `AZ` as `vibrationZ`, `TEMP_V` as raw `tempSensorVoltage`, and `HALL` as the printed RPM value. Rename those firmware fields before the full model path is used so units are explicit.

## Sampling Rules

- Dashboard refresh and raw sampling are different rates.
- Feed the ML bridge raw ADXL samples at the same rate passed with `--sample-rate`.
- Start with 400 Hz raw vibration sampling and 2 second model windows to match the bundled demo model.
- Do not classify from the current commutation loop with six `HAL_Delay(20)` calls. That path is only about 8 raw loops per second and is too slow for the existing FFT feature bands.
- Keep the bridge API post rate lower than the raw UART rate. The bridge buffers every parsed UART line before throttling web posts.

## Bridge Commands

Telemetry bring-up:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --motor front --dry-run
```

`--dry-run` parses UART without posting. For live local API testing, run a serverless dev environment that serves `api/telemetry.js`, or point the bridge at the deployed API while Vite serves the UI.

Live model path:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --model bldc_fault_ml\outputs\20260516_111827\model.pkl --sample-rate 400
```

The dashboard receives these ML fields when a complete feature window is available:

```json
{
  "front": {
    "ml": {
      "status": "ready",
      "predictedFault": "normal",
      "confidence": 0.92,
      "rulMinutes": 6300,
      "healthIndex": 99,
      "topFaultProbabilities": { "normal": 0.92 }
    },
    "signal": {
      "fft": {
        "axis": "bandpass_vibration_magnitude",
        "dominantHz": 34.5,
        "points": [{ "hz": 0, "amp": 0 }]
      }
    }
  }
}
```

## STM32 Notes For The Shared Code

- `TEMP_V` is ADC voltage, not motor temperature in degrees C. Convert it with the chosen sensor calibration before training and inference.
- If your ADC captures PWM phase terminal voltage rather than back EMF, name and label it explicitly and retrain the feature schema instead of silently treating it as back EMF.
- `HAL_ADC_Start()` is called once while ADC continuous conversion is disabled. Restart conversion for each sample or configure continuous/DMA acquisition.
- RPM from one Hall rising edge needs the correct pulses-per-revolution or pole-pair scaling for the motor and Hall strategy.
- A production controller should separate commutation timing, fast sensor sampling, and UART publishing. Delays inside the commutation loop will fight real-time data capture.

## Production Boundary

The current `/api/telemetry` endpoint stores the latest packet for one dashboard stream. A multi-user deployment needs a real identity and data layer:

1. User authentication with server-issued sessions.
2. Device ownership and per-user authorization on read and write APIs.
3. Durable time-series storage for telemetry and model outputs.
4. Device keys or certificates for each bridge/controller.
5. Audit logs, alert rules, retention, and transport hardening.

Do not replace that with a browser-only login screen. It would look like access control while leaving motor data exposed.
