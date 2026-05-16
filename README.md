# EV Telemetry Hub

Dashboard for live STM32 BLDC motor telemetry. The STM32 sends data over USB serial to the laptop, and the laptop bridge posts each packet to the website's telemetry API.

## Data Flow

```text
STM32 USB serial -> laptop bridge script -> /api/telemetry -> dashboard polling UI
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

Run the bridge:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry
```

If you set a secret on the deployment, set `TELEMETRY_API_KEY` in Vercel and pass the same value to the bridge:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --key YOUR_SECRET
```

Dry run mode is useful for checking parsing before posting:

```powershell
python tools/stm32_serial_bridge.py --port COM5 --baud 115200 --url https://your-site.vercel.app/api/telemetry --dry-run
```

## STM32 Serial Formats

Best format: send one JSON object per line:

```json
{"front":{"rpm":2100,"temp":42.5,"inputVoltage":48.2,"currentDraw":13.4},"rear":{"rpm":2050,"temp":43.1,"inputVoltage":48.1,"currentDraw":12.9},"torqueSplit":{"front":50,"rear":50},"mlStatus":"Normal"}
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
