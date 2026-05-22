import { Fragment, useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Battery,
  CheckCircle2,
  Cpu,
  Database,
  FileText,
  Gauge,
  LineChart as LineChartIcon,
  Microscope,
  Radio,
  RotateCcw,
  Server,
  Thermometer,
  Trash2,
  Waves,
  Zap,
} from 'lucide-react';

const TELEMETRY_API_URL = import.meta.env.VITE_TELEMETRY_API_URL || '/api/telemetry';
const TELEMETRY_POLL_MS = Number(import.meta.env.VITE_TELEMETRY_POLL_MS || 250);
const STALE_AFTER_MS = Number(import.meta.env.VITE_TELEMETRY_STALE_AFTER_MS || 3000);
const MAX_HISTORY_POINTS = Number(import.meta.env.VITE_MAX_HISTORY_POINTS || 2400);
const HISTORY_STORAGE_KEY = 'ev-dashboard-telemetry-history-v2';
const INCIDENT_STORAGE_KEY = 'ev-dashboard-incident-reports-v1';
const MAX_INCIDENT_REPORTS = 48;
const MODEL_REPORT_SUMMARY_URL = '/ml-report/latest.json';

const EMPTY_MOTOR = {
  rpm: 0,
  temp: 0,
  tempSensorVoltage: undefined,
  inputVoltage: 0,
  currentDraw: 0,
  phaseU: 0,
  phaseV: 0,
  phaseW: 0,
  phaseUBackEmf: 0,
  phaseVBackEmf: 0,
  phaseWBackEmf: 0,
  vibrationX: 0,
  vibrationY: 0,
  vibrationZ: 0,
  health: 0,
  ml: { status: 'collecting' },
  signal: {},
};

const TREND_PRESETS = [
  {
    id: 'rpm',
    label: 'RPM',
    unit: 'RPM',
    series: [
      { label: 'Front RPM', key: 'frontRpm', color: '#2dd4bf' },
      { label: 'Rear RPM', key: 'rearRpm', color: '#f59e0b' },
    ],
  },
  {
    id: 'temperature',
    label: 'Temperature',
    unit: 'C',
    warningAt: 80,
    series: [
      { label: 'Front temp', key: 'frontTemp', color: '#f97316' },
      { label: 'Rear temp', key: 'rearTemp', color: '#fb7185' },
    ],
  },
  {
    id: 'current',
    label: 'Current',
    unit: 'A',
    series: [
      { label: 'Front bus current', key: 'frontCurrent', color: '#fde047' },
      { label: 'Rear bus current', key: 'rearCurrent', color: '#a3e635' },
    ],
  },
  {
    id: 'vibration',
    label: 'Vibration',
    unit: 'g',
    warningAt: 2.5,
    series: [
      { label: 'Front Z', key: 'frontVibrationZ', color: '#38bdf8' },
      { label: 'Rear Z', key: 'rearVibrationZ', color: '#c084fc' },
    ],
  },
  {
    id: 'health',
    label: 'Health',
    unit: '%',
    series: [
      { label: 'Front health', key: 'frontHealth', color: '#34d399' },
      { label: 'Rear health', key: 'rearHealth', color: '#60a5fa' },
    ],
  },
  {
    id: 'rul',
    label: 'RUL',
    unit: 'min',
    series: [
      { label: 'Front RUL', key: 'frontRul', color: '#5eead4' },
      { label: 'Rear RUL', key: 'rearRul', color: '#fda4af' },
    ],
  },
];

const INCIDENT_THRESHOLDS = {
  temperatureC: 52,
  currentA: 1.9,
  vibrationG: 0.45,
  rpmJump: 650,
  rpmRise: 900,
  healthPercent: 65,
  rulMinutes: 90,
};

const finiteNumber = (...values) => {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const titleCase = (value = '') =>
  String(value)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const formatNumber = (value, digits = 1, unit = '', active = true) => {
  if (!active || !Number.isFinite(Number(value))) return '--';
  return `${Number(value).toFixed(digits)}${unit ? ` ${unit}` : ''}`;
};

const formatWhole = (value, unit = '', active = true) => {
  if (!active || !Number.isFinite(Number(value))) return '--';
  return `${Math.round(Number(value)).toLocaleString()}${unit ? ` ${unit}` : ''}`;
};

const formatRul = (minutes) => {
  if (!Number.isFinite(Number(minutes))) return '--';
  if (minutes >= 60 * 48) return `${(minutes / (60 * 24)).toFixed(1)} d`;
  if (minutes >= 120) return `${(minutes / 60).toFixed(1)} h`;
  return `${Math.round(minutes)} min`;
};

const readStoredHistory = () => {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(-MAX_HISTORY_POINTS) : [];
  } catch {
    return [];
  }
};

const readStoredIncidentReports = () => {
  try {
    const raw = window.localStorage.getItem(INCIDENT_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(0, MAX_INCIDENT_REPORTS) : [];
  } catch {
    return [];
  }
};

const normalizeMl = (incoming = {}, current = {}) => ({
  ...current,
  ...incoming,
  status: incoming.status || current.status || 'collecting',
  confidence: finiteNumber(incoming.confidence, current.confidence),
  rulMinutes: finiteNumber(incoming.rulMinutes, incoming.estimatedRulMinutes, current.rulMinutes),
  healthIndex: finiteNumber(incoming.healthIndex, current.healthIndex),
  topFaultProbabilities: incoming.topFaultProbabilities || current.topFaultProbabilities || {},
  missingFields: incoming.missingFields || current.missingFields || [],
});

const normalizeMotor = (incoming = {}, current = EMPTY_MOTOR) => {
  const ml = normalizeMl(incoming.ml || incoming.prediction, current.ml);
  return {
    rpm: finiteNumber(incoming.rpm, incoming.speedRpm, incoming.speed, current.rpm) ?? 0,
    temp: finiteNumber(incoming.temp, incoming.temperature, incoming.motorTemp, current.temp) ?? 0,
    tempSensorVoltage: finiteNumber(
      incoming.tempSensorVoltage,
      incoming.temperatureVoltage,
      current.tempSensorVoltage,
    ),
    inputVoltage: finiteNumber(
      incoming.inputVoltage,
      incoming.batteryVoltage,
      incoming.busVoltage,
      incoming.voltage,
      current.inputVoltage,
    ) ?? 0,
    currentDraw: finiteNumber(
      incoming.currentDraw,
      incoming.batteryCurrent,
      incoming.totalCurrent,
      incoming.current,
      current.currentDraw,
    ) ?? 0,
    phaseU: finiteNumber(incoming.phaseU, incoming.phaseCurrentU, incoming.u, current.phaseU) ?? 0,
    phaseV: finiteNumber(incoming.phaseV, incoming.phaseCurrentV, incoming.v, current.phaseV) ?? 0,
    phaseW: finiteNumber(incoming.phaseW, incoming.phaseCurrentW, incoming.w, current.phaseW) ?? 0,
    phaseUBackEmf: finiteNumber(
      incoming.phaseUBackEmf,
      incoming.backEmfU,
      incoming.bemfU,
      incoming.backEmf,
      current.phaseUBackEmf,
    ) ?? 0,
    phaseVBackEmf: finiteNumber(
      incoming.phaseVBackEmf,
      incoming.backEmfV,
      incoming.bemfV,
      incoming.backEmf,
      current.phaseVBackEmf,
    ) ?? 0,
    phaseWBackEmf: finiteNumber(
      incoming.phaseWBackEmf,
      incoming.backEmfW,
      incoming.bemfW,
      incoming.backEmf,
      current.phaseWBackEmf,
    ) ?? 0,
    vibrationX: finiteNumber(incoming.vibrationX, incoming.vibration?.x, incoming.imu?.x, current.vibrationX) ?? 0,
    vibrationY: finiteNumber(incoming.vibrationY, incoming.vibration?.y, incoming.imu?.y, current.vibrationY) ?? 0,
    vibrationZ: finiteNumber(incoming.vibrationZ, incoming.vibration?.z, incoming.imu?.z, current.vibrationZ) ?? 0,
    health: finiteNumber(incoming.health, ml.healthIndex, current.health) ?? 0,
    ml,
    signal: incoming.signal || current.signal || {},
  };
};

const buildHistoryPoint = (front, rear) => ({
  timestamp: Date.now(),
  time: new Date().toLocaleTimeString([], { hour12: false }),
  frontRpm: front.rpm,
  frontTemp: front.temp,
  frontCurrent: front.currentDraw,
  frontVoltage: front.inputVoltage,
  frontVibrationZ: front.vibrationZ,
  frontHealth: front.health,
  frontRul: front.ml.rulMinutes,
  rearRpm: rear.rpm,
  rearTemp: rear.temp,
  rearCurrent: rear.currentDraw,
  rearVoltage: rear.inputVoltage,
  rearVibrationZ: rear.vibrationZ,
  rearHealth: rear.health,
  rearRul: rear.ml.rulMinutes,
});

const motorVibrationPeak = (motor) =>
  Math.max(Number(motor.vibrationX) || 0, Number(motor.vibrationY) || 0, Number(motor.vibrationZ) || 0);

const phaseSpread = (values) => {
  const phases = values.map((value) => Math.abs(Number(value) || 0));
  return Math.max(...phases) - Math.min(...phases);
};

const explainIncident = (checks, motor, previousMotor) => {
  if (checks.some((check) => check.code === 'vibration') && checks.some((check) => check.code === 'rpm-rise')) {
    return 'Vibration rose while speed was climbing. Inspect rotor balance, mount stiffness, alignment, and resonance around this operating point.';
  }
  if (checks.some((check) => check.code === 'ml-fault')) {
    return `The live fault window classified ${titleCase(motor.ml.predictedFault)} with elevated confidence. Compare the phase-current, back-EMF, and vibration signatures saved in this snapshot.`;
  }
  if (checks.some((check) => check.code === 'current') && checks.some((check) => check.code === 'temperature')) {
    return 'Current draw and temperature are both elevated. Check load, commutation quality, phase resistance, cooling, and wiring before sustained running.';
  }
  if (checks.some((check) => check.code === 'phase-current')) {
    return 'Phase-current spread is high for the saved operating point. Check phase sensing calibration and inspect for open-phase or winding asymmetry.';
  }
  if (checks.some((check) => check.code === 'back-emf')) {
    return 'Back-EMF spread is high for the saved operating point. Inspect per-phase sensing, commutation timing, magnets, and phase continuity.';
  }
  if (checks.some((check) => check.code === 'vibration')) {
    return previousMotor?.rpm < motor.rpm
      ? 'Vibration is above the report threshold during a higher-speed operating point. Review rotor balance, fasteners, and vibration spectrum.'
      : 'Vibration is above the report threshold. Review mechanical balance, bearing state, mounting, and the FFT snapshot.';
  }
  if (checks.some((check) => check.code === 'temperature')) {
    return 'Temperature exceeded the report threshold. Check the thermal sensor calibration, load current, winding heating, and cooling path.';
  }
  return 'Telemetry crossed an abnormal condition rule. Review the saved snapshot against recent trends before continuing the run.';
};

const buildIncidentCandidate = (motorId, motor, previousMotor) => {
  const vibration = motorVibrationPeak(motor);
  const rpmDelta = previousMotor ? motor.rpm - previousMotor.rpm : 0;
  const checks = [];
  if (motor.temp >= INCIDENT_THRESHOLDS.temperatureC) {
    checks.push({ code: 'temperature', severity: 'warning', label: `Temperature ${motor.temp.toFixed(1)} C` });
  }
  if (motor.currentDraw >= INCIDENT_THRESHOLDS.currentA) {
    checks.push({ code: 'current', severity: 'warning', label: `Bus current ${motor.currentDraw.toFixed(2)} A` });
  }
  if (vibration >= INCIDENT_THRESHOLDS.vibrationG) {
    checks.push({ code: 'vibration', severity: vibration >= INCIDENT_THRESHOLDS.vibrationG * 1.5 ? 'critical' : 'warning', label: `Vibration peak ${vibration.toFixed(3)} g` });
  }
  if (rpmDelta >= INCIDENT_THRESHOLDS.rpmJump) {
    checks.push({ code: 'rpm-rise', severity: 'warning', label: `RPM jump +${rpmDelta.toFixed(0)}` });
  }
  if (previousMotor && vibration >= INCIDENT_THRESHOLDS.vibrationG && rpmDelta >= INCIDENT_THRESHOLDS.rpmRise / 3) {
    checks.push({ code: 'vibration-speed', severity: 'critical', label: 'Vibration and speed rise together' });
  }
  if (motor.health > 0 && motor.health <= INCIDENT_THRESHOLDS.healthPercent) {
    checks.push({ code: 'health', severity: 'critical', label: `Health index ${motor.health.toFixed(0)}%` });
  }
  if (Number.isFinite(motor.ml.rulMinutes) && motor.ml.rulMinutes <= INCIDENT_THRESHOLDS.rulMinutes) {
    checks.push({ code: 'rul', severity: 'critical', label: `RUL ${formatRul(motor.ml.rulMinutes)}` });
  }
  if (motor.ml.status === 'ready' && motor.ml.predictedFault && motor.ml.predictedFault !== 'normal' && motor.ml.confidence >= 0.72) {
    checks.push({ code: 'ml-fault', severity: 'critical', label: `${titleCase(motor.ml.predictedFault)} ${formatNumber(motor.ml.confidence * 100, 1, '%')}` });
  }

  const currentSpread = phaseSpread([motor.phaseU, motor.phaseV, motor.phaseW]);
  if (currentSpread >= Math.max(0.35, motor.currentDraw * 0.45)) {
    checks.push({ code: 'phase-current', severity: 'warning', label: `Phase current spread ${currentSpread.toFixed(2)} A` });
  }
  const emfSpread = phaseSpread([motor.phaseUBackEmf, motor.phaseVBackEmf, motor.phaseWBackEmf]);
  if (emfSpread >= 1.1) {
    checks.push({ code: 'back-emf', severity: 'warning', label: `Back-EMF spread ${emfSpread.toFixed(2)} V` });
  }
  if (!checks.length) return null;

  const severity = checks.some((check) => check.severity === 'critical') ? 'critical' : 'warning';
  const title = checks.some((check) => check.code === 'vibration-speed')
    ? 'Vibration rise with speed increase'
    : checks.some((check) => check.code === 'ml-fault')
      ? `${titleCase(motor.ml.predictedFault)} model alert`
      : `${checks[0].label} threshold report`;
  return {
    motorId,
    title,
    severity,
    checks,
    reason: explainIncident(checks, motor, previousMotor),
    signature: `${motorId}:${checks.map((check) => check.code).sort().join('-')}:${motor.ml.predictedFault || 'telemetry'}`,
  };
};

const incidentSnapshot = (motor) => ({
  rpm: motor.rpm,
  temperatureC: motor.temp,
  voltageV: motor.inputVoltage,
  currentA: motor.currentDraw,
  vibrationX: motor.vibrationX,
  vibrationY: motor.vibrationY,
  vibrationZ: motor.vibrationZ,
  phaseCurrents: { u: motor.phaseU, v: motor.phaseV, w: motor.phaseW },
  phaseBackEmf: { u: motor.phaseUBackEmf, v: motor.phaseVBackEmf, w: motor.phaseWBackEmf },
  ml: {
    status: motor.ml.status,
    fault: motor.ml.predictedFault,
    confidence: motor.ml.confidence,
    rulMinutes: motor.ml.rulMinutes,
  },
  fft: motor.signal?.fft
    ? {
        dominantHz: motor.signal.fft.dominantHz,
        peakAmplitude: motor.signal.fft.peakAmplitude,
      }
    : undefined,
});

const statusTone = (status) => {
  if (status === 'ready') return 'text-emerald-300 border-emerald-700 bg-emerald-950/50';
  if (status === 'waiting_for_channels') return 'text-amber-200 border-amber-700 bg-amber-950/40';
  return 'text-zinc-300 border-zinc-700 bg-zinc-900';
};

const Panel = ({ children, className = '' }) => (
  <section className={`rounded-lg border border-zinc-800 bg-zinc-950/90 p-4 md:p-5 ${className}`}>
    {children}
  </section>
);

const Metric = ({ icon: Icon, label, value, tone = 'text-white', note }) => (
  <div className="min-h-24 border border-zinc-800 bg-zinc-900/65 p-3">
    <div className="flex items-center gap-2 text-xs text-zinc-400">
      <Icon className="h-4 w-4" />
      <span>{label}</span>
    </div>
    <div className={`mt-3 break-words font-mono text-xl ${tone}`}>{value}</div>
    {note && <div className="mt-1 text-xs text-zinc-500">{note}</div>}
  </div>
);

const HealthBar = ({ value, active }) => {
  const score = clamp(Number(value) || 0, 0, 100);
  const color = score >= 85 ? 'bg-emerald-400' : score >= 60 ? 'bg-amber-400' : 'bg-rose-400';
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-zinc-400">
        <span>Health index</span>
        <span className="font-mono text-zinc-200">{active ? `${score.toFixed(0)}%` : '--'}</span>
      </div>
      <div className="h-2 overflow-hidden bg-zinc-800">
        <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${active ? score : 0}%` }} />
      </div>
    </div>
  );
};

const PhaseTable = ({ title, rows, unit, active }) => (
  <div className="border border-zinc-800">
    <div className="border-b border-zinc-800 px-3 py-2 text-xs text-zinc-400">{title}</div>
    <div className="grid grid-cols-3 divide-x divide-zinc-800">
      {rows.map((row) => (
        <div key={row.label} className="px-3 py-3">
          <div className="text-xs text-zinc-500">{row.label}</div>
          <div className="mt-1 font-mono text-sm text-zinc-100">{formatNumber(row.value, 2, unit, active)}</div>
        </div>
      ))}
    </div>
  </div>
);

const MiniBars = ({ title, rows, unit, active }) => (
  <div className="border border-zinc-800">
    <div className="border-b border-zinc-800 px-3 py-2 text-xs text-zinc-400">{title}</div>
    <div className="grid grid-cols-3 gap-2 px-3 py-3">
      {rows.map((row) => {
        const height = clamp((Math.abs(Number(row.value) || 0) / Math.max(row.max, 1e-9)) * 100, 4, 100);
        return (
          <div key={row.label}>
            <div className="flex h-28 items-end bg-zinc-900">
              <div className="w-full bg-cyan-400/85" style={{ height: `${active ? height : 0}%` }} />
            </div>
            <div className="mt-2 flex items-center justify-between gap-2 text-xs">
              <span className="text-zinc-500">{row.label}</span>
              <span className="font-mono text-zinc-100">{formatNumber(row.value, 3, unit, active)}</span>
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

const ProbabilityBars = ({ probabilities }) => {
  const entries = Object.entries(probabilities || {});
  if (!entries.length) {
    return <div className="text-sm text-zinc-500">No probability output yet.</div>;
  }

  return (
    <div className="space-y-3">
      {entries.map(([label, probability]) => (
        <div key={label}>
          <div className="mb-1 flex items-center justify-between gap-3 text-xs">
            <span className="truncate text-zinc-300">{titleCase(label)}</span>
            <span className="font-mono text-zinc-400">{(Number(probability) * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-zinc-800">
            <div className="h-full bg-cyan-400" style={{ width: `${clamp(Number(probability) * 100, 0, 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
};

const ChartFrame = ({ children, empty, height = 280 }) => (
  <div className="min-h-0 overflow-hidden border border-zinc-800 bg-[#08090b]" style={{ height }}>
    {empty ? <div className="flex h-full items-center justify-center px-6 text-center text-sm text-zinc-500">{empty}</div> : children}
  </div>
);

const LineChart = ({ data, preset }) => {
  const points = data.filter((point) =>
    preset.series.some((series) => Number.isFinite(Number(point[series.key]))),
  );
  if (points.length < 2) {
    return <ChartFrame empty="Waiting for live samples." />;
  }

  const width = 940;
  const height = 280;
  const left = 58;
  const top = 22;
  const right = 22;
  const bottom = 42;
  const values = preset.series.flatMap((series) =>
    points.map((point) => Number(point[series.key])).filter(Number.isFinite),
  );
  if (preset.warningAt !== undefined) values.push(preset.warningAt);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const pad = Math.max((maxValue - minValue) * 0.15, 0.5);
  const yMin = preset.id === 'health' ? 0 : minValue - pad;
  const yMax = preset.id === 'health' ? 100 : maxValue + pad;
  const range = Math.max(yMax - yMin, 1);
  const x = (index) => left + (index / Math.max(points.length - 1, 1)) * (width - left - right);
  const y = (value) => top + (1 - (value - yMin) / range) * (height - top - bottom);
  const yTicks = [yMax, (yMin + yMax) / 2, yMin];
  const xTicks = [0, Math.floor((points.length - 1) / 2), points.length - 1];

  return (
    <ChartFrame>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full font-mono" role="img">
        <rect width={width} height={height} fill="#08090b" />
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke="#27272a" strokeDasharray="5 5" />
            <text x={left - 10} y={y(tick) + 4} fill="#a1a1aa" fontSize="11" textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        {preset.warningAt !== undefined && (
          <g>
            <line
              x1={left}
              x2={width - right}
              y1={y(preset.warningAt)}
              y2={y(preset.warningAt)}
              stroke="#fb7185"
              strokeDasharray="7 5"
            />
            <text x={width - right} y={y(preset.warningAt) - 7} fill="#fb7185" fontSize="11" textAnchor="end">
              limit
            </text>
          </g>
        )}
        <line x1={left} x2={left} y1={top} y2={height - bottom} stroke="#52525b" />
        <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke="#52525b" />
        {preset.series.map((series) => (
          <polyline
            key={series.key}
            fill="none"
            stroke={series.color}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2.5"
            points={points.map((point, index) => `${x(index)},${y(Number(point[series.key]))}`).join(' ')}
          />
        ))}
        {xTicks.map((index) => (
          <text
            key={index}
            x={x(index)}
            y={height - 14}
            fill="#71717a"
            fontSize="11"
            textAnchor={index === 0 ? 'start' : index === points.length - 1 ? 'end' : 'middle'}
          >
            {points[index]?.time}
          </text>
        ))}
      </svg>
    </ChartFrame>
  );
};

const SpectrumChart = ({ fft }) => {
  const points = Array.isArray(fft?.points) ? fft.points : [];
  if (points.length < 2) {
    return <ChartFrame empty="FFT output appears after the bridge has a complete vibration window." height={310} />;
  }

  const width = 940;
  const height = 310;
  const left = 58;
  const top = 20;
  const right = 22;
  const bottom = 42;
  const maxHz = Math.max(...points.map((point) => Number(point.hz) || 0), 1);
  const maxAmp = Math.max(...points.map((point) => Number(point.amp) || 0), 1e-6);
  const x = (hz) => left + (hz / maxHz) * (width - left - right);
  const y = (amp) => top + (1 - amp / maxAmp) * (height - top - bottom);
  const yTicks = [maxAmp, maxAmp / 2, 0];
  const xTicks = [0, maxHz / 2, maxHz];

  return (
    <ChartFrame height={310}>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full font-mono" role="img">
        <rect width={width} height={height} fill="#08090b" />
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke="#27272a" strokeDasharray="5 5" />
            <text x={left - 10} y={y(tick) + 4} fill="#a1a1aa" fontSize="11" textAnchor="end">
              {tick.toFixed(3)}
            </text>
          </g>
        ))}
        <line x1={left} x2={left} y1={top} y2={height - bottom} stroke="#52525b" />
        <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke="#52525b" />
        <polyline
          fill="none"
          stroke="#22d3ee"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2.5"
          points={points.map((point) => `${x(Number(point.hz))},${y(Number(point.amp))}`).join(' ')}
        />
        {xTicks.map((tick) => (
          <text
            key={tick}
            x={x(tick)}
            y={height - 14}
            fill="#71717a"
            fontSize="11"
            textAnchor={tick === 0 ? 'start' : tick === maxHz ? 'end' : 'middle'}
          >
            {tick.toFixed(0)} Hz
          </text>
        ))}
      </svg>
    </ChartFrame>
  );
};

const HorizontalBars = ({ title, rows, valueLabel = (value) => value.toFixed(3), empty }) => {
  const safeRows = (rows || []).filter((row) => Number.isFinite(Number(row.value)));
  if (!safeRows.length) {
    return <ChartFrame empty={empty || 'No chart data exported.'} height={360} />;
  }
  const max = Math.max(...safeRows.map((row) => Number(row.value)), 1e-9);
  return (
    <div className="border border-zinc-800 bg-[#08090b] p-4">
      <div className="mb-4 text-sm font-medium text-white">{title}</div>
      <div className="space-y-3">
        {safeRows.map((row, index) => (
          <div key={row.label}>
            <div className="mb-1 flex items-center justify-between gap-3 text-xs">
              <span className="truncate text-zinc-300">{titleCase(row.label)}</span>
              <span className="font-mono text-zinc-400">{valueLabel(Number(row.value))}</span>
            </div>
            <div className="h-2 bg-zinc-800">
              <div
                className={index % 3 === 0 ? 'h-full bg-cyan-400' : index % 3 === 1 ? 'h-full bg-emerald-400' : 'h-full bg-fuchsia-400'}
                style={{ width: `${clamp((Number(row.value) / max) * 100, 2, 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ConfusionMatrixChart = ({ labels = [], matrix = [] }) => {
  if (!labels.length || !matrix.length) {
    return <ChartFrame empty="No confusion matrix exported." height={470} />;
  }
  const max = Math.max(...matrix.flat().map((value) => Number(value) || 0), 1);
  return (
    <div className="overflow-x-auto border border-zinc-800 bg-[#08090b] p-4">
      <div className="mb-4 text-sm font-medium text-white">Fault Confusion Matrix</div>
      <div className="grid min-w-[620px] gap-1" style={{ gridTemplateColumns: `180px repeat(${labels.length}, minmax(52px, 1fr))` }}>
        <div />
        {labels.map((label) => (
          <div key={label} className="flex min-h-16 items-end justify-center pb-2 text-[11px] text-zinc-400">
            <span className="-rotate-45 whitespace-nowrap">{titleCase(label)}</span>
          </div>
        ))}
        {labels.map((label, rowIndex) => (
          <Fragment key={label}>
            <div className="flex items-center pr-3 text-xs text-zinc-300">{titleCase(label)}</div>
            {labels.map((_, columnIndex) => {
              const value = Number(matrix[rowIndex]?.[columnIndex]) || 0;
              const intensity = value / max;
              return (
                <div
                  key={`${label}-${columnIndex}`}
                  className="flex aspect-square items-center justify-center border border-zinc-900 font-mono text-sm"
                  style={{
                    backgroundColor: `rgba(34, 211, 238, ${0.08 + intensity * 0.82})`,
                    color: intensity > 0.45 ? '#ecfeff' : '#a1a1aa',
                  }}
                >
                  {value}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
};

const ScatterChart = ({ points = [] }) => {
  const values = points.filter((point) =>
    Number.isFinite(Number(point.actualMinutes)) && Number.isFinite(Number(point.predictedMinutes)),
  );
  if (values.length < 2) {
    return <ChartFrame empty="No RUL test points exported." height={360} />;
  }
  const width = 680;
  const height = 360;
  const left = 64;
  const top = 24;
  const right = 24;
  const bottom = 52;
  const limit = Math.max(
    ...values.flatMap((point) => [Number(point.actualMinutes), Number(point.predictedMinutes)]),
    1,
  );
  const x = (value) => left + (value / limit) * (width - left - right);
  const y = (value) => top + (1 - value / limit) * (height - top - bottom);
  const ticks = [0, limit / 2, limit];
  return (
    <div className="border border-zinc-800 bg-[#08090b] p-4">
      <div className="mb-4 text-sm font-medium text-white">RUL Actual vs Predicted</div>
      <ChartFrame height={360}>
        <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full font-mono" role="img">
          <rect width={width} height={height} fill="#08090b" />
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke="#27272a" strokeDasharray="5 5" />
              <line x1={x(tick)} x2={x(tick)} y1={top} y2={height - bottom} stroke="#27272a" strokeDasharray="5 5" />
              <text x={left - 10} y={y(tick) + 4} fill="#71717a" fontSize="11" textAnchor="end">{Math.round(tick)}</text>
              <text x={x(tick)} y={height - 25} fill="#71717a" fontSize="11" textAnchor="middle">{Math.round(tick)}</text>
            </g>
          ))}
          <line x1={left} y1={y(0)} x2={x(limit)} y2={top} stroke="#fb7185" strokeWidth="2" strokeDasharray="7 6" />
          <line x1={left} x2={left} y1={top} y2={height - bottom} stroke="#52525b" />
          <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke="#52525b" />
          {values.map((point, index) => (
            <circle
              key={`${point.fault}-${index}`}
              cx={x(Number(point.actualMinutes))}
              cy={y(Number(point.predictedMinutes))}
              r="4"
              fill={index % 2 ? '#22d3ee' : '#34d399'}
              opacity="0.82"
            />
          ))}
          <text x={width / 2} y={height - 8} fill="#a1a1aa" fontSize="12" textAnchor="middle">Actual minutes</text>
          <text x="16" y={height / 2} fill="#a1a1aa" fontSize="12" textAnchor="middle" transform={`rotate(-90,16,${height / 2})`}>Predicted minutes</text>
        </svg>
      </ChartFrame>
    </div>
  );
};

const MlState = ({ ml }) => {
  if (ml.status === 'ready') {
    return (
      <div className="grid gap-3 sm:grid-cols-3">
        <div>
          <div className="text-xs text-zinc-500">Fault class</div>
          <div className="mt-1 text-base text-zinc-100">{titleCase(ml.predictedFault)}</div>
        </div>
        <div>
          <div className="text-xs text-zinc-500">Confidence</div>
          <div className="mt-1 font-mono text-base text-zinc-100">{formatNumber(ml.confidence * 100, 1, '%')}</div>
        </div>
        <div>
          <div className="text-xs text-zinc-500">Maintenance window</div>
          <div className="mt-1 font-mono text-base text-zinc-100">{formatRul(ml.rulMinutes)}</div>
        </div>
      </div>
    );
  }

  if (ml.status === 'waiting_for_channels') {
    const missing = (ml.missingFields || []).slice(0, 4).map(titleCase).join(', ');
    return (
      <div className="text-sm text-amber-100">
        ML window buffered. Waiting for {missing || 'the remaining model channels'}.
      </div>
    );
  }

  return (
    <div className="text-sm text-zinc-400">
      Collecting {ml.sampleCount || 0}/{ml.requiredSamples || '--'} samples for the next model window.
    </div>
  );
};

const ModelReportPanel = ({ reportState }) => {
  const report = reportState.data;
  const classifier = report?.classifierTest || {};
  const rul = report?.regressorTest?.ensemble_log_rul || {};
  const profile = report?.metadata?.synthetic_profile;
  const classRows = (report?.classDistribution || []).map((item) => ({ label: item.label, value: item.windows }));
  const featureRows = (report?.topFeatures || []).slice(0, 12).map((item) => ({ label: item.feature, value: item.importance }));

  return (
    <Panel>
      <div className="mb-4 flex flex-col gap-3 border-b border-zinc-800 pb-4 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2 text-base font-semibold text-white">
            <LineChartIcon className="h-5 w-5 text-fuchsia-300" />
            ML Training Report
          </div>
          <div className="mt-1 text-sm text-zinc-500">{profile?.title || 'Latest exported classifier and RUL evidence'}</div>
        </div>
        {report?.generatedAt && (
          <div className="border border-zinc-800 px-3 py-2 text-xs text-zinc-400">
            Exported <span className="font-mono text-zinc-200">{report.generatedAt}</span>
          </div>
        )}
      </div>

      {reportState.status === 'ready' ? (
        <>
          <div className="mb-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <Metric
              icon={Activity}
              label="Test accuracy"
              value={formatNumber(classifier.test_accuracy * 100, 1, '%')}
              tone="text-emerald-100"
            />
            <Metric
              icon={Activity}
              label="Macro F1"
              value={formatNumber(classifier.test_macro_f1, 3)}
              tone="text-cyan-100"
            />
            <Metric
              icon={Database}
              label="Training windows"
              value={formatWhole(report.metadata?.windows)}
              tone="text-white"
            />
            <Metric
              icon={Gauge}
              label="RUL MAE"
              value={formatNumber(rul.mae, 1, 'min')}
              tone="text-amber-100"
            />
            <Metric
              icon={Waves}
              label="Feature count"
              value={formatWhole(report.metadata?.features)}
              tone="text-sky-100"
            />
          </div>
          <div className="mb-4 grid gap-4 xl:grid-cols-2">
            <HorizontalBars title="Class Distribution" rows={classRows} valueLabel={(value) => `${Math.round(value)} windows`} />
            <HorizontalBars title="Top Feature Importance" rows={featureRows} />
          </div>
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(360px,0.8fr)]">
            <ConfusionMatrixChart labels={report.metadata?.fault_classes} matrix={report.confusionMatrix} />
            <ScatterChart points={report.rulScatter} />
          </div>
        </>
      ) : (
        <div className="flex min-h-44 items-center justify-center border border-zinc-800 bg-zinc-900/50 px-5 text-center text-sm text-zinc-500">
          {reportState.status === 'loading' ? 'Loading the exported ML report.' : 'No exported ML report is deployed yet.'}
        </div>
      )}
    </Panel>
  );
};

const MlLiveSignalPanel = ({ motors, selectedMotor, onSelect, active }) => {
  const motor = motors.find((item) => item.id === selectedMotor) || motors[0];
  return (
    <Panel>
      <div className="mb-4 flex flex-col gap-3 border-b border-zinc-800 pb-4 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2 text-base font-semibold text-white">
            <Waves className="h-5 w-5 text-sky-300" />
            Live ML Signal Views
          </div>
          <div className="mt-1 text-sm text-zinc-500">Current signal inputs and bridge FFT window</div>
        </div>
        <div className="flex gap-2">
          {motors.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={`min-h-9 border px-3 text-sm ${
                item.id === selectedMotor
                  ? 'border-sky-500 bg-sky-950/60 text-sky-100'
                  : 'border-zinc-800 bg-zinc-900 text-zinc-300 hover:border-zinc-600'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(360px,0.8fr)]">
        <div>
          <SpectrumChart fft={motor.data.signal?.fft} />
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Metric icon={Gauge} label="RPM input" value={formatWhole(motor.data.rpm, 'RPM', active)} tone="text-teal-100" />
            <Metric icon={Waves} label="FFT dominant" value={formatNumber(motor.data.signal?.fft?.dominantHz, 2, 'Hz')} tone="text-sky-100" />
          </div>
        </div>
        <div className="grid gap-3">
          <MiniBars
            title="Vibration input axes"
            unit="g"
            active={active}
            rows={[
              { label: 'X', value: motor.data.vibrationX, max: Math.max(INCIDENT_THRESHOLDS.vibrationG, motorVibrationPeak(motor.data)) },
              { label: 'Y', value: motor.data.vibrationY, max: Math.max(INCIDENT_THRESHOLDS.vibrationG, motorVibrationPeak(motor.data)) },
              { label: 'Z', value: motor.data.vibrationZ, max: Math.max(INCIDENT_THRESHOLDS.vibrationG, motorVibrationPeak(motor.data)) },
            ]}
          />
          <PhaseTable
            title="ML phase current inputs"
            active={active}
            unit="A"
            rows={[
              { label: 'U', value: motor.data.phaseU },
              { label: 'V', value: motor.data.phaseV },
              { label: 'W', value: motor.data.phaseW },
            ]}
          />
        </div>
      </div>
    </Panel>
  );
};

const MotorPanel = ({ title, motor, active }) => {
  const warning = active && (motor.temp >= 80 || Math.max(motor.vibrationX, motor.vibrationY, motor.vibrationZ) >= 2.5);
  return (
    <Panel className={warning ? 'border-rose-800/80' : ''}>
      <div className="mb-5 flex flex-col gap-3 border-b border-zinc-800 pb-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <div className={`border p-2 ${warning ? 'border-rose-700 bg-rose-950/60 text-rose-200' : 'border-teal-800 bg-teal-950/50 text-teal-200'}`}>
            <RotateCcw className={`h-5 w-5 ${active && motor.rpm > 0 ? 'animate-[spin_3s_linear_infinite]' : ''}`} />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">{title}</h2>
            <div className={`mt-2 inline-flex border px-2 py-1 text-xs ${statusTone(motor.ml.status)}`}>
              {titleCase(motor.ml.status)}
            </div>
          </div>
        </div>
        <div className="w-full sm:w-48">
          <HealthBar value={motor.health} active={active} />
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <Metric icon={Gauge} label="Speed" value={formatWhole(motor.rpm, 'RPM', active)} tone="text-teal-100" />
        <Metric
          icon={Thermometer}
          label="Motor temperature"
          value={formatNumber(motor.temp, 1, 'C', active)}
          tone={motor.temp >= 80 ? 'text-rose-200' : 'text-orange-100'}
          note={motor.tempSensorVoltage !== undefined ? `ADC input ${motor.tempSensorVoltage.toFixed(2)} V` : undefined}
        />
        <Metric icon={Battery} label="Bus voltage" value={formatNumber(motor.inputVoltage, 2, 'V', active)} tone="text-sky-100" />
        <Metric icon={Zap} label="Bus current" value={formatNumber(motor.currentDraw, 2, 'A', active)} tone="text-lime-100" />
      </div>

      <div className="mt-4 grid gap-3">
        <PhaseTable
          title="Phase currents"
          active={active}
          unit="A"
          rows={[
            { label: 'U', value: motor.phaseU },
            { label: 'V', value: motor.phaseV },
            { label: 'W', value: motor.phaseW },
          ]}
        />
        <PhaseTable
          title="Phase back EMF"
          active={active}
          unit="V"
          rows={[
            { label: 'U', value: motor.phaseUBackEmf },
            { label: 'V', value: motor.phaseVBackEmf },
            { label: 'W', value: motor.phaseWBackEmf },
          ]}
        />
      </div>

      <div className="mt-4 border border-zinc-800 px-3 py-3">
        <div className="mb-3 flex items-center gap-2 text-xs text-zinc-400">
          <Waves className="h-4 w-4" />
          <span>Vibration axes</span>
        </div>
        <div className="grid grid-cols-3 gap-3 font-mono text-sm">
          <div><span className="text-zinc-500">X</span> <span className="text-zinc-100">{formatNumber(motor.vibrationX, 3, 'g', active)}</span></div>
          <div><span className="text-zinc-500">Y</span> <span className="text-zinc-100">{formatNumber(motor.vibrationY, 3, 'g', active)}</span></div>
          <div><span className="text-zinc-500">Z</span> <span className="text-zinc-100">{formatNumber(motor.vibrationZ, 3, 'g', active)}</span></div>
        </div>
      </div>
    </Panel>
  );
};

const ReportSnapshot = ({ report, selected, onSelect }) => (
  <button
    onClick={() => onSelect(report.id)}
    className={`w-full border px-3 py-3 text-left transition-colors ${
      selected ? 'border-rose-600 bg-rose-950/35' : 'border-zinc-800 bg-zinc-900/55 hover:border-zinc-600'
    }`}
  >
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="text-sm font-medium text-zinc-100">{report.title}</div>
        <div className="mt-1 text-xs text-zinc-500">{report.motorLabel} | {new Date(report.createdAt).toLocaleTimeString()}</div>
      </div>
      <div className={`border px-2 py-1 text-[11px] ${report.severity === 'critical' ? 'border-rose-700 text-rose-200' : 'border-amber-700 text-amber-200'}`}>
        {titleCase(report.severity)}
      </div>
    </div>
    <div className="mt-2 line-clamp-2 text-xs text-zinc-400">{report.reason}</div>
  </button>
);

const IncidentReportDetail = ({ report }) => {
  if (!report) {
    return (
      <div className="flex min-h-72 items-center justify-center border border-zinc-800 bg-[#08090b] p-6 text-center text-sm text-zinc-500">
        Reports appear when abnormal telemetry or a non-normal ML window crosses a report rule.
      </div>
    );
  }
  const snapshot = report.snapshot;
  return (
    <div className="border border-zinc-800 bg-[#08090b] p-4">
      <div className="flex flex-col gap-3 border-b border-zinc-800 pb-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="text-lg font-semibold text-white">{report.title}</div>
          <div className="mt-1 text-sm text-zinc-400">{report.reason}</div>
        </div>
        <div className="text-xs text-zinc-500 sm:text-right">
          <div>{report.motorLabel}</div>
          <div className="mt-1 font-mono text-zinc-300">{new Date(report.createdAt).toLocaleString()}</div>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {report.checks.map((check) => (
          <span key={check.label} className={`border px-2 py-1 text-xs ${check.severity === 'critical' ? 'border-rose-800 bg-rose-950/40 text-rose-100' : 'border-amber-800 bg-amber-950/35 text-amber-100'}`}>
            {check.label}
          </span>
        ))}
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric icon={Gauge} label="RPM snapshot" value={formatWhole(snapshot.rpm, 'RPM')} tone="text-teal-100" />
        <Metric icon={Thermometer} label="Temperature" value={formatNumber(snapshot.temperatureC, 1, 'C')} tone="text-orange-100" />
        <Metric icon={Zap} label="Bus current" value={formatNumber(snapshot.currentA, 2, 'A')} tone="text-lime-100" />
        <Metric icon={Waves} label="Vibration peak" value={formatNumber(Math.max(snapshot.vibrationX, snapshot.vibrationY, snapshot.vibrationZ), 3, 'g')} tone="text-sky-100" />
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        <PhaseTable
          title="Saved phase currents"
          active
          unit="A"
          rows={[
            { label: 'U', value: snapshot.phaseCurrents.u },
            { label: 'V', value: snapshot.phaseCurrents.v },
            { label: 'W', value: snapshot.phaseCurrents.w },
          ]}
        />
        <PhaseTable
          title="Saved phase back EMF"
          active
          unit="V"
          rows={[
            { label: 'U', value: snapshot.phaseBackEmf.u },
            { label: 'V', value: snapshot.phaseBackEmf.v },
            { label: 'W', value: snapshot.phaseBackEmf.w },
          ]}
        />
        <div className="border border-zinc-800 p-3">
          <div className="text-xs text-zinc-400">Saved model state</div>
          <div className="mt-3 text-sm text-zinc-100">{titleCase(snapshot.ml.fault || snapshot.ml.status)}</div>
          <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-xs text-zinc-400">
            <span>Confidence</span><span className="text-right text-zinc-100">{formatNumber(snapshot.ml.confidence * 100, 1, '%')}</span>
            <span>RUL</span><span className="text-right text-zinc-100">{formatRul(snapshot.ml.rulMinutes)}</span>
            <span>FFT peak</span><span className="text-right text-zinc-100">{formatNumber(snapshot.fft?.dominantHz, 2, 'Hz')}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const IncidentReportsPanel = ({ reports, selectedId, onSelect, onClear }) => {
  const selected = reports.find((report) => report.id === selectedId) || reports[0];
  return (
    <Panel>
      <div className="mb-4 flex flex-col gap-3 border-b border-zinc-800 pb-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-base font-semibold text-white">
            <FileText className="h-5 w-5 text-rose-300" />
            Generated Reports
          </div>
          <div className="mt-1 text-sm text-zinc-500">{reports.length} abnormal snapshots retained locally</div>
        </div>
        <button
          onClick={onClear}
          disabled={!reports.length}
          className="flex h-9 w-9 items-center justify-center border border-zinc-800 bg-zinc-900 text-zinc-300 transition-colors hover:border-rose-700 hover:text-rose-200 disabled:cursor-not-allowed disabled:opacity-40"
          title="Clear generated reports"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
      <div className="grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="max-h-[620px] space-y-2 overflow-y-auto pr-1">
          {reports.length ? reports.map((report) => (
            <ReportSnapshot key={report.id} report={report} selected={selected?.id === report.id} onSelect={onSelect} />
          )) : (
            <div className="border border-zinc-800 bg-zinc-900/40 p-4 text-sm text-zinc-500">No abnormal snapshot reports yet.</div>
          )}
        </div>
        <IncidentReportDetail report={selected} />
      </div>
    </Panel>
  );
};

export default function App() {
  const [frontMotor, setFrontMotor] = useState(EMPTY_MOTOR);
  const [rearMotor, setRearMotor] = useState(EMPTY_MOTOR);
  const [history, setHistory] = useState(readStoredHistory);
  const [incidentReports, setIncidentReports] = useState(readStoredIncidentReports);
  const [connected, setConnected] = useState(false);
  const [hasTelemetry, setHasTelemetry] = useState(false);
  const [streamMode, setStreamMode] = useState('Waiting for bridge');
  const [lastPacketAt, setLastPacketAt] = useState(null);
  const [source, setSource] = useState('stm32');
  const [activeView, setActiveView] = useState('console');
  const [selectedTrendId, setSelectedTrendId] = useState('rpm');
  const [selectedSignalMotor, setSelectedSignalMotor] = useState('front');
  const [selectedReportId, setSelectedReportId] = useState(null);
  const [modelReport, setModelReport] = useState({ status: 'loading', data: null });

  const frontRef = useRef(frontMotor);
  const rearRef = useRef(rearMotor);
  const reportCooldownRef = useRef({});
  const lastSignature = useRef('');

  useEffect(() => {
    frontRef.current = frontMotor;
  }, [frontMotor]);

  useEffect(() => {
    rearRef.current = rearMotor;
  }, [rearMotor]);

  useEffect(() => {
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(-MAX_HISTORY_POINTS)));
    } catch {
      // Live data should continue even when browser storage is unavailable.
    }
  }, [history]);

  useEffect(() => {
    try {
      window.localStorage.setItem(INCIDENT_STORAGE_KEY, JSON.stringify(incidentReports.slice(0, MAX_INCIDENT_REPORTS)));
    } catch {
      // Reports remain available until the page closes when storage is blocked.
    }
  }, [incidentReports]);

  useEffect(() => {
    const applyTelemetry = (payload, updatedAt) => {
      const data = payload?.data || payload;
      if (!data) return;

      const nextFront = normalizeMotor(data.front || data.frontMotor, frontRef.current);
      const nextRear = normalizeMotor(data.rear || data.rearMotor, rearRef.current);
      const signature = JSON.stringify({ updatedAt, nextFront, nextRear });
      if (signature === lastSignature.current) return;
      lastSignature.current = signature;

      const candidates = [
        buildIncidentCandidate('front', nextFront, frontRef.current),
        buildIncidentCandidate('rear', nextRear, rearRef.current),
      ].filter(Boolean);
      const createdAt = updatedAt || data.receivedAt || data.timestamp || new Date().toISOString();
      const newReports = candidates
        .filter((candidate) => {
          const previousReportAt = reportCooldownRef.current[candidate.signature] || 0;
          const now = Date.now();
          if (now - previousReportAt < 15000) return false;
          reportCooldownRef.current[candidate.signature] = now;
          return true;
        })
        .map((candidate) => {
          const motor = candidate.motorId === 'front' ? nextFront : nextRear;
          return {
            id: `${candidate.motorId}-${Date.now()}-${candidate.signature}`,
            createdAt,
            motorId: candidate.motorId,
            motorLabel: candidate.motorId === 'front' ? 'Front drive' : 'Rear drive',
            title: candidate.title,
            severity: candidate.severity,
            reason: candidate.reason,
            checks: candidate.checks,
            snapshot: incidentSnapshot(motor),
          };
        });

      setFrontMotor(nextFront);
      setRearMotor(nextRear);
      setHistory((current) => [...current, buildHistoryPoint(nextFront, nextRear)].slice(-MAX_HISTORY_POINTS));
      if (newReports.length) {
        setIncidentReports((current) => [...newReports, ...current].slice(0, MAX_INCIDENT_REPORTS));
        setSelectedReportId(newReports[0].id);
      }
      setLastPacketAt(updatedAt || data.receivedAt || data.timestamp || new Date().toISOString());
      setSource(data.source || 'stm32');
      setHasTelemetry(true);
      setConnected(true);
      setStreamMode('Live');
    };

    const pollTelemetry = async () => {
      try {
        const response = await fetch(`${TELEMETRY_API_URL}?t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Telemetry API returned ${response.status}`);
        const payload = await response.json();
        if (!payload.data) {
          setConnected(false);
          setHasTelemetry(false);
          setLastPacketAt(null);
          setStreamMode('Waiting for bridge');
          return;
        }

        const updatedMs = payload.updatedAt ? new Date(payload.updatedAt).getTime() : Date.now();
        if (Date.now() - updatedMs > STALE_AFTER_MS) {
          setConnected(false);
          setStreamMode('Packet stale');
          return;
        }
        applyTelemetry(payload.data, payload.updatedAt);
      } catch {
        setConnected(false);
        setStreamMode('API unavailable');
      }
    };

    pollTelemetry();
    const interval = window.setInterval(pollTelemetry, TELEMETRY_POLL_MS);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    let active = true;
    const loadModelReport = async () => {
      try {
        const response = await fetch(`${MODEL_REPORT_SUMMARY_URL}?t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Model report returned ${response.status}`);
        const data = await response.json();
        if (active) setModelReport({ status: 'ready', data });
      } catch {
        if (active) setModelReport({ status: 'missing', data: null });
      }
    };
    loadModelReport();
    return () => {
      active = false;
    };
  }, []);

  const selectedTrend = TREND_PRESETS.find((preset) => preset.id === selectedTrendId) || TREND_PRESETS[0];
  const selectedSignal = selectedSignalMotor === 'front' ? frontMotor : rearMotor;
  const motors = [
    { id: 'front', label: 'Front drive', data: frontMotor },
    { id: 'rear', label: 'Rear drive', data: rearMotor },
  ];
  const historySpan = history.length > 1 ? Math.max(0, history.at(-1).timestamp - history[0].timestamp) : 0;
  const ingestHz = historySpan > 0 ? ((history.length - 1) * 1000) / historySpan : 0;
  const busVoltageValues = [frontMotor.inputVoltage, rearMotor.inputVoltage].filter((value) => value > 0);
  const busVoltage = busVoltageValues.length
    ? busVoltageValues.reduce((sum, value) => sum + value, 0) / busVoltageValues.length
    : 0;
  const totalCurrent = frontMotor.currentDraw + rearMotor.currentDraw;
  const estimatedPowerKw = (busVoltage * totalCurrent) / 1000;
  const readyModels = motors.filter((motor) => motor.data.ml.status === 'ready').length;

  const clearHistory = () => {
    setHistory([]);
    try {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // Clearing memory is enough when storage is blocked.
    }
  };

  const clearIncidentReports = () => {
    setIncidentReports([]);
    setSelectedReportId(null);
    try {
      window.localStorage.removeItem(INCIDENT_STORAGE_KEY);
    } catch {
      // In-memory reports were cleared already.
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0d10] px-4 py-5 text-zinc-200 md:px-6 lg:px-8">
      <header className="mx-auto flex max-w-[1480px] flex-col gap-4 border-b border-zinc-800 pb-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm text-zinc-400">
            <Cpu className="h-4 w-4 text-teal-300" />
            <span>Dual BLDC predictive maintenance</span>
          </div>
          <h1 className="text-2xl font-semibold text-white md:text-3xl">EV Motor Health Console</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setActiveView('console')}
            className={`flex min-h-10 items-center gap-2 border px-3 text-sm transition-colors ${
              activeView === 'console'
                ? 'border-teal-600 bg-teal-950/45 text-teal-100'
                : 'border-zinc-800 bg-zinc-950 text-zinc-300 hover:border-zinc-600'
            }`}
            title="Open live motor console"
          >
            <Gauge className="h-4 w-4" />
            <span>Console</span>
          </button>
          <button
            onClick={() => setActiveView('ml')}
            className={`flex min-h-10 items-center gap-2 border px-3 text-sm transition-colors ${
              activeView === 'ml'
                ? 'border-fuchsia-600 bg-fuchsia-950/45 text-fuchsia-100'
                : 'border-zinc-800 bg-zinc-950 text-zinc-300 hover:border-zinc-600'
            }`}
            title="Open ML training graphs"
          >
            <Microscope className="h-4 w-4" />
            <span>ML</span>
          </button>
          <div className="flex min-h-10 items-center gap-2 border border-zinc-800 bg-zinc-950 px-3 text-sm">
            <Server className="h-4 w-4 text-zinc-400" />
            <span className="text-zinc-400">Bridge</span>
            {connected ? (
              <span className="flex items-center gap-1 text-emerald-300"><CheckCircle2 className="h-4 w-4" /> Live</span>
            ) : (
              <span className="flex items-center gap-1 text-rose-300"><AlertTriangle className="h-4 w-4" /> {streamMode}</span>
            )}
          </div>
          <div className="flex min-h-10 items-center gap-2 border border-zinc-800 bg-zinc-950 px-3 text-sm text-zinc-400">
            <Radio className="h-4 w-4 text-cyan-300" />
            <span className="font-mono text-zinc-200">{TELEMETRY_POLL_MS} ms</span>
          </div>
        </div>
      </header>

      {activeView === 'console' ? (
      <>
      <main className="mx-auto mt-5 grid max-w-[1480px] gap-4 xl:grid-cols-12">
        <div className="grid gap-4 xl:col-span-8">
          <Panel className="grid gap-3 md:grid-cols-4">
            <Metric icon={Battery} label="Bus average" value={formatNumber(busVoltage, 2, 'V', hasTelemetry)} tone="text-sky-100" />
            <Metric icon={Zap} label="System current" value={formatNumber(totalCurrent, 2, 'A', hasTelemetry)} tone="text-lime-100" />
            <Metric icon={Activity} label="Estimated power" value={formatNumber(estimatedPowerKw, 2, 'kW', hasTelemetry)} tone="text-amber-100" />
            <Metric icon={Database} label="Stored rate" value={formatNumber(ingestHz, 1, 'Hz', history.length > 1)} tone="text-cyan-100" />
          </Panel>

          {hasTelemetry ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <MotorPanel title="Front Drive" motor={frontMotor} active={hasTelemetry} />
              <MotorPanel title="Rear Drive" motor={rearMotor} active={hasTelemetry} />
            </div>
          ) : (
            <Panel className="flex min-h-[470px] items-center justify-center text-center">
              <div>
                <Server className="mx-auto mb-4 h-9 w-9 text-zinc-500" />
                <h2 className="text-xl font-semibold text-white">Waiting for STM32 UART telemetry</h2>
                <p className="mt-2 max-w-md text-sm text-zinc-400">
                  The console will populate after the laptop bridge posts its first packet.
                </p>
              </div>
            </Panel>
          )}

          <Panel>
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="flex items-center gap-2 text-base font-semibold text-white">
                  <LineChartIcon className="h-5 w-5 text-cyan-300" />
                  Live Trends
                </div>
                <div className="mt-1 text-sm text-zinc-500">{history.length.toLocaleString()} samples retained in this browser</div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {TREND_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => setSelectedTrendId(preset.id)}
                    className={`min-h-9 border px-3 text-sm transition-colors ${
                      preset.id === selectedTrend.id
                        ? 'border-cyan-500 bg-cyan-950/60 text-cyan-100'
                        : 'border-zinc-800 bg-zinc-900 text-zinc-300 hover:border-zinc-600'
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
                <button
                  onClick={clearHistory}
                  disabled={!history.length}
                  className="flex h-9 w-9 items-center justify-center border border-zinc-800 bg-zinc-900 text-zinc-300 transition-colors hover:border-rose-700 hover:text-rose-200 disabled:cursor-not-allowed disabled:opacity-40"
                  title="Clear trend history"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
            <div className="mb-3 flex flex-wrap gap-4 text-xs text-zinc-400">
              {selectedTrend.series.map((series) => (
                <span key={series.key} className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5" style={{ backgroundColor: series.color }} />
                  {series.label}
                </span>
              ))}
            </div>
            <LineChart data={history} preset={selectedTrend} />
          </Panel>
        </div>

        <aside className="grid gap-4 xl:col-span-4">
          <Panel>
            <div className="mb-4 flex items-center justify-between gap-3 border-b border-zinc-800 pb-4">
              <div>
                <div className="flex items-center gap-2 text-base font-semibold text-white">
                  <Activity className="h-5 w-5 text-emerald-300" />
                  Predictive Maintenance
                </div>
                <div className="mt-1 text-sm text-zinc-500">{readyModels}/2 model outputs ready</div>
              </div>
              <div className="text-right text-xs text-zinc-500">
                <div>Last packet</div>
                <div className="mt-1 font-mono text-zinc-300">
                  {lastPacketAt ? new Date(lastPacketAt).toLocaleTimeString() : '--'}
                </div>
              </div>
            </div>

            <div className="space-y-5">
              {motors.map((motor) => (
                <div key={motor.id} className="border-b border-zinc-800 pb-5 last:border-b-0 last:pb-0">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div className="font-medium text-white">{motor.label}</div>
                    <div className={`border px-2 py-1 text-xs ${statusTone(motor.data.ml.status)}`}>
                      {titleCase(motor.data.ml.status)}
                    </div>
                  </div>
                  <MlState ml={motor.data.ml} />
                  {motor.data.ml.status === 'ready' && (
                    <div className="mt-4">
                      <ProbabilityBars probabilities={motor.data.ml.topFaultProbabilities} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Panel>

          <Panel>
            <div className="mb-4 flex flex-col gap-3 border-b border-zinc-800 pb-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="flex items-center gap-2 text-base font-semibold text-white">
                  <Waves className="h-5 w-5 text-sky-300" />
                  Band-Pass FFT
                </div>
                <div className="mt-1 text-sm text-zinc-500">
                  {selectedSignal.signal?.fft?.bandpassHz
                    ? `${selectedSignal.signal.fft.bandpassHz[0]}-${selectedSignal.signal.fft.bandpassHz[1]} Hz`
                    : 'Awaiting bridge spectrum'}
                </div>
              </div>
              <div className="flex gap-2">
                {motors.map((motor) => (
                  <button
                    key={motor.id}
                    onClick={() => setSelectedSignalMotor(motor.id)}
                    className={`min-h-9 border px-3 text-sm ${
                      selectedSignalMotor === motor.id
                        ? 'border-sky-500 bg-sky-950/60 text-sky-100'
                        : 'border-zinc-800 bg-zinc-900 text-zinc-300 hover:border-zinc-600'
                    }`}
                  >
                    {motor.id === 'front' ? 'Front' : 'Rear'}
                  </button>
                ))}
              </div>
            </div>
            <SpectrumChart fft={selectedSignal.signal?.fft} />
            <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
              <div className="border border-zinc-800 px-3 py-2">
                <div className="text-xs text-zinc-500">Dominant frequency</div>
                <div className="mt-1 font-mono text-zinc-100">{formatNumber(selectedSignal.signal?.fft?.dominantHz, 2, 'Hz')}</div>
              </div>
              <div className="border border-zinc-800 px-3 py-2">
                <div className="text-xs text-zinc-500">Peak amplitude</div>
                <div className="mt-1 font-mono text-zinc-100">{formatNumber(selectedSignal.signal?.fft?.peakAmplitude, 4, 'g')}</div>
              </div>
            </div>
          </Panel>

          <Panel>
            <div className="mb-4 flex items-center gap-2 text-base font-semibold text-white">
              <Radio className="h-5 w-5 text-amber-300" />
              Link State
            </div>
            <div className="grid gap-3 text-sm sm:grid-cols-2 xl:grid-cols-1">
              <div className="border border-zinc-800 px-3 py-3">
                <div className="text-xs text-zinc-500">Source</div>
                <div className="mt-1 break-words font-mono text-zinc-100">{source}</div>
              </div>
              <div className="border border-zinc-800 px-3 py-3">
                <div className="text-xs text-zinc-500">Telemetry API</div>
                <div className="mt-1 break-all font-mono text-zinc-100">{TELEMETRY_API_URL}</div>
              </div>
              <div className="border border-zinc-800 px-3 py-3">
                <div className="text-xs text-zinc-500">Freshness limit</div>
                <div className="mt-1 font-mono text-zinc-100">{STALE_AFTER_MS.toLocaleString()} ms</div>
              </div>
              <div className="border border-zinc-800 px-3 py-3">
                <div className="text-xs text-zinc-500">Trend capacity</div>
                <div className="mt-1 font-mono text-zinc-100">{MAX_HISTORY_POINTS.toLocaleString()} samples</div>
              </div>
            </div>
          </Panel>
        </aside>
      </main>
      <div className="mx-auto mt-4 max-w-[1480px]">
        <IncidentReportsPanel
          reports={incidentReports}
          selectedId={selectedReportId}
          onSelect={setSelectedReportId}
          onClear={clearIncidentReports}
        />
      </div>
      </>
      ) : (
        <main className="mx-auto mt-5 grid max-w-[1480px] gap-4">
          <MlLiveSignalPanel
            motors={motors}
            selectedMotor={selectedSignalMotor}
            onSelect={setSelectedSignalMotor}
            active={hasTelemetry}
          />
          <ModelReportPanel reportState={modelReport} />
        </main>
      )}
    </div>
  );
}
