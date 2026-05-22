import { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Battery,
  CheckCircle2,
  Cpu,
  Database,
  Gauge,
  LineChart as LineChartIcon,
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

export default function App() {
  const [frontMotor, setFrontMotor] = useState(EMPTY_MOTOR);
  const [rearMotor, setRearMotor] = useState(EMPTY_MOTOR);
  const [history, setHistory] = useState(readStoredHistory);
  const [connected, setConnected] = useState(false);
  const [hasTelemetry, setHasTelemetry] = useState(false);
  const [streamMode, setStreamMode] = useState('Waiting for bridge');
  const [lastPacketAt, setLastPacketAt] = useState(null);
  const [source, setSource] = useState('stm32');
  const [selectedTrendId, setSelectedTrendId] = useState('rpm');
  const [selectedSignalMotor, setSelectedSignalMotor] = useState('front');

  const frontRef = useRef(frontMotor);
  const rearRef = useRef(rearMotor);
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
    const applyTelemetry = (payload, updatedAt) => {
      const data = payload?.data || payload;
      if (!data) return;

      const nextFront = normalizeMotor(data.front || data.frontMotor, frontRef.current);
      const nextRear = normalizeMotor(data.rear || data.rearMotor, rearRef.current);
      const signature = JSON.stringify({ updatedAt, nextFront, nextRear });
      if (signature === lastSignature.current) return;
      lastSignature.current = signature;

      setFrontMotor(nextFront);
      setRearMotor(nextRear);
      setHistory((current) => [...current, buildHistoryPoint(nextFront, nextRear)].slice(-MAX_HISTORY_POINTS));
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
    </div>
  );
}
