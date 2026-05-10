import { useCallback, useState, useEffect, useRef } from 'react';
import { 
  Activity, Thermometer, Zap, Waves, AlertTriangle, 
  Cpu, Gauge, Battery, Server, CheckCircle2, RotateCcw,
  Sparkles, X, FileText, LineChart as LineChartIcon, Database, Trash2
} from 'lucide-react';

// --- Utility Components ---
const SVGLineChart = ({ data, series, unit, warningAt }) => {
  const activeSeries = series || [];
  const points = (data || []).filter((point) =>
    activeSeries.some((item) => Number.isFinite(Number(point[item.dataKey]))),
  );

  if (points.length < 2) {
    return <div className="text-slate-500 flex items-center justify-center h-full">Waiting for stored samples...</div>;
  }

  const values = activeSeries.flatMap((item) =>
    points.map((point) => Number(point[item.dataKey])).filter(Number.isFinite),
  );
  const maxVal = Math.max(...values, warningAt ?? -Infinity);
  const minVal = Math.min(...values, warningAt ?? Infinity);
  const pad = (maxVal - minVal) * 0.15 || 2;
  const maxPlot = maxVal + pad;
  const minPlot = Math.max(0, minVal - pad);
  const range = maxPlot - minPlot || 1;
  const chartWidth = Math.max(980, points.length * 16);
  const chartHeight = 300;
  const left = 64;
  const right = 28;
  const top = 24;
  const bottom = 44;
  const plotWidth = chartWidth - left - right;
  const plotHeight = chartHeight - top - bottom;
  const yFor = (value) => top + (1 - ((value - minPlot) / range)) * plotHeight;
  const xFor = (index) => left + (index / (points.length - 1)) * plotWidth;
  const yTicks = [maxPlot, (maxPlot + minPlot) / 2, minPlot];
  const xTicks = [0, Math.floor((points.length - 1) / 2), points.length - 1];

  const stats = activeSeries.map((item) => {
    const itemValues = points.map((point) => Number(point[item.dataKey])).filter(Number.isFinite);
    const latest = itemValues[itemValues.length - 1];
    const min = Math.min(...itemValues);
    const max = Math.max(...itemValues);
    const avg = itemValues.reduce((sum, value) => sum + value, 0) / itemValues.length;
    return { ...item, latest, min, max, avg };
  });

  return (
    <div className="h-full flex flex-col gap-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stats.map((item) => (
          <div key={item.dataKey} className="rounded-lg bg-slate-950/70 border border-slate-800 p-3">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
              {item.label}
            </div>
            <div className="mt-2 text-xl font-mono text-white">{item.latest.toFixed(1)} <span className="text-xs text-slate-500">{unit}</span></div>
            <div className="mt-1 text-[11px] text-slate-500 font-mono">min {item.min.toFixed(1)} / avg {item.avg.toFixed(1)} / max {item.max.toFixed(1)}</div>
          </div>
        ))}
      </div>

      <div className="flex-1 min-h-0 overflow-x-auto overflow-y-hidden rounded-lg border border-slate-800 bg-slate-950/70">
        <svg width={chartWidth} height={chartHeight} className="block font-mono">
          <rect x="0" y="0" width={chartWidth} height={chartHeight} fill="#020617" />
          {yTicks.map((tick) => (
            <g key={tick}>
              <line x1={left} x2={chartWidth - right} y1={yFor(tick)} y2={yFor(tick)} stroke="#1e293b" strokeDasharray="5 5" />
              <text x={left - 10} y={yFor(tick) + 4} textAnchor="end" fill="#94a3b8" fontSize="11">{tick.toFixed(1)}</text>
            </g>
          ))}
          {warningAt !== undefined && (
            <g>
              <line x1={left} x2={chartWidth - right} y1={yFor(warningAt)} y2={yFor(warningAt)} stroke="#fb7185" strokeDasharray="8 5" />
              <text x={chartWidth - right - 8} y={yFor(warningAt) - 6} textAnchor="end" fill="#fb7185" fontSize="11">limit {warningAt} {unit}</text>
            </g>
          )}
          <line x1={left} x2={left} y1={top} y2={chartHeight - bottom} stroke="#475569" />
          <line x1={left} x2={chartWidth - right} y1={chartHeight - bottom} y2={chartHeight - bottom} stroke="#475569" />

          {activeSeries.map((item) => {
            const path = points.map((point, index) => `${xFor(index)},${yFor(Number(point[item.dataKey]))}`).join(' ');
            return (
              <polyline
                key={item.dataKey}
                points={path}
                fill="none"
                stroke={item.color}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            );
          })}

          {xTicks.map((index) => (
            <text key={index} x={xFor(index)} y={chartHeight - 16} textAnchor={index === 0 ? 'start' : index === points.length - 1 ? 'end' : 'middle'} fill="#64748b" fontSize="11">
              {points[index]?.time}
            </text>
          ))}
        </svg>
      </div>
    </div>
  );
};

const generateGeminiContent = async (prompt, retries = 5) => {
  const apiKey = ""; // Populated securely at runtime
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;
  const delays = [1000, 2000, 4000, 8000, 16000];

  for (let i = 0; i <= retries; i++) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          systemInstruction: { parts: [{ text: "You are an expert EV hardware engineer and TinyML specialist monitoring a dual BLDC motor system." }] }
        })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated.";
    } catch (error) {
      if (i === retries) throw error;
      await new Promise(r => setTimeout(r, delays[i]));
    }
  }
};

const Card = ({ children, className = '' }) => (
  <div className={`bg-slate-900 border border-slate-800 rounded-xl shadow-lg p-5 ${className}`}>
    {children}
  </div>
);

const ProgressBar = ({ value, max, colorClass, label, unit }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="w-full mt-2">
      <div className="flex justify-between text-xs mb-1 text-slate-400">
        <span>{label}</span>
        <span className="font-mono">{value.toFixed(1)} {unit}</span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div 
          className={`h-full ${colorClass} transition-all duration-500 ease-out`} 
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

const TELEMETRY_API_URL = import.meta.env.VITE_TELEMETRY_API_URL || '/api/telemetry';
const TELEMETRY_POLL_MS = Number(import.meta.env.VITE_TELEMETRY_POLL_MS || 750);
const STALE_AFTER_MS = Number(import.meta.env.VITE_TELEMETRY_STALE_AFTER_MS || 5000);
const HISTORY_STORAGE_KEY = 'ev-dashboard-telemetry-history-v1';
const MAX_HISTORY_POINTS = Number(import.meta.env.VITE_MAX_HISTORY_POINTS || 5000);

const readNumber = (...values) => {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
};

const withMotorDefaults = (incoming = {}, current = {}) => ({
  rpm: readNumber(incoming.rpm, incoming.speedRpm, incoming.speed, current.rpm) ?? 0,
  temp: readNumber(incoming.temp, incoming.temperature, incoming.motorTemp, current.temp) ?? 0,
  inputVoltage: readNumber(incoming.inputVoltage, incoming.voltage, incoming.busVoltage, current.inputVoltage) ?? 0,
  currentDraw: readNumber(incoming.currentDraw, incoming.current, incoming.totalCurrent, current.currentDraw) ?? 0,
  phaseU: readNumber(incoming.phaseU, incoming.u, incoming.phaseCurrentU, current.phaseU) ?? 0,
  phaseV: readNumber(incoming.phaseV, incoming.v, incoming.phaseCurrentV, current.phaseV) ?? 0,
  phaseW: readNumber(incoming.phaseW, incoming.w, incoming.phaseCurrentW, current.phaseW) ?? 0,
  vibrationX: readNumber(incoming.vibrationX, incoming.vibration?.x, incoming.imu?.x, current.vibrationX) ?? 0,
  vibrationY: readNumber(incoming.vibrationY, incoming.vibration?.y, incoming.imu?.y, current.vibrationY) ?? 0,
  vibrationZ: readNumber(incoming.vibrationZ, incoming.vibration?.z, incoming.imu?.z, current.vibrationZ) ?? 0,
  backEmf: readNumber(incoming.backEmf, incoming.bemf, current.backEmf) ?? 0,
  health: readNumber(incoming.health, current.health) ?? 100,
});

const buildHistoryPoint = (front, rear) => ({
  timestamp: Date.now(),
  time: new Date().toLocaleTimeString([], { hour12: false }),
  frontTemp: front.temp,
  frontCurrent: front.currentDraw,
  frontRpm: front.rpm,
  frontInputVoltage: front.inputVoltage,
  frontBackEmf: front.backEmf,
  frontPhaseU: front.phaseU,
  frontPhaseV: front.phaseV,
  frontPhaseW: front.phaseW,
  frontVibrationX: front.vibrationX,
  frontVibrationY: front.vibrationY,
  frontVibrationZ: front.vibrationZ,
  rearTemp: rear.temp,
  rearCurrent: rear.currentDraw,
  rearRpm: rear.rpm,
  rearInputVoltage: rear.inputVoltage,
  rearBackEmf: rear.backEmf,
  rearPhaseU: rear.phaseU,
  rearPhaseV: rear.phaseV,
  rearPhaseW: rear.phaseW,
  rearVibrationX: rear.vibrationX,
  rearVibrationY: rear.vibrationY,
  rearVibrationZ: rear.vibrationZ,
});

const loadStoredHistory = () => {
  try {
    const rawHistory = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!rawHistory) return [];
    const parsed = JSON.parse(rawHistory);
    return Array.isArray(parsed) ? parsed.slice(-MAX_HISTORY_POINTS) : [];
  } catch {
    return [];
  }
};

// --- Main Application ---
export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [hasTelemetry, setHasTelemetry] = useState(false);
  const [streamMode, setStreamMode] = useState('Waiting for Pico');
  const [lastPacketAt, setLastPacketAt] = useState(null);
  const [mlStatus, setMlStatus] = useState("Adaptive AWD Active");
  const [torqueSplit, setTorqueSplit] = useState({ front: 50, rear: 50 });
  
  const [frontMotor, setFrontMotor] = useState({
    rpm: 0, temp: 0, inputVoltage: 0, currentDraw: 0,
    phaseU: 0, phaseV: 0, phaseW: 0,
    vibrationX: 0.1, vibrationY: 0.1, vibrationZ: 0.1,
    backEmf: 0, health: 0
  });

  const [rearMotor, setRearMotor] = useState({
    rpm: 0, temp: 0, inputVoltage: 0, currentDraw: 0,
    phaseU: 0, phaseV: 0, phaseW: 0,
    vibrationX: 0.1, vibrationY: 0.2, vibrationZ: 0.1,
    backEmf: 0, health: 0
  });

  const [alerts, setAlerts] = useState([
    { id: 1, time: new Date().toLocaleTimeString(), type: 'info', msg: 'System Initialized. Awaiting Pico stream.' }
  ]);

  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [aiReportContent, setAiReportContent] = useState(null);
  const [aiAnalysisState, setAiAnalysisState] = useState({ loading: false, alertId: null, content: null });

  // History state for graphs
  const [history, setHistory] = useState(loadStoredHistory);
  const [graphModal, setGraphModal] = useState(null); 
  
  const frontMotorRef = useRef(frontMotor);
  const rearMotorRef = useRef(rearMotor);
  const lastPayloadSignature = useRef('');

  useEffect(() => {
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(-MAX_HISTORY_POINTS)));
    } catch {
      // Browser storage can be full or disabled; live telemetry should continue either way.
    }
  }, [history]);

  useEffect(() => {
    frontMotorRef.current = frontMotor;
  }, [frontMotor]);

  useEffect(() => {
    rearMotorRef.current = rearMotor;
  }, [rearMotor]);

  useEffect(() => {
    const applyTelemetry = (payload, updatedAt) => {
      const data = payload?.data || payload;
      if (!data) return false;

      const nextFront = withMotorDefaults(data.front || data.frontMotor, frontMotorRef.current);
      const nextRear = withMotorDefaults(data.rear || data.rearMotor, rearMotorRef.current);
      const signature = JSON.stringify({ updatedAt, nextFront, nextRear, torqueSplit: data.torqueSplit });

      if (signature === lastPayloadSignature.current) return true;
      lastPayloadSignature.current = signature;

      setFrontMotor(nextFront);
      setRearMotor(nextRear);

      if (data.torqueSplit) {
        const front = readNumber(data.torqueSplit.front, data.frontTorque, data.torqueFront);
        const rear = readNumber(data.torqueSplit.rear, data.rearTorque, data.torqueRear);
        if (front !== undefined && rear !== undefined) setTorqueSplit({ front, rear });
      }

      if (data.mlStatus || data.controllerMode) {
        setMlStatus(data.mlStatus || data.controllerMode);
      }

      setHistory(prev => [...prev, buildHistoryPoint(nextFront, nextRear)].slice(-MAX_HISTORY_POINTS));
      setLastPacketAt(updatedAt || data.receivedAt || data.timestamp || new Date().toISOString());
      setHasTelemetry(true);
      setIsConnected(true);
      setStreamMode('Pico live');
      return true;
    };

    const pollTelemetry = async () => {
      try {
        const response = await fetch(`${TELEMETRY_API_URL}?t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Telemetry API returned ${response.status}`);
        const payload = await response.json();

        if (!payload.data) {
          setIsConnected(false);
          setHasTelemetry(false);
          setLastPacketAt(null);
          setStreamMode('Waiting for Pico');
          return;
        }

        const updatedTime = payload.updatedAt ? new Date(payload.updatedAt).getTime() : Date.now();
        if (Date.now() - updatedTime > STALE_AFTER_MS) {
          setIsConnected(false);
          setHasTelemetry(false);
          setStreamMode('Pico stale');
          return;
        }

        applyTelemetry(payload.data, payload.updatedAt);
      } catch {
        setIsConnected(false);
        setHasTelemetry(false);
        setStreamMode('API unavailable');
      }
    };

    pollTelemetry();
    const interval = setInterval(pollTelemetry, TELEMETRY_POLL_MS);
    return () => clearInterval(interval);
  }, []);

  const addAlert = useCallback((type, msg) => {
    const newAlert = { id: Date.now(), time: new Date().toLocaleTimeString(), type, msg };
    setAlerts(prev => [newAlert, ...prev].slice(0, 8));
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (rearMotor.temp > 85 && alerts.length < 5) {
        addAlert('warning', 'Rear motor temperature exceeding optimal bounds (85°C+)');
      }
      if (frontMotor.vibrationX > 3.0) {
        addAlert('critical', 'High vibration anomaly detected on Front Drive IMU');
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [addAlert, alerts.length, frontMotor.temp, rearMotor.temp, frontMotor.vibrationX]);

  const handleGenerateAiReport = async () => {
    setIsGeneratingReport(true);
    setAiReportContent(null);
    const prompt = `Analyze the current state of the dual-motor EV system.
    Front Motor: ${frontMotor.rpm.toFixed(0)} RPM, ${frontMotor.temp.toFixed(1)}°C, Peak Vibration: ${frontMotor.vibrationX.toFixed(2)}g, Current: ${frontMotor.currentDraw.toFixed(1)}A
    Rear Motor: ${rearMotor.rpm.toFixed(0)} RPM, ${rearMotor.temp.toFixed(1)}°C, Peak Vibration: ${rearMotor.vibrationX.toFixed(2)}g, Current: ${rearMotor.currentDraw.toFixed(1)}A
    Torque Split: Front ${torqueSplit.front.toFixed(0)}% / Rear ${torqueSplit.rear.toFixed(0)}%
    Recent Alerts: ${alerts.map(a => a.msg).join(', ')}

    Provide a brief, professional diagnostic report. Format it nicely using markdown. Include:
    1. A 1-sentence overall system health summary.
    2. Potential anomalies to watch.
    3. Recommended maintenance actions based on the current data.`;

    try {
      const result = await generateGeminiContent(prompt);
      setAiReportContent(result);
    } catch {
      setAiReportContent("Failed to generate report due to an API error. Please try again.");
    }
    setIsGeneratingReport(false);
  };

  const handleAnalyzeAlert = async (alert) => {
    setAiAnalysisState({ loading: true, alertId: alert.id, content: null });
    const prompt = `An alert occurred in our EV motor telemetry: "${alert.msg}".
    Context: Front Motor Temp is ${frontMotor.temp.toFixed(1)}°C, Vibration is ${frontMotor.vibrationX.toFixed(2)}g. Rear Motor Temp is ${rearMotor.temp.toFixed(1)}°C, Vibration is ${rearMotor.vibrationX.toFixed(2)}g.

    As an EV hardware specialist, explain in 2-3 short sentences what physical issues might be causing this in the BLDC motor or ESC hardware, and suggest an immediate hardware check.`;

    try {
      const result = await generateGeminiContent(prompt);
      setAiAnalysisState({ loading: false, alertId: alert.id, content: result });
    } catch {
      setAiAnalysisState({ loading: false, alertId: alert.id, content: "Analysis failed. Please check network connection." });
    }
  };

  const formatValue = (value, digits = 1, unit = '') => {
    if (!hasTelemetry) return '--';
    return `${value.toFixed(digits)}${unit ? ` ${unit}` : ''}`;
  };

  const formatWhole = (value) => {
    if (!hasTelemetry) return '--';
    return value.toFixed(0);
  };

  const clearStoredHistory = () => {
    setHistory([]);
    try {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // Ignore storage failures; clearing in-memory history is still useful.
    }
    setGraphModal(null);
    addAlert('info', 'Stored telemetry history was reset.');
  };

  const openTrend = (title, unit, series, warningAt) => {
    setGraphModal({ title, unit, series, warningAt });
  };

  const openSingleTrend = (title, dataKey, color, unit, warningAt) => {
    openTrend(title, unit, [{ label: title, dataKey, color }], warningAt);
  };

  const historyStart = history[0]?.timestamp ? new Date(history[0].timestamp).toLocaleTimeString() : '--';
  const historyEnd = history.at(-1)?.timestamp ? new Date(history.at(-1).timestamp).toLocaleTimeString() : '--';

  const renderMotorPanel = (title, motorId, data) => {
    const isWarning = data.temp > 80 || data.vibrationX > 2.5;
    
    return (
      <Card className={`relative overflow-hidden ${isWarning ? 'border-rose-900/50 bg-rose-950/10' : ''}`}>
        <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${isWarning ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
              <RotateCcw className={`w-6 h-6 ${data.rpm > 0 ? 'animate-[spin_3s_linear_infinite]' : ''}`} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">{title}</h2>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="flex items-center gap-1"><Activity className="w-3 h-3"/> Health: {hasTelemetry ? `${data.health}%` : '--'}</span>
              </div>
            </div>
          </div>
          <div 
            onClick={() => openSingleTrend(`${title} RPM`, `${motorId}Rpm`, '#10b981', 'RPM')}
            className="text-right cursor-pointer group relative"
          >
            <LineChartIcon className="absolute -left-6 top-1/2 -translate-y-1/2 w-4 h-4 text-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="text-3xl font-black text-white font-mono tracking-tight group-hover:text-emerald-400 transition-colors">
              {formatWhole(data.rpm)} <span className="text-sm font-medium text-slate-500">RPM</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div 
            onClick={() => openSingleTrend(`${title} Current Draw`, `${motorId}Current`, '#facc15', 'A', 120)}
            className="bg-slate-950/50 p-4 rounded-lg border border-slate-800/50 cursor-pointer group hover:bg-slate-800/60 hover:border-slate-600 transition-all relative"
          >
            <LineChartIcon className="absolute top-3 right-3 w-4 h-4 text-slate-600 group-hover:text-yellow-400 transition-colors opacity-0 group-hover:opacity-100" />
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <Zap className="w-4 h-4 text-yellow-400" /> Total Current
            </div>
            <div className="text-2xl font-mono text-white">{formatValue(data.currentDraw)} <span className="text-sm text-slate-500">A</span></div>
            <ProgressBar value={data.currentDraw} max={150} colorClass="bg-yellow-400" label="Load" unit="A" />
          </div>
          
          <div 
            onClick={() => setGraphModal({ 
              title: `${title} Temperature`, 
              dataKey: `${motorId}Temp`, 
              color: data.temp > 80 ? '#f43f5e' : '#f97316',
              unit: '°C' 
            })}
            className="bg-slate-950/50 p-4 rounded-lg border border-slate-800/50 cursor-pointer group hover:bg-slate-800/60 hover:border-slate-600 transition-all relative"
          >
            <LineChartIcon className="absolute top-3 right-3 w-4 h-4 text-slate-600 group-hover:text-orange-400 transition-colors opacity-0 group-hover:opacity-100" />
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <Thermometer className={`w-4 h-4 ${data.temp > 80 ? 'text-rose-500' : 'text-orange-400'}`} /> Temperature
            </div>
            <div className={`text-2xl font-mono ${data.temp > 80 ? 'text-rose-400' : 'text-white'}`}>
              {data.temp.toFixed(1)} <span className="text-sm text-slate-500">°C</span>
            </div>
            <ProgressBar value={data.temp} max={120} colorClass={data.temp > 80 ? 'bg-rose-500' : 'bg-orange-500'} label="Heat" unit="°C" />
          </div>
        </div>

        <div className="space-y-3">
          <div 
            onClick={() => setGraphModal({ title: `${title} Input Voltage`, dataKey: `${motorId}InputVoltage`, color: '#94a3b8', unit: 'V' })}
            className="flex justify-between items-center p-3 rounded-lg bg-slate-800/30 cursor-pointer group hover:bg-slate-800/60 transition-colors"
          >
            <span className="text-sm text-slate-400 flex items-center gap-2"><Gauge className="w-4 h-4"/> Input Voltage</span>
            <span className="font-mono text-white flex items-center gap-2">
              {data.inputVoltage.toFixed(1)} V
              <LineChartIcon className="w-3 h-3 text-slate-500 opacity-0 group-hover:opacity-100" />
            </span>
          </div>
          
          <div 
            onClick={() => setGraphModal({ title: `${title} Back EMF`, dataKey: `${motorId}BackEmf`, color: '#a78bfa', unit: 'V' })}
            className="flex justify-between items-center p-3 rounded-lg bg-slate-800/30 cursor-pointer group hover:bg-slate-800/60 transition-colors"
          >
            <span className="text-sm text-slate-400 flex items-center gap-2"><Zap className="w-4 h-4"/> Back EMF</span>
            <span className="font-mono text-white flex items-center gap-2">
              {data.backEmf.toFixed(1)} V
              <LineChartIcon className="w-3 h-3 text-slate-500 opacity-0 group-hover:opacity-100" />
            </span>
          </div>
          
          <div className="p-3 rounded-lg bg-slate-800/30">
            <span className="text-sm text-slate-400 mb-2 block">Phase Currents (U, V, W)</span>
            <div className="flex justify-between text-xs font-mono">
              <span 
                onClick={() => setGraphModal({ title: `${title} Phase U`, dataKey: `${motorId}PhaseU`, color: '#60a5fa', unit: 'A' })}
                className="text-blue-400 cursor-pointer hover:underline hover:text-blue-300"
              >U: {data.phaseU.toFixed(1)}A</span>
              <span 
                onClick={() => setGraphModal({ title: `${title} Phase V`, dataKey: `${motorId}PhaseV`, color: '#4ade80', unit: 'A' })}
                className="text-green-400 cursor-pointer hover:underline hover:text-green-300"
              >V: {data.phaseV.toFixed(1)}A</span>
              <span 
                onClick={() => setGraphModal({ title: `${title} Phase W`, dataKey: `${motorId}PhaseW`, color: '#c084fc', unit: 'A' })}
                className="text-purple-400 cursor-pointer hover:underline hover:text-purple-300"
              >W: {data.phaseW.toFixed(1)}A</span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-800/30">
            <span className="text-sm text-slate-400 mb-2 flex items-center gap-2"><Waves className="w-4 h-4"/> IMU Vibration (Peak)</span>
            <div className="flex items-end gap-2 h-8">
              <div className="flex-1 bg-slate-700 rounded-t" style={{ height: `${Math.min(100, data.vibrationX * 20)}%` }}></div>
              <div className="flex-1 bg-slate-600 rounded-t" style={{ height: `${Math.min(100, data.vibrationY * 20)}%` }}></div>
              <div className="flex-1 bg-slate-700 rounded-t" style={{ height: `${Math.min(100, data.vibrationZ * 20)}%` }}></div>
            </div>
            <div className="flex justify-between text-[10px] mt-1 text-slate-500 font-mono">
              <span 
                onClick={() => setGraphModal({ title: `${title} Vibration X`, dataKey: `${motorId}VibrationX`, color: '#94a3b8', unit: 'g' })}
                className="cursor-pointer hover:underline hover:text-slate-300"
              >X: {data.vibrationX.toFixed(2)}g</span>
              <span 
                onClick={() => setGraphModal({ title: `${title} Vibration Y`, dataKey: `${motorId}VibrationY`, color: '#94a3b8', unit: 'g' })}
                className="cursor-pointer hover:underline hover:text-slate-300"
              >Y: {data.vibrationY.toFixed(2)}g</span>
              <span 
                onClick={() => setGraphModal({ title: `${title} Vibration Z`, dataKey: `${motorId}VibrationZ`, color: '#94a3b8', unit: 'g' })}
                className="cursor-pointer hover:underline hover:text-slate-300"
              >Z: {data.vibrationZ.toFixed(2)}g</span>
            </div>
          </div>
        </div>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans selection:bg-emerald-500/30">
      <header className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-black text-white tracking-tight flex items-center gap-3">
            <Cpu className="w-8 h-8 text-emerald-500" />
            EV Telemetry Hub
          </h1>
          <p className="text-slate-400 mt-1">STM32 + Pi Pico Gateway • Dual BLDC ESC Monitor</p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-full text-sm font-medium">
            <Server className="w-4 h-4 text-slate-400" />
            <span>Pico Gateway:</span>
            {isConnected ? (
              <span className="flex items-center gap-1 text-emerald-400"><CheckCircle2 className="w-4 h-4"/> LIVE</span>
            ) : (
              <span className="flex items-center gap-1 text-rose-400"><AlertTriangle className="w-4 h-4"/> OFFLINE</span>
            )}
            <span className="hidden sm:inline text-slate-500">({streamMode})</span>
          </div>
          <button 
            onClick={handleGenerateAiReport}
            disabled={isGeneratingReport}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-full text-sm font-bold transition-colors shadow-lg shadow-indigo-900/20"
          >
            {isGeneratingReport ? (
              <RotateCcw className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4 text-indigo-200" />
            )}
            {isGeneratingReport ? 'Analyzing...' : 'AI Diagnostics ✨'}
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {hasTelemetry ? (
            <>
              {renderMotorPanel("Front Drive", "front", frontMotor)}
              {renderMotorPanel("Rear Drive", "rear", rearMotor)}
            </>
          ) : (
            <Card className="md:col-span-2 min-h-[520px] flex items-center justify-center text-center">
              <div>
                <Server className="w-10 h-10 text-slate-500 mx-auto mb-4" />
                <h2 className="text-xl font-bold text-white">Waiting for ESP32 telemetry</h2>
                <p className="text-sm text-slate-400 mt-2">No motor values will be shown until the first packet reaches the API.</p>
              </div>
            </Card>
          )}
        </div>

        <div className="lg:col-span-4 space-y-6">
          <Card className="bg-gradient-to-br from-slate-900 to-slate-950 border-emerald-900/30 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 blur-[50px] rounded-full"></div>
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-emerald-400" /> 
              TinyML Controller
            </h3>
            
            <div className="mb-6">
              <div className="text-sm text-slate-400 mb-1">Active Profile</div>
              <div className="text-xl font-bold text-emerald-400 bg-emerald-950/30 inline-block px-3 py-1 rounded border border-emerald-900/50">
                {mlStatus}
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm text-slate-400 mb-2">
                <span>Torque Vectoring</span>
                <span className="font-mono">
                  {hasTelemetry ? `F ${torqueSplit.front.toFixed(0)}% / R ${torqueSplit.rear.toFixed(0)}%` : '--'}
                </span>
              </div>
              <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex">
                <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${torqueSplit.front}%` }}></div>
                <div className="h-full bg-emerald-500 transition-all duration-300" style={{ width: `${torqueSplit.rear}%` }}></div>
              </div>
              <div className="flex justify-between text-xs mt-2 text-slate-500">
                <span>Front Bias</span>
                <span>Rear Bias</span>
              </div>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Battery className="w-5 h-5 text-yellow-400" /> 
              Power Bus
            </h3>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-sm text-slate-400">Total Draw</div>
                <div className="text-2xl font-mono text-white">
                  {hasTelemetry ? `${(frontMotor.currentDraw + rearMotor.currentDraw).toFixed(1)} A` : '--'}
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-slate-400">Est. Power</div>
                <div className="text-2xl font-mono text-white">
                  {hasTelemetry ? `${(((frontMotor.currentDraw + rearMotor.currentDraw) * 72) / 1000).toFixed(1)} kW` : '--'}
                </div>
              </div>
            </div>
            <div className="pt-4 mt-4 border-t border-slate-800 text-xs text-slate-400 space-y-2">
              <div className="flex justify-between gap-3">
                <span>Telemetry API</span>
                <span className="font-mono text-slate-300 text-right break-all">{TELEMETRY_API_URL}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span>Last Packet</span>
                <span className="font-mono text-slate-300">
                  {lastPacketAt ? new Date(lastPacketAt).toLocaleTimeString() : 'waiting'}
                </span>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-start justify-between gap-3 mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Database className="w-5 h-5 text-cyan-400" />
                Trend Recorder
              </h3>
              <button
                onClick={clearStoredHistory}
                disabled={history.length === 0}
                className="p-2 rounded-lg bg-slate-800 text-slate-300 hover:bg-rose-950 hover:text-rose-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                title="Reset stored telemetry"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
            <div className="grid grid-cols-3 gap-3 mb-4 text-center">
              <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-3">
                <div className="text-[11px] text-slate-500 uppercase">Samples</div>
                <div className="text-xl font-mono text-white">{history.length}</div>
              </div>
              <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-3">
                <div className="text-[11px] text-slate-500 uppercase">From</div>
                <div className="text-sm font-mono text-white">{historyStart}</div>
              </div>
              <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-3">
                <div className="text-[11px] text-slate-500 uppercase">To</div>
                <div className="text-sm font-mono text-white">{historyEnd}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => openTrend('Motor Temperature Comparison', 'C', [
                  { label: 'Front Temp', dataKey: 'frontTemp', color: '#fb923c' },
                  { label: 'Rear Temp', dataKey: 'rearTemp', color: '#f43f5e' },
                ], 80)}
                disabled={history.length < 2}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-sm text-slate-200 transition-colors"
              >
                Temperature
              </button>
              <button
                onClick={() => openTrend('Current Draw Comparison', 'A', [
                  { label: 'Front Current', dataKey: 'frontCurrent', color: '#fde047' },
                  { label: 'Rear Current', dataKey: 'rearCurrent', color: '#facc15' },
                ], 120)}
                disabled={history.length < 2}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-sm text-slate-200 transition-colors"
              >
                Current
              </button>
              <button
                onClick={() => openTrend('RPM Comparison', 'RPM', [
                  { label: 'Front RPM', dataKey: 'frontRpm', color: '#34d399' },
                  { label: 'Rear RPM', dataKey: 'rearRpm', color: '#22c55e' },
                ])}
                disabled={history.length < 2}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-sm text-slate-200 transition-colors"
              >
                RPM
              </button>
              <button
                onClick={() => openTrend('Vibration X Comparison', 'g', [
                  { label: 'Front Vib X', dataKey: 'frontVibrationX', color: '#38bdf8' },
                  { label: 'Rear Vib X', dataKey: 'rearVibrationX', color: '#818cf8' },
                ], 2.5)}
                disabled={history.length < 2}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-sm text-slate-200 transition-colors"
              >
                Vibration
              </button>
            </div>
            <p className="mt-3 text-[11px] text-slate-500">
              Stored locally in this browser, capped at {MAX_HISTORY_POINTS.toLocaleString()} samples.
            </p>
          </Card>

          <Card className="flex-1 flex flex-col h-[400px]">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-rose-400" /> 
                System Logs
              </h3>
              <span className="text-xs px-2 py-1 bg-slate-800 rounded-full text-slate-400">{alerts.length} Events</span>
            </div>
            
            <div className="flex-1 overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
              {alerts.map((alert) => (
                <div key={alert.id} className="text-sm p-3 rounded-lg bg-slate-950/50 border border-slate-800/50 flex flex-col gap-2 animate-in fade-in slide-in-from-right-4 duration-300">
                  <div className="flex gap-3 items-start">
                    <span className="text-slate-500 font-mono text-xs whitespace-nowrap mt-0.5">{alert.time}</span>
                    <div className="flex-1">
                      <p className={`${
                        alert.type === 'critical' ? 'text-rose-400 font-medium' : 
                        alert.type === 'warning' ? 'text-yellow-400' : 'text-slate-300'
                      }`}>
                        {alert.msg}
                      </p>
                    </div>
                  </div>
                  
                  {(alert.type === 'warning' || alert.type === 'critical') && (
                    <div className="ml-14 mt-1 border-t border-slate-800/50 pt-2">
                      {aiAnalysisState.alertId === alert.id ? (
                        <div className="text-indigo-300 text-xs bg-indigo-950/30 p-2 rounded border border-indigo-900/50">
                          {aiAnalysisState.loading ? (
                            <span className="flex items-center gap-2"><RotateCcw className="w-3 h-3 animate-spin"/> AI is analyzing...</span>
                          ) : (
                            <div>
                              <div className="font-bold flex items-center gap-1 mb-1"><Sparkles className="w-3 h-3"/> AI Insight</div>
                              {aiAnalysisState.content}
                            </div>
                          )}
                        </div>
                      ) : (
                        <button 
                          onClick={() => handleAnalyzeAlert(alert)}
                          className="text-xs flex items-center gap-1 text-indigo-400 hover:text-indigo-300 transition-colors"
                        >
                          <Sparkles className="w-3 h-3" /> Investigate Anomaly ✨
                        </button>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>
      </main>

      {/* AI Report Modal */}
      {aiReportContent && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-800/50">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <FileText className="w-5 h-5 text-indigo-400" />
                AI Diagnostic Report ✨
              </h3>
              <button onClick={() => setAiReportContent(null)} className="text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 overflow-y-auto text-slate-300 text-sm leading-relaxed whitespace-pre-wrap font-sans">
              {aiReportContent}
            </div>
            <div className="p-4 border-t border-slate-800 bg-slate-950 text-xs text-slate-500 text-center">
              Generated by Gemini. AI recommendations should be verified by a certified technician.
            </div>
          </div>
        </div>
      )}

      {/* Historical Data Graph Modal */}
      {graphModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-6xl w-full max-h-[88vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-800/50">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <LineChartIcon className="w-5 h-5" style={{ color: graphModal.color || graphModal.series?.[0]?.color }} />
                {graphModal.title}
              </h3>
              <button onClick={() => setGraphModal(null)} className="text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-5 h-[520px] min-h-0">
              <SVGLineChart 
                data={history} 
                series={graphModal.series || [{ label: graphModal.title, dataKey: graphModal.dataKey, color: graphModal.color }]}
                unit={graphModal.unit} 
                warningAt={graphModal.warningAt}
              />
            </div>
            <div className="p-4 border-t border-slate-800 bg-slate-950 text-xs text-slate-500 flex flex-col sm:flex-row justify-between gap-2">
              <span>Stored telemetry: {history.length.toLocaleString()} samples, scroll horizontally to inspect older data.</span>
              <span>Range: {historyStart} to {historyEnd}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
