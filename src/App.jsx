import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, Thermometer, Zap, Waves, AlertTriangle, 
  Download, Cpu, Gauge, Battery, Server, CheckCircle2, RotateCcw,
  Sparkles, X, FileText, LineChart as LineChartIcon
} from 'lucide-react';

// --- Utility Components ---
const SVGLineChart = ({ data, dataKey, strokeColor, unit }) => {
  if (!data || data.length < 2) return <div className="text-slate-500 flex items-center justify-center h-full">Gathering data...</div>;
  
  const values = data.map(d => d[dataKey]);
  const maxVal = Math.max(...values);
  const minVal = Math.min(...values);
  const pad = (maxVal - minVal) * 0.2 || 2; // Add 20% padding
  const maxPlot = maxVal + pad;
  const minPlot = Math.max(0, minVal - pad); 
  const range = maxPlot - minPlot;

  const pts = values.map((val, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - ((val - minPlot) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="relative w-full h-full flex flex-col font-mono">
       <div className="flex-1 border-l border-b border-slate-700 mt-2 ml-10 mb-6 relative">
          <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 100 100">
            <line x1="0" y1="25" x2="100" y2="25" stroke="#334155" strokeDasharray="4" strokeWidth="0.5"/>
            <line x1="0" y1="50" x2="100" y2="50" stroke="#334155" strokeDasharray="4" strokeWidth="0.5"/>
            <line x1="0" y1="75" x2="100" y2="75" stroke="#334155" strokeDasharray="4" strokeWidth="0.5"/>
            <polyline points={pts} fill="none" stroke={strokeColor} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div className="absolute -left-2 top-0 text-[10px] text-slate-400 -translate-x-full -translate-y-1/2">{maxPlot.toFixed(1)} {unit}</div>
          <div className="absolute -left-2 top-1/2 text-[10px] text-slate-400 -translate-x-full -translate-y-1/2">{((maxPlot+minPlot)/2).toFixed(1)} {unit}</div>
          <div className="absolute -left-2 bottom-0 text-[10px] text-slate-400 -translate-x-full translate-y-1/2">{minPlot.toFixed(1)} {unit}</div>
       </div>
       <div className="flex justify-between px-10 text-[10px] text-slate-500">
         <span>{data[0].time}</span>
         <span>{data[Math.floor(data.length/2)].time}</span>
         <span>{data[data.length-1].time}</span>
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

// --- Main Application ---
export default function App() {
  const [isConnected, setIsConnected] = useState(true);
  const [mlStatus, setMlStatus] = useState("Adaptive AWD Active");
  const [torqueSplit, setTorqueSplit] = useState({ front: 50, rear: 50 });
  
  const [frontMotor, setFrontMotor] = useState({
    rpm: 0, temp: 45, inputVoltage: 72, currentDraw: 0,
    phaseU: 0, phaseV: 0, phaseW: 0,
    vibrationX: 0.1, vibrationY: 0.1, vibrationZ: 0.1,
    backEmf: 0, health: 100
  });

  const [rearMotor, setRearMotor] = useState({
    rpm: 0, temp: 48, inputVoltage: 72, currentDraw: 0,
    phaseU: 0, phaseV: 0, phaseW: 0,
    vibrationX: 0.1, vibrationY: 0.2, vibrationZ: 0.1,
    backEmf: 0, health: 98
  });

  const [alerts, setAlerts] = useState([
    { id: 1, time: new Date().toLocaleTimeString(), type: 'info', msg: 'System Initialized. Awaiting Pico stream.' }
  ]);

  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [aiReportContent, setAiReportContent] = useState(null);
  const [aiAnalysisState, setAiAnalysisState] = useState({ loading: false, alertId: null, content: null });

  // History state for graphs
  const [history, setHistory] = useState([]);
  const [graphModal, setGraphModal] = useState(null); 
  
  const latestData = useRef({
    frontTemp: 45, frontCurrent: 0, frontRpm: 0, frontInputVoltage: 72, frontBackEmf: 0, frontPhaseU: 0, frontPhaseV: 0, frontPhaseW: 0, frontVibrationX: 0, frontVibrationY: 0, frontVibrationZ: 0,
    rearTemp: 48, rearCurrent: 0, rearRpm: 0, rearInputVoltage: 72, rearBackEmf: 0, rearPhaseU: 0, rearPhaseV: 0, rearPhaseW: 0, rearVibrationX: 0, rearVibrationY: 0, rearVibrationZ: 0
  });

  useEffect(() => {
    const interval = setInterval(() => {
      const baseRpm = 3500 + Math.random() * 500;
      const baseCurrent = 40 + Math.random() * 15;
      
      setFrontMotor(prev => {
        const next = {
          ...prev,
          rpm: baseRpm + (Math.random() * 100 - 50),
          temp: Math.min(120, prev.temp + (Math.random() * 0.4 - 0.1)),
          currentDraw: baseCurrent,
          phaseU: baseCurrent * 0.33 + Math.random() * 2,
          phaseV: baseCurrent * 0.33 + Math.random() * 2,
          phaseW: baseCurrent * 0.33 + Math.random() * 2,
          vibrationX: Math.abs(Math.sin(Date.now() / 1000) * 1.5) + 0.5,
          vibrationY: Math.abs(Math.cos(Date.now() / 1200) * 1.0) + 0.5,
          vibrationZ: Math.abs(Math.sin(Date.now() / 900) * 0.8) + 0.5,
          backEmf: (baseRpm * 0.015) + Math.random() * 2
        };
        latestData.current.frontTemp = next.temp;
        latestData.current.frontCurrent = next.currentDraw;
        latestData.current.frontRpm = next.rpm;
        latestData.current.frontInputVoltage = next.inputVoltage;
        latestData.current.frontBackEmf = next.backEmf;
        latestData.current.frontPhaseU = next.phaseU;
        latestData.current.frontPhaseV = next.phaseV;
        latestData.current.frontPhaseW = next.phaseW;
        latestData.current.frontVibrationX = next.vibrationX;
        latestData.current.frontVibrationY = next.vibrationY;
        latestData.current.frontVibrationZ = next.vibrationZ;
        return next;
      });

      setRearMotor(prev => {
        const next = {
          ...prev,
          rpm: baseRpm + (Math.random() * 120 - 60),
          temp: Math.min(120, prev.temp + (Math.random() * 0.5 - 0.1)),
          currentDraw: baseCurrent * 1.2,
          phaseU: (baseCurrent * 1.2) * 0.33 + Math.random() * 2,
          phaseV: (baseCurrent * 1.2) * 0.33 + Math.random() * 2,
          phaseW: (baseCurrent * 1.2) * 0.33 + Math.random() * 2,
          vibrationX: Math.abs(Math.cos(Date.now() / 800) * 2.0) + 0.5,
          vibrationY: Math.abs(Math.sin(Date.now() / 1000) * 1.5) + 0.5,
          vibrationZ: Math.abs(Math.cos(Date.now() / 1100) * 1.0) + 0.5,
          backEmf: (baseRpm * 0.015) + Math.random() * 2
        };
        latestData.current.rearTemp = next.temp;
        latestData.current.rearCurrent = next.currentDraw;
        latestData.current.rearRpm = next.rpm;
        latestData.current.rearInputVoltage = next.inputVoltage;
        latestData.current.rearBackEmf = next.backEmf;
        latestData.current.rearPhaseU = next.phaseU;
        latestData.current.rearPhaseV = next.phaseV;
        latestData.current.rearPhaseW = next.phaseW;
        latestData.current.rearVibrationX = next.vibrationX;
        latestData.current.rearVibrationY = next.vibrationY;
        latestData.current.rearVibrationZ = next.vibrationZ;
        return next;
      });

      const split = 40 + Math.random() * 20;
      setTorqueSplit({ front: split, rear: 100 - split });

      setHistory(prev => [...prev, {
        time: new Date().toLocaleTimeString([], { hour12: false }),
        ...latestData.current
      }].slice(-60));

    }, 800);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (rearMotor.temp > 85 && alerts.length < 5) {
      addAlert('warning', 'Rear motor temperature exceeding optimal bounds (85°C+)');
    }
    if (frontMotor.vibrationX > 3.0) {
      addAlert('critical', 'High vibration anomaly detected on Front Drive IMU');
    }
  }, [frontMotor.temp, rearMotor.temp, frontMotor.vibrationX]);

  const addAlert = (type, msg) => {
    const newAlert = { id: Date.now(), time: new Date().toLocaleTimeString(), type, msg };
    setAlerts(prev => [newAlert, ...prev].slice(0, 8));
  };

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
    } catch (e) {
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
    } catch (e) {
      setAiAnalysisState({ loading: false, alertId: alert.id, content: "Analysis failed. Please check network connection." });
    }
  };

  const MotorPanel = ({ title, motorId, data }) => {
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
                <span className="flex items-center gap-1"><Activity className="w-3 h-3"/> Health: {data.health}%</span>
              </div>
            </div>
          </div>
          <div 
            onClick={() => setGraphModal({ title: `${title} RPM`, dataKey: `${motorId}Rpm`, color: '#10b981', unit: 'RPM' })}
            className="text-right cursor-pointer group relative"
          >
            <LineChartIcon className="absolute -left-6 top-1/2 -translate-y-1/2 w-4 h-4 text-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="text-3xl font-black text-white font-mono tracking-tight group-hover:text-emerald-400 transition-colors">
              {data.rpm.toFixed(0)} <span className="text-sm font-medium text-slate-500">RPM</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div 
            onClick={() => setGraphModal({ 
              title: `${title} Current Draw`, 
              dataKey: `${motorId}Current`, 
              color: '#facc15', 
              unit: 'A' 
            })}
            className="bg-slate-950/50 p-4 rounded-lg border border-slate-800/50 cursor-pointer group hover:bg-slate-800/60 hover:border-slate-600 transition-all relative"
          >
            <LineChartIcon className="absolute top-3 right-3 w-4 h-4 text-slate-600 group-hover:text-yellow-400 transition-colors opacity-0 group-hover:opacity-100" />
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <Zap className="w-4 h-4 text-yellow-400" /> Total Current
            </div>
            <div className="text-2xl font-mono text-white">{data.currentDraw.toFixed(1)} <span className="text-sm text-slate-500">A</span></div>
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
          <MotorPanel title="Front Drive" motorId="front" data={frontMotor} />
          <MotorPanel title="Rear Drive" motorId="rear" data={rearMotor} />
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
                <span className="font-mono">F {torqueSplit.front.toFixed(0)}% / R {torqueSplit.rear.toFixed(0)}%</span>
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
                <div className="text-2xl font-mono text-white">{(frontMotor.currentDraw + rearMotor.currentDraw).toFixed(1)} A</div>
              </div>
              <div className="text-right">
                <div className="text-sm text-slate-400">Est. Power</div>
                <div className="text-2xl font-mono text-white">
                  {(((frontMotor.currentDraw + rearMotor.currentDraw) * 72) / 1000).toFixed(1)} kW
                </div>
              </div>
            </div>
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
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-2xl w-full flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-800/50">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <LineChartIcon className="w-5 h-5" style={{ color: graphModal.color }} />
                {graphModal.title} over Time
              </h3>
              <button onClick={() => setGraphModal(null)} className="text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 h-[350px]">
              <SVGLineChart 
                data={history} 
                dataKey={graphModal.dataKey} 
                strokeColor={graphModal.color}
                unit={graphModal.unit} 
              />
            </div>
            <div className="p-4 border-t border-slate-800 bg-slate-950 text-xs text-slate-500 text-center">
              Real-time local telemetry buffer (last ~45 seconds)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}