import { useState, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────
// GLOBAL STYLES
// ─────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: #0c0c0f;
    color: #e8e8ed;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  :root {
    --bg:        #0c0c0f;
    --bg1:       #131317;
    --bg2:       #1a1a20;
    --bg3:       #222228;
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.12);
    --indigo:    #6366f1;
    --indigo2:   #818cf8;
    --teal:      #14b8a6;
    --teal2:     #2dd4bf;
    --red:       #f87171;
    --amber:     #fbbf24;
    --green:     #34d399;
    --text1:     #f0f0f5;
    --text2:     #9999aa;
    --text3:     #55555f;
    --mono:      'DM Mono', monospace;
  }

  input, select, textarea {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg2);
    border: 1px solid var(--border2);
    color: var(--text1);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    outline: none;
    width: 100%;
    transition: border-color 0.15s;
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--indigo);
  }
  select option { background: #1a1a20; }

  button { cursor: pointer; font-family: 'DM Sans', sans-serif; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes scaleIn {
    from { opacity: 0; transform: scale(0.96); }
    to   { opacity: 1; transform: scale(1); }
  }
  .fade-up  { animation: fadeUp 0.3s ease both; }
  .fade-in  { animation: fadeIn 0.2s ease both; }
  .scale-in { animation: scaleIn 0.25s ease both; }

  .stagger-1 { animation-delay: 0.05s; }
  .stagger-2 { animation-delay: 0.10s; }
  .stagger-3 { animation-delay: 0.15s; }
  .stagger-4 { animation-delay: 0.20s; }
  .stagger-5 { animation-delay: 0.25s; }
`;

// ─────────────────────────────────────────────────────────────
// STATS ENGINE (Python ab_engine.py → JS)
// ─────────────────────────────────────────────────────────────
const MIN_SAMPLES = 10;

function mean(arr) {
  return arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : 0;
}
function variance(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
}
function std(arr) {
  return Math.sqrt(variance(arr));
}
function normalCDF(z) {
  const t = 1 / (1 + 0.3275911 * Math.abs(z));
  const p = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z / 2);
  return z >= 0 ? p : 1 - p;
}
function welchPValue(a, b) {
  if (a.length < 2 || b.length < 2) return 1;
  const mA = mean(a), mB = mean(b);
  const vA = variance(a), vB = variance(b);
  const nA = a.length, nB = b.length;
  const se = Math.sqrt(vA / nA + vB / nB);
  if (se === 0) return 1;
  const t = (mA - mB) / se;
  const df = (vA / nA + vB / nB) ** 2 / ((vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1));
  const x = df / (df + t * t);
  // Beta regularized incomplete function approximation for t-distribution CDF
  const p = 2 * Math.min(normalCDF(t), normalCDF(-t));
  return Math.max(0, Math.min(1, p));
}
function cohensD(a, b) {
  const pooled = Math.sqrt((variance(a) + variance(b)) / 2);
  if (pooled === 0) return 0;
  return Math.abs(mean(a) - mean(b)) / pooled;
}
function effectLabel(d) {
  if (d >= 0.8) return "large";
  if (d >= 0.5) return "medium";
  if (d >= 0.2) return "small";
  return "negligible";
}
function proportionZTest(succA, nA, succB, nB) {
  if (nA === 0 || nB === 0) return 1;
  const pA = succA / nA, pB = succB / nB;
  const pPool = (succA + succB) / (nA + nB);
  if (pPool === 0 || pPool === 1) return 1;
  const se = Math.sqrt(pPool * (1 - pPool) * (1 / nA + 1 / nB));
  if (se === 0) return 1;
  const z = (pA - pB) / se;
  return 2 * (1 - normalCDF(Math.abs(z)));
}

function analyzeMetric(entries, variants, metric) {
  const groups = variants.map(v =>
    entries.filter(e => e.variant === v).map(e => e.metrics[metric.name] ?? null).filter(x => x !== null)
  );
  const ns = groups.map(g => g.length);
  const minN = Math.min(...ns);
  const sampleWarning = minN < MIN_SAMPLES ? `Need ${MIN_SAMPLES - minN} more data point(s) per variant for reliable results` : null;

  let result = { metric: metric.name, type: metric.type, unit: metric.unit, primary: metric.primary, sampleWarning };

  if (metric.type === "binary") {
    const counts  = groups.map(g => g.filter(x => x === 1).length);
    const totals  = ns;
    const rates   = counts.map((c, i) => totals[i] > 0 ? c / totals[i] : 0);
    const pVal    = proportionZTest(counts[0], totals[0], counts[1], totals[1]);
    const conf    = (1 - pVal) * 100;
    const winIdx  = rates[0] >= rates[1] ? 0 : 1;
    const loserR  = rates[1 - winIdx];
    const lift    = loserR > 0 ? ((rates[winIdx] - loserR) / loserR * 100) : null;
    result = { ...result, variantStats: variants.map((v, i) => ({ variant: v, n: totals[i], successes: counts[i], rate: +(rates[i] * 100).toFixed(1) })), winner: variants[winIdx], lift: lift !== null ? +lift.toFixed(1) : null, pValue: +pVal.toFixed(4), confidence: +conf.toFixed(1), significant: pVal < 0.05, cohensD: null, effectSize: null };
  } else {
    const means   = groups.map(mean);
    const stds    = groups.map(std);
    const pVal    = welchPValue(groups[0], groups[1]);
    const conf    = (1 - pVal) * 100;
    const d       = cohensD(groups[0], groups[1]);
    const winIdx  = means[0] >= means[1] ? 0 : 1;
    const loserM  = means[1 - winIdx];
    const lift    = loserM !== 0 ? ((means[winIdx] - loserM) / loserM * 100) : null;
    result = { ...result, variantStats: variants.map((v, i) => ({ variant: v, n: ns[i], mean: +means[i].toFixed(2), std: +stds[i].toFixed(2) })), winner: variants[winIdx], lift: lift !== null ? +lift.toFixed(1) : null, pValue: +pVal.toFixed(4), confidence: +conf.toFixed(1), significant: pVal < 0.05, cohensD: +d.toFixed(2), effectSize: effectLabel(d) };
  }
  return result;
}

function analyzeExperiment(experiment) {
  return experiment.metrics.map(m => analyzeMetric(experiment.entries, experiment.variants, m));
}

// ─────────────────────────────────────────────────────────────
// LOCAL STORAGE
// ─────────────────────────────────────────────────────────────
const STORAGE_KEY = "ab_experiments_v1";

const SEED_DATA = [
  {
    id: "exp_1", name: "Gym vs Café", variants: ["Gym", "Café"], createdAt: "2024-01-01",
    metrics: [
      { name: "number_exchanged", type: "binary",     unit: null,    primary: true  },
      { name: "conversation_duration", type: "continuous", unit: "mins", primary: false },
    ],
    entries: [
      { id:"e1",  date:"2024-01-01", variant:"Gym",  metrics:{ number_exchanged:1, conversation_duration:5.0 }},
      { id:"e2",  date:"2024-01-02", variant:"Café", metrics:{ number_exchanged:0, conversation_duration:1.5 }},
      { id:"e3",  date:"2024-01-03", variant:"Gym",  metrics:{ number_exchanged:1, conversation_duration:6.5 }},
      { id:"e4",  date:"2024-01-04", variant:"Café", metrics:{ number_exchanged:0, conversation_duration:2.0 }},
      { id:"e5",  date:"2024-01-05", variant:"Gym",  metrics:{ number_exchanged:0, conversation_duration:3.0 }},
      { id:"e6",  date:"2024-01-06", variant:"Café", metrics:{ number_exchanged:1, conversation_duration:4.0 }},
      { id:"e7",  date:"2024-01-07", variant:"Gym",  metrics:{ number_exchanged:1, conversation_duration:7.0 }},
      { id:"e8",  date:"2024-01-08", variant:"Café", metrics:{ number_exchanged:0, conversation_duration:1.0 }},
      { id:"e9",  date:"2024-01-09", variant:"Gym",  metrics:{ number_exchanged:1, conversation_duration:5.5 }},
      { id:"e10", date:"2024-01-10", variant:"Café", metrics:{ number_exchanged:0, conversation_duration:2.5 }},
      { id:"e11", date:"2024-01-11", variant:"Gym",  metrics:{ number_exchanged:0, conversation_duration:4.0 }},
      { id:"e12", date:"2024-01-12", variant:"Café", metrics:{ number_exchanged:1, conversation_duration:5.0 }},
    ]
  },
  {
    id: "exp_2", name: "Coffee vs No Coffee", variants: ["Coffee", "No Coffee"], createdAt: "2024-01-01",
    metrics: [
      { name: "productivity", type: "scale",  unit: "1-10", primary: true  },
      { name: "focus",        type: "scale",  unit: "1-10", primary: false },
      { name: "headache",     type: "binary", unit: null,   primary: false },
    ],
    entries: [
      { id:"f1",  date:"2024-01-01", variant:"Coffee",    metrics:{ productivity:8, focus:8, headache:0 }},
      { id:"f2",  date:"2024-01-02", variant:"No Coffee", metrics:{ productivity:5, focus:5, headache:1 }},
      { id:"f3",  date:"2024-01-03", variant:"Coffee",    metrics:{ productivity:7, focus:7, headache:0 }},
      { id:"f4",  date:"2024-01-04", variant:"No Coffee", metrics:{ productivity:6, focus:6, headache:0 }},
      { id:"f5",  date:"2024-01-05", variant:"Coffee",    metrics:{ productivity:9, focus:8, headache:0 }},
      { id:"f6",  date:"2024-01-06", variant:"No Coffee", metrics:{ productivity:4, focus:4, headache:1 }},
      { id:"f7",  date:"2024-01-07", variant:"Coffee",    metrics:{ productivity:8, focus:9, headache:0 }},
      { id:"f8",  date:"2024-01-08", variant:"No Coffee", metrics:{ productivity:6, focus:5, headache:1 }},
      { id:"f9",  date:"2024-01-09", variant:"Coffee",    metrics:{ productivity:7, focus:7, headache:0 }},
      { id:"f10", date:"2024-01-10", variant:"No Coffee", metrics:{ productivity:5, focus:5, headache:0 }},
    ]
  }
];

function loadExperiments() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  localStorage.setItem(STORAGE_KEY, JSON.stringify(SEED_DATA));
  return SEED_DATA;
}
function saveExperiments(exps) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(exps));
}
function uid() { return "id_" + Math.random().toString(36).slice(2) + Date.now(); }

// ─────────────────────────────────────────────────────────────
// SMALL UI COMPONENTS
// ─────────────────────────────────────────────────────────────

const Badge = ({ children, color = "indigo" }) => {
  const colors = {
    indigo: { bg: "rgba(99,102,241,0.15)", text: "#818cf8", border: "rgba(99,102,241,0.25)" },
    teal:   { bg: "rgba(20,184,166,0.15)", text: "#2dd4bf", border: "rgba(20,184,166,0.25)" },
    red:    { bg: "rgba(248,113,113,0.15)", text: "#fca5a5", border: "rgba(248,113,113,0.3)" },
    amber:  { bg: "rgba(251,191,36,0.12)",  text: "#fcd34d", border: "rgba(251,191,36,0.25)" },
    green:  { bg: "rgba(52,211,153,0.12)",  text: "#6ee7b7", border: "rgba(52,211,153,0.25)" },
  };
  const c = colors[color] || colors.indigo;
  return (
    <span style={{ background: c.bg, color: c.text, border: `1px solid ${c.border}`, borderRadius: 6, padding: "2px 10px", fontSize: 12, fontWeight: 500, fontFamily: "var(--mono)", whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
};

const Btn = ({ children, onClick, variant = "primary", small, disabled, style: sx }) => {
  const styles = {
    primary:  { background: "var(--indigo)", color: "#fff", border: "none" },
    ghost:    { background: "transparent", color: "var(--text2)", border: "1px solid var(--border2)" },
    danger:   { background: "rgba(248,113,113,0.12)", color: "var(--red)", border: "1px solid rgba(248,113,113,0.25)" },
    teal:     { background: "rgba(20,184,166,0.15)", color: "var(--teal2)", border: "1px solid rgba(20,184,166,0.25)" },
  };
  return (
    <button onClick={onClick} disabled={disabled} style={{ ...styles[variant], borderRadius: 8, padding: small ? "6px 14px" : "10px 20px", fontSize: small ? 13 : 14, fontWeight: 500, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1, transition: "opacity 0.15s, background 0.15s", ...sx }}>
      {children}
    </button>
  );
};

const StatCard = ({ label, value, sub, color, delay = 0 }) => (
  <div className="fade-up" style={{ animationDelay: `${delay}s`, background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 12, padding: "16px 18px" }}>
    <div style={{ fontSize: 12, color: "var(--text2)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 500 }}>{label}</div>
    <div style={{ fontSize: 24, fontWeight: 600, color: color || "var(--text1)", fontFamily: "var(--mono)", letterSpacing: "-0.02em" }}>{value}</div>
    {sub && <div style={{ fontSize: 12, color: "var(--text3)", marginTop: 4 }}>{sub}</div>}
  </div>
);

// Mini bar chart for variant comparison
const VariantBar = ({ label, value, max, color, suffix = "" }) => (
  <div style={{ marginBottom: 12 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
      <span style={{ fontSize: 13, color: "var(--text2)" }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 500, fontFamily: "var(--mono)", color }}>{value}{suffix}</span>
    </div>
    <div style={{ height: 6, background: "var(--bg3)", borderRadius: 99, overflow: "hidden" }}>
      <div style={{ height: "100%", width: `${Math.min(100, (value / max) * 100)}%`, background: color, borderRadius: 99, transition: "width 0.6s ease" }} />
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────
// MODAL
// ─────────────────────────────────────────────────────────────
const Modal = ({ title, onClose, children, wide }) => (
  <div className="fade-in" onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
    <div className="scale-in" onClick={e => e.stopPropagation()} style={{ background: "var(--bg1)", border: "1px solid var(--border2)", borderRadius: 16, padding: "28px 28px 24px", width: "100%", maxWidth: wide ? 680 : 520, maxHeight: "90vh", overflowY: "auto" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: "var(--text1)" }}>{title}</h2>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--text3)", fontSize: 22, lineHeight: 1, cursor: "pointer", padding: "0 4px" }}>×</button>
      </div>
      {children}
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────
// CREATE EXPERIMENT MODAL
// ─────────────────────────────────────────────────────────────
const METRIC_TYPES = [
  { value: "binary",     label: "Binary (yes/no)",  desc: "e.g. number exchanged, workout done" },
  { value: "continuous", label: "Continuous",        desc: "e.g. duration, calories, revenue" },
  { value: "scale",      label: "Scale (1–10)",      desc: "e.g. mood, focus, confidence" },
];

function CreateExperimentModal({ onClose, onCreate }) {
  const [name, setName] = useState("");
  const [variantA, setVariantA] = useState("");
  const [variantB, setVariantB] = useState("");
  const [metrics, setMetrics] = useState([{ name: "", type: "continuous", unit: "", primary: true }]);

  const addMetric = () => setMetrics(m => [...m, { name: "", type: "continuous", unit: "", primary: false }]);
  const removeMetric = i => setMetrics(m => m.filter((_, idx) => idx !== i));
  const updateMetric = (i, key, val) => setMetrics(m => m.map((x, idx) => idx === i ? { ...x, [key]: val } : x));
  const setPrimary = i => setMetrics(m => m.map((x, idx) => ({ ...x, primary: idx === i })));

  const valid = name.trim() && variantA.trim() && variantB.trim() && metrics.every(m => m.name.trim());

  const submit = () => {
    if (!valid) return;
    onCreate({
      id: uid(), name: name.trim(),
      variants: [variantA.trim(), variantB.trim()],
      createdAt: new Date().toISOString().slice(0, 10),
      metrics: metrics.map(m => ({ name: m.name.trim().toLowerCase().replace(/\s+/g, "_"), type: m.type, unit: m.unit || null, primary: m.primary })),
      entries: []
    });
    onClose();
  };

  return (
    <Modal title="New experiment" onClose={onClose} wide>
      <div style={{ display: "grid", gap: 16, marginBottom: 20 }}>
        <div>
          <label style={{ fontSize: 13, color: "var(--text2)", display: "block", marginBottom: 6 }}>Experiment name</label>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Gym vs Café" />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <label style={{ fontSize: 13, color: "var(--text2)", display: "block", marginBottom: 6 }}>Variant A</label>
            <input value={variantA} onChange={e => setVariantA(e.target.value)} placeholder="e.g. Gym" />
          </div>
          <div>
            <label style={{ fontSize: 13, color: "var(--text2)", display: "block", marginBottom: 6 }}>Variant B</label>
            <input value={variantB} onChange={e => setVariantB(e.target.value)} placeholder="e.g. Café" />
          </div>
        </div>
      </div>

      <div style={{ borderTop: "1px solid var(--border)", paddingTop: 20, marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <span style={{ fontSize: 14, fontWeight: 500, color: "var(--text1)" }}>Metrics</span>
          <Btn small variant="ghost" onClick={addMetric}>+ Add metric</Btn>
        </div>
        {metrics.map((m, i) => (
          <div key={i} style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 10, padding: "14px 16px", marginBottom: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
              <div>
                <label style={{ fontSize: 12, color: "var(--text3)", display: "block", marginBottom: 5 }}>Metric name</label>
                <input value={m.name} onChange={e => updateMetric(i, "name", e.target.value)} placeholder="e.g. conversation_duration" />
              </div>
              <div>
                <label style={{ fontSize: 12, color: "var(--text3)", display: "block", marginBottom: 5 }}>Unit (optional)</label>
                <input value={m.unit} onChange={e => updateMetric(i, "unit", e.target.value)} placeholder="e.g. mins, 1-10" />
              </div>
            </div>
            <div style={{ marginBottom: 10 }}>
              <label style={{ fontSize: 12, color: "var(--text3)", display: "block", marginBottom: 5 }}>Type</label>
              <select value={m.type} onChange={e => updateMetric(i, "type", e.target.value)}>
                {METRIC_TYPES.map(t => <option key={t.value} value={t.value}>{t.label} — {t.desc}</option>)}
              </select>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label style={{ fontSize: 12, color: "var(--text2)", display: "flex", alignItems: "center", gap: 7, cursor: "pointer" }}>
                <input type="radio" checked={m.primary} onChange={() => setPrimary(i)} style={{ width: "auto", accentColor: "var(--indigo)" }} />
                Primary metric
              </label>
              {metrics.length > 1 && <Btn small variant="danger" onClick={() => removeMetric(i)}>Remove</Btn>}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
        <Btn onClick={submit} disabled={!valid}>Create experiment</Btn>
      </div>
    </Modal>
  );
}

// ─────────────────────────────────────────────────────────────
// LOG ENTRY MODAL
// ─────────────────────────────────────────────────────────────
function LogEntryModal({ experiment, onClose, onLog }) {
  const [variant, setVariant] = useState(experiment.variants[0]);
  const [date, setDate]       = useState(new Date().toISOString().slice(0, 10));
  const [values, setValues]   = useState(Object.fromEntries(experiment.metrics.map(m => [m.name, ""])));

  const setVal = (name, val) => setValues(v => ({ ...v, [name]: val }));
  const valid  = experiment.metrics.every(m => values[m.name] !== "");

  const submit = () => {
    const metrics = Object.fromEntries(
      experiment.metrics.map(m => [m.name, m.type === "binary" ? Number(values[m.name]) : parseFloat(values[m.name])])
    );
    onLog({ id: uid(), date, variant, metrics });
    onClose();
  };

  return (
    <Modal title={`Log entry — ${experiment.name}`} onClose={onClose}>
      <div style={{ display: "grid", gap: 14, marginBottom: 20 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <label style={{ fontSize: 13, color: "var(--text2)", display: "block", marginBottom: 6 }}>Variant</label>
            <select value={variant} onChange={e => setVariant(e.target.value)}>
              {experiment.variants.map(v => <option key={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label style={{ fontSize: 13, color: "var(--text2)", display: "block", marginBottom: 6 }}>Date</label>
            <input type="date" value={date} onChange={e => setDate(e.target.value)} />
          </div>
        </div>

        {experiment.metrics.map(m => (
          <div key={m.name}>
            <label style={{ fontSize: 13, color: "var(--text2)", display: "block", marginBottom: 6 }}>
              {m.name.replace(/_/g, " ")} {m.unit ? `(${m.unit})` : ""} {m.primary && <Badge color="indigo">primary</Badge>}
            </label>
            {m.type === "binary" ? (
              <select value={values[m.name]} onChange={e => setVal(m.name, e.target.value)}>
                <option value="">— select —</option>
                <option value="1">Yes / Success (1)</option>
                <option value="0">No / Failure (0)</option>
              </select>
            ) : m.type === "scale" ? (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(10,1fr)", gap: 6 }}>
                {[1,2,3,4,5,6,7,8,9,10].map(n => (
                  <button key={n} onClick={() => setVal(m.name, n)} style={{ padding: "8px 0", background: values[m.name] === n ? "var(--indigo)" : "var(--bg2)", border: `1px solid ${values[m.name] === n ? "var(--indigo)" : "var(--border2)"}`, borderRadius: 8, color: values[m.name] === n ? "#fff" : "var(--text2)", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all 0.15s" }}>
                    {n}
                  </button>
                ))}
              </div>
            ) : (
              <input type="number" step="any" placeholder="Enter value" value={values[m.name]} onChange={e => setVal(m.name, e.target.value)} />
            )}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
        <Btn onClick={submit} disabled={!valid}>Log entry</Btn>
      </div>
    </Modal>
  );
}

// ─────────────────────────────────────────────────────────────
// RESULTS VIEW
// ─────────────────────────────────────────────────────────────
function ResultsView({ experiment, onBack, onLog }) {
  const results  = analyzeExperiment(experiment);
  const primary  = results.find(r => r.primary) || results[0];
  const others   = results.filter(r => !r.primary);
  const totalN   = experiment.entries.length;
  const [A, B]   = experiment.variants;

  const confColor = c => c >= 95 ? "#34d399" : c >= 80 ? "#fbbf24" : "#f87171";
  const dColor    = d => d >= 0.8 ? "#34d399" : d >= 0.5 ? "#fbbf24" : "#9999aa";

  const MetricBlock = ({ r, delay = 0 }) => {
    const isB = r.type === "binary";
    const statA = r.variantStats[0], statB = r.variantStats[1];
    const maxVal = isB ? 100 : Math.max(statA.mean || 0, statB.mean || 0) * 1.2;

    return (
      <div className="fade-up" style={{ animationDelay: `${delay}s`, background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 14, padding: "20px 22px", marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text1)", textTransform: "capitalize" }}>{r.metric.replace(/_/g, " ")}</span>
          {r.unit && <span style={{ fontSize: 12, color: "var(--text3)" }}>{r.unit}</span>}
          {r.primary && <Badge color="indigo">primary</Badge>}
          <Badge color="indigo">{r.type}</Badge>
          {r.significant ? <Badge color="green">significant</Badge> : <Badge color="amber">not significant</Badge>}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 18 }}>
          {[statA, statB].map((s, i) => (
            <div key={i} style={{ background: "var(--bg3)", borderRadius: 10, padding: "14px 16px", border: r.winner === s.variant ? "1px solid rgba(99,102,241,0.4)" : "1px solid var(--border)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 13, color: "var(--text2)" }}>{s.variant}</span>
                {r.winner === s.variant && <Badge color="indigo">winner</Badge>}
              </div>
              {isB ? (
                <div style={{ fontSize: 26, fontWeight: 600, fontFamily: "var(--mono)", color: "var(--text1)" }}>{s.rate}%<span style={{ fontSize: 13, color: "var(--text3)", fontWeight: 400 }}> ({s.successes}/{s.n})</span></div>
              ) : (
                <div style={{ fontSize: 26, fontWeight: 600, fontFamily: "var(--mono)", color: "var(--text1)" }}>{s.mean}<span style={{ fontSize: 13, color: "var(--text3)", fontWeight: 400 }}> ±{s.std} (n={s.n})</span></div>
              )}
            </div>
          ))}
        </div>

        <div style={{ marginBottom: 16 }}>
          <VariantBar label={statA.variant} value={isB ? statA.rate : statA.mean} max={isB ? 100 : maxVal} color="#6366f1" suffix={isB ? "%" : ""} />
          <VariantBar label={statB.variant} value={isB ? statB.rate : statB.mean} max={isB ? 100 : maxVal} color="#14b8a6" suffix={isB ? "%" : ""} />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
          <div style={{ background: "var(--bg3)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>Lift</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: "var(--mono)", color: r.lift > 0 ? "var(--green)" : "var(--text1)" }}>{r.lift !== null ? `+${r.lift}%` : "—"}</div>
          </div>
          <div style={{ background: "var(--bg3)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>Confidence</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: "var(--mono)", color: confColor(r.confidence) }}>{r.confidence}%</div>
          </div>
          <div style={{ background: "var(--bg3)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>p-value</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: "var(--mono)", color: r.pValue < 0.05 ? "var(--green)" : "var(--amber)" }}>{r.pValue}</div>
          </div>
          <div style={{ background: "var(--bg3)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>Cohen's d</div>
            <div style={{ fontSize: 18, fontWeight: 600, fontFamily: "var(--mono)", color: r.cohensD !== null ? dColor(r.cohensD) : "var(--text3)" }}>
              {r.cohensD !== null ? r.cohensD : "—"}
            </div>
            {r.effectSize && <div style={{ fontSize: 11, color: "var(--text3)", marginTop: 2 }}>{r.effectSize}</div>}
          </div>
        </div>

        {r.sampleWarning && (
          <div style={{ marginTop: 14, background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.2)", borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "var(--amber)", display: "flex", gap: 8, alignItems: "center" }}>
            <span>⚠</span> {r.sampleWarning}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 28 }}>
        <button onClick={onBack} style={{ background: "var(--bg2)", border: "1px solid var(--border2)", borderRadius: 8, color: "var(--text2)", padding: "8px 14px", fontSize: 13, cursor: "pointer" }}>← Back</button>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 600, color: "var(--text1)" }}>{experiment.name}</h1>
          <div style={{ fontSize: 13, color: "var(--text3)", marginTop: 2 }}>{A} vs {B} · {totalN} entries · started {experiment.createdAt}</div>
        </div>
        <Btn variant="teal" small onClick={onLog} style={{ marginLeft: "auto" }}>+ Log entry</Btn>
      </div>

      {/* Headline */}
      {totalN > 0 && (
        <div className="fade-up" style={{ background: "linear-gradient(135deg, rgba(99,102,241,0.12), rgba(20,184,166,0.08))", border: "1px solid rgba(99,102,241,0.25)", borderRadius: 14, padding: "20px 24px", marginBottom: 24 }}>
          <div style={{ fontSize: 12, color: "var(--indigo2)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8, fontWeight: 500 }}>Primary result</div>
          <div style={{ fontSize: 20, fontWeight: 600, color: "var(--text1)", marginBottom: 6 }}>
            <span style={{ color: "var(--indigo2)" }}>{primary.winner}</span> wins on {primary.metric.replace(/_/g, " ")}
            {primary.lift !== null && <span style={{ color: "var(--green)", fontSize: 16, marginLeft: 10 }}>+{primary.lift}% lift</span>}
          </div>
          <div style={{ fontSize: 14, color: "var(--text2)" }}>
            {primary.confidence}% confidence · p={primary.pValue} ·
            {primary.significant ? <span style={{ color: "var(--green)", marginLeft: 5 }}>statistically significant</span> : <span style={{ color: "var(--amber)", marginLeft: 5 }}>not yet significant</span>}
          </div>
        </div>
      )}

      {totalN === 0 && (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--text3)", fontSize: 15 }}>
          No entries yet. Log your first result to see analysis.
        </div>
      )}

      {totalN > 0 && (
        <>
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text3)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>Primary metric</div>
          <MetricBlock r={primary} delay={0.05} />

          {others.length > 0 && (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text3)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12, marginTop: 24 }}>Supporting metrics</div>
              {others.map((r, i) => <MetricBlock key={r.metric} r={r} delay={0.1 + i * 0.05} />)}
            </>
          )}

          {/* Entry log */}
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text3)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12, marginTop: 24 }}>Entry log</div>
          <div className="fade-up stagger-4" style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 14, overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ padding: "12px 16px", color: "var(--text3)", fontWeight: 500, textAlign: "left" }}>Date</th>
                  <th style={{ padding: "12px 16px", color: "var(--text3)", fontWeight: 500, textAlign: "left" }}>Variant</th>
                  {experiment.metrics.map(m => (
                    <th key={m.name} style={{ padding: "12px 16px", color: "var(--text3)", fontWeight: 500, textAlign: "right", textTransform: "capitalize" }}>{m.name.replace(/_/g, " ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...experiment.entries].reverse().map((e, i) => (
                  <tr key={e.id} style={{ borderBottom: "1px solid var(--border)", background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)" }}>
                    <td style={{ padding: "11px 16px", color: "var(--text2)", fontFamily: "var(--mono)", fontSize: 12 }}>{e.date}</td>
                    <td style={{ padding: "11px 16px" }}><Badge color={experiment.variants.indexOf(e.variant) === 0 ? "indigo" : "teal"}>{e.variant}</Badge></td>
                    {experiment.metrics.map(m => (
                      <td key={m.name} style={{ padding: "11px 16px", color: "var(--text1)", fontFamily: "var(--mono)", fontSize: 13, textAlign: "right" }}>
                        {m.type === "binary" ? (e.metrics[m.name] === 1 ? "✓ yes" : "✗ no") : e.metrics[m.name]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// DASHBOARD
// ─────────────────────────────────────────────────────────────
function Dashboard({ experiments, onSelect, onCreate, onDelete }) {
  const confColor = c => c >= 95 ? "var(--green)" : c >= 80 ? "var(--amber)" : "var(--red)";

  return (
    <div>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 32 }}>
        <div>
          <div style={{ fontSize: 12, color: "var(--indigo2)", letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 500, marginBottom: 6 }}>Personal A/B Testing</div>
          <h1 style={{ fontSize: 28, fontWeight: 600, color: "var(--text1)", letterSpacing: "-0.02em" }}>Your experiments</h1>
        </div>
        <Btn onClick={onCreate}>+ New experiment</Btn>
      </div>

      {experiments.length === 0 && (
        <div style={{ textAlign: "center", padding: "80px 20px", color: "var(--text3)" }}>
          <div style={{ fontSize: 40, marginBottom: 16, opacity: 0.4 }}>⚗</div>
          <div style={{ fontSize: 16, marginBottom: 8 }}>No experiments yet</div>
          <div style={{ fontSize: 14 }}>Create your first A/B test to get started</div>
        </div>
      )}

      <div style={{ display: "grid", gap: 14 }}>
        {experiments.map((exp, i) => {
          const results = exp.entries.length > 0 ? analyzeExperiment(exp) : null;
          const primary = results?.find(r => r.primary) || results?.[0];
          const totalN  = exp.entries.length;

          return (
            <div key={exp.id} className="fade-up" style={{ animationDelay: `${i * 0.06}s`, background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: 14, padding: "20px 22px", cursor: "pointer", transition: "border-color 0.15s" }}
              onMouseEnter={e => e.currentTarget.style.borderColor = "var(--border2)"}
              onMouseLeave={e => e.currentTarget.style.borderColor = "var(--border)"}
              onClick={() => onSelect(exp.id)}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: "var(--text1)", marginBottom: 4 }}>{exp.name}</div>
                  <div style={{ fontSize: 13, color: "var(--text3)" }}>{exp.variants[0]} vs {exp.variants[1]} · {exp.metrics.length} metric{exp.metrics.length > 1 ? "s" : ""} · {totalN} entries</div>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }} onClick={e => e.stopPropagation()}>
                  {primary && primary.significant && <Badge color="green">significant</Badge>}
                  {primary && !primary.significant && totalN > 0 && <Badge color="amber">collecting</Badge>}
                  {totalN === 0 && <Badge color="indigo">new</Badge>}
                  <button onClick={() => onDelete(exp.id)} style={{ background: "none", border: "none", color: "var(--text3)", fontSize: 18, cursor: "pointer", padding: "2px 6px", borderRadius: 6 }} title="Delete">×</button>
                </div>
              </div>

              {primary && totalN > 0 ? (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                  <div style={{ background: "var(--bg2)", borderRadius: 8, padding: "10px 14px" }}>
                    <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.05em" }}>Winner</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "var(--indigo2)" }}>{primary.winner}</div>
                  </div>
                  <div style={{ background: "var(--bg2)", borderRadius: 8, padding: "10px 14px" }}>
                    <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.05em" }}>Lift</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "var(--green)", fontFamily: "var(--mono)" }}>{primary.lift !== null ? `+${primary.lift}%` : "—"}</div>
                  </div>
                  <div style={{ background: "var(--bg2)", borderRadius: 8, padding: "10px 14px" }}>
                    <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.05em" }}>Confidence</div>
                    <div style={{ fontSize: 14, fontWeight: 600, fontFamily: "var(--mono)", color: confColor(primary.confidence) }}>{primary.confidence}%</div>
                  </div>
                  <div style={{ background: "var(--bg2)", borderRadius: 8, padding: "10px 14px" }}>
                    <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.05em" }}>p-value</div>
                    <div style={{ fontSize: 14, fontWeight: 600, fontFamily: "var(--mono)", color: primary.pValue < 0.05 ? "var(--green)" : "var(--amber)" }}>{primary.pValue}</div>
                  </div>
                </div>
              ) : totalN === 0 ? (
                <div style={{ fontSize: 13, color: "var(--text3)", fontStyle: "italic" }}>No entries yet — click to open and start logging</div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// ROOT APP
// ─────────────────────────────────────────────────────────────
export default function App() {
  const [experiments, setExperiments]   = useState(() => loadExperiments());
  const [selectedId, setSelectedId]     = useState(null);
  const [showCreate, setShowCreate]     = useState(false);
  const [showLog, setShowLog]           = useState(false);

  useEffect(() => { saveExperiments(experiments); }, [experiments]);

  const selected = experiments.find(e => e.id === selectedId);

  const handleCreate = useCallback(exp => {
    setExperiments(prev => [...prev, exp]);
  }, []);

  const handleDelete = useCallback(id => {
    if (window.confirm("Delete this experiment?")) {
      setExperiments(prev => prev.filter(e => e.id !== id));
      if (selectedId === id) setSelectedId(null);
    }
  }, [selectedId]);

  const handleLog = useCallback(entry => {
    setExperiments(prev => prev.map(e => e.id === selectedId ? { ...e, entries: [...e.entries, entry] } : e));
  }, [selectedId]);

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
        {/* Header */}
        <div style={{ borderBottom: "1px solid var(--border)", background: "var(--bg1)" }}>
          <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px", height: 56, display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 28, height: 28, background: "var(--indigo)", borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>⚗</div>
              <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text1)", letterSpacing: "-0.01em" }}>ABTest</span>
            </div>
            {selectedId && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "var(--text3)" }}>
                <span>·</span>
                <span style={{ cursor: "pointer", color: "var(--text2)" }} onClick={() => setSelectedId(null)}>Experiments</span>
                <span>›</span>
                <span style={{ color: "var(--text1)" }}>{selected?.name}</span>
              </div>
            )}
            <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
              {selectedId && <Btn small variant="teal" onClick={() => setShowLog(true)}>+ Log entry</Btn>}
              {!selectedId && <Btn small onClick={() => setShowCreate(true)}>+ New experiment</Btn>}
            </div>
          </div>
        </div>

        {/* Main */}
        <div style={{ maxWidth: 900, margin: "0 auto", padding: "36px 24px" }}>
          {!selectedId ? (
            <Dashboard
              experiments={experiments}
              onSelect={setSelectedId}
              onCreate={() => setShowCreate(true)}
              onDelete={handleDelete}
            />
          ) : selected ? (
            <ResultsView
              experiment={selected}
              onBack={() => setSelectedId(null)}
              onLog={() => setShowLog(true)}
            />
          ) : null}
        </div>
      </div>

      {showCreate && (
        <CreateExperimentModal onClose={() => setShowCreate(false)} onCreate={handleCreate} />
      )}
      {showLog && selected && (
        <LogEntryModal experiment={selected} onClose={() => setShowLog(false)} onLog={handleLog} />
      )}
    </>
  );
}