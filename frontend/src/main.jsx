import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ClipboardCheck,
  FileJson,
  FileText,
  HeartPulse,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import "./styles.css";

const SAMPLE_NOTES = [
  {
    label: "Respiratory",
    note: "Patient reports fever for 3 days, mild cough, fatigue. Temperature 101 F. No prior conditions.",
  },
  {
    label: "Stroke",
    note:
      "Patient is a 65-year-old female brought to the clinic with complaints of sudden onset weakness on the right side of the body since morning. Family reports slurred speech and difficulty in understanding commands. No history of head injury.\n\nPast medical history includes hypertension and atrial fibrillation. Patient is on anticoagulant therapy but compliance is uncertain.\n\nVital signs: BP 170/105 mmHg, HR 88 bpm, irregular rhythm noted.\n\nOn examination, right-sided motor weakness (power 3/5), facial droop present. No seizure activity observed.",
  },
  {
    label: "Heart Failure",
    note:
      "Patient is a 62-year-old male presenting with swelling in both legs and breathlessness for the past 2 weeks. Symptoms worsen on lying flat and improve on sitting up.\n\nReports fatigue and reduced exercise tolerance.\n\nPast history of hypertension and coronary artery disease.\n\nVital signs: BP 140/90 mmHg, HR 95 bpm, SpO2 94%.\n\nOn examination, bilateral pitting edema and basal lung crackles noted.",
  },
  {
    label: "Ankle Injury",
    note:
      "Patient is a 27-year-old male presenting with left ankle pain following a sports injury 24 hours ago. Reports twisting injury while running. Immediate swelling and difficulty bearing weight noted.\n\nPain is localized around lateral ankle, worsens with movement.\n\nVital signs stable.\n\nOn examination, swelling and tenderness over lateral malleolus. Range of motion limited due to pain.",
  },
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [note, setNote] = React.useState(SAMPLE_NOTES[0].note);
  const [report, setReport] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  const noteStats = React.useMemo(() => getNoteStats(note), [note]);

  async function analyzeNote() {
    setLoading(true);
    setError("");
    setReport(null);
    try {
      const response = await fetch(`${API_BASE_URL}/analyze-note`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note }),
      });
      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.detail || "Unable to analyze note.");
      }
      setReport(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Clinical Decision Support</p>
            <h1>Clinical Workflow Assistant</h1>
          </div>
          <div className="status-cluster" aria-label="service status">
            <span className="status-pill">
              <Activity size={16} />
              Pipeline Ready
            </span>
            <span className="api-pill">{formatApiHost(API_BASE_URL)}</span>
          </div>
        </header>

        <div className="content-grid">
          <section className="input-panel">
            <div className="panel-title">
              <Stethoscope size={20} />
              <h2>Clinical Note</h2>
            </div>

            <div className="sample-row" aria-label="sample notes">
              {SAMPLE_NOTES.map((sample) => (
                <button
                  className="sample-button"
                  key={sample.label}
                  onClick={() => {
                    setNote(sample.note);
                    setReport(null);
                    setError("");
                  }}
                  type="button"
                >
                  {sample.label}
                </button>
              ))}
            </div>

            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Paste raw clinical notes here..."
            />

            <div className="input-footer">
              <div className="note-stats">
                <span>{noteStats.words} words</span>
                <span>{noteStats.characters} chars</span>
              </div>
              <div className="actions">
                <button onClick={analyzeNote} disabled={loading || !note.trim()}>
                  {loading ? <Loader2 className="spin" size={18} /> : <ClipboardCheck size={18} />}
                  {loading ? "Analyzing" : "Analyze"}
                </button>
                <button
                  className="secondary"
                  onClick={() => {
                    setNote("");
                    setReport(null);
                    setError("");
                  }}
                  type="button"
                >
                  <RotateCcw size={17} />
                  Clear
                </button>
              </div>
            </div>
            {error ? <p className="error">{error}</p> : null}
          </section>

          <section className="result-panel">
            <div className="panel-title result-title">
              <FileText size={20} />
              <h2>Report Output</h2>
            </div>
            {!report ? <EmptyState loading={loading} /> : <ReportView report={report} />}
          </section>
        </div>
      </section>
    </main>
  );
}

function EmptyState({ loading }) {
  return (
    <div className="empty-state">
      {loading ? <Loader2 className="spin" size={26} /> : <HeartPulse size={30} />}
      <p>{loading ? "Analyzing note and preparing structured output." : "Run an analysis to view extracted findings, differential, workup, medication options, and safety checks."}</p>
    </div>
  );
}

function ReportView({ report }) {
  const recommendations = report.recommendations || {};
  const priority = getPriority(recommendations);

  return (
    <div className="report-stack">
      <section className={`priority-banner ${priority.level}`}>
        {priority.level === "urgent" ? <AlertTriangle size={19} /> : <ShieldCheck size={19} />}
        <div>
          <strong>{priority.title}</strong>
          <p>{priority.copy}</p>
        </div>
      </section>

      <section className="summary-panel">
        <div>
          <span className="section-kicker">Summary</span>
          <p>{report.summary}</p>
        </div>
        <div className="summary-meta">
          <span>{formatDate(report.created_at)}</span>
          <span>ID {String(report.id).slice(0, 8)}</span>
        </div>
      </section>

      <section className="section-group">
        <SectionHeader icon={<HeartPulse size={18} />} title="Clinical Reasoning" />
        <div className="reasoning-grid">
          <ListBlock title="Impression" items={recommendations.clinical_impression} />
          <ListBlock title="Differential" items={recommendations.possible_conditions} />
        </div>
      </section>

      <section className="section-group">
        <SectionHeader icon={<ClipboardCheck size={18} />} title="Next Steps" />
        <div className="reasoning-grid">
          <ListBlock title="Tests / Workup" items={recommendations.recommended_tests} />
          <ListBlock title="Medication Options" items={recommendations.medications} />
        </div>
      </section>

      <section className="section-group safety-grid">
        <ListBlock title="Medication Cautions" items={recommendations.medication_cautions} tone="caution" />
        <ListBlock title="Red Flags" items={recommendations.red_flags} tone="danger" />
        <ListBlock title="Missing Information" items={recommendations.missing_information} />
        <ListBlock title="Follow-up" items={recommendations.follow_ups} />
      </section>

      <section className="section-group">
        <SectionHeader icon={<Stethoscope size={18} />} title="Extracted Information" />
        <ExtractedInfo extracted={report.extracted_info} />
      </section>

      <details className="readable-report">
        <summary>
          <FileText size={17} />
          Readable Report
        </summary>
        <pre>{report.report}</pre>
      </details>

      <details className="json-details">
        <summary>
          <FileJson size={17} />
          Structured JSON
        </summary>
        <pre>{JSON.stringify(report, null, 2)}</pre>
      </details>
    </div>
  );
}

function SectionHeader({ icon, title }) {
  return (
    <div className="section-header">
      {icon}
      <h3>{title}</h3>
    </div>
  );
}

function ListBlock({ title, items, tone = "default" }) {
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  return (
    <div className={`list-block ${tone}`}>
      <strong>{title}</strong>
      {values.length ? (
        <ul>
          {values.map((item, index) => (
            <li key={`${title}-${index}`}>{renderValue(item)}</li>
          ))}
        </ul>
      ) : (
        <span className="muted">None documented</span>
      )}
    </div>
  );
}

function ExtractedInfo({ extracted }) {
  const entries = Object.entries(extracted || {}).filter(([key]) => key !== "medical_references");
  const references = extracted?.medical_references || [];

  return (
    <div className="extracted-layout">
      <div className="info-grid">
        {entries.map(([key, value]) => (
          <ListBlock key={key} title={formatLabel(key)} items={Array.isArray(value) ? value : [value]} />
        ))}
      </div>
      {references.length ? (
        <details className="reference-details">
          <summary>Medical references ({references.length})</summary>
          <ul>
            {references.map((item, index) => (
              <li key={`reference-${index}`}>{renderValue(item)}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function renderValue(value) {
  if (typeof value === "object" && value !== null) {
    return Object.entries(value)
      .filter(([, item]) => item !== null && item !== "")
      .map(([key, item]) => `${formatLabel(key)}: ${item}`)
      .join(", ");
  }
  return value;
}

function formatLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function getNoteStats(value) {
  const trimmed = value.trim();
  return {
    words: trimmed ? trimmed.split(/\s+/).length : 0,
    characters: value.length,
  };
}

function getPriority(recommendations) {
  const text = [
    ...(recommendations.red_flags || []),
    ...(recommendations.recommended_tests || []),
    ...(recommendations.medications || []),
  ]
    .join(" ")
    .toLowerCase();

  if (/(emergency|urgent|stroke|sepsis|anaphylaxis|severe|inability to bear weight|chest pain)/.test(text)) {
    return {
      level: "urgent",
      title: "High-priority review",
      copy: "The output contains urgent safety signals. Escalation and clinician verification should come before routine follow-up.",
    };
  }

  return {
    level: "routine",
    title: "Clinician review required",
    copy: "Use the structured output as decision support and verify medications, tests, and safety checks.",
  };
}

function formatDate(value) {
  if (!value) return "New report";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatApiHost(value) {
  try {
    return new URL(value).host;
  } catch {
    return value;
  }
}

createRoot(document.getElementById("root")).render(<App />);
