import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, ClipboardCheck, FileText, Loader2, Stethoscope } from "lucide-react";
import "./styles.css";

const SAMPLE_NOTE =
  "Patient reports fever for 3 days, mild cough, fatigue. Temperature 101 F. No prior conditions.";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [note, setNote] = React.useState(SAMPLE_NOTE);
  const [report, setReport] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

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
            <p className="eyebrow">Healthcare AI Agent MVP</p>
            <h1>Clinical Workflow Assistant</h1>
          </div>
          <div className="status-pill">
            <Activity size={16} />
            Agent Pipeline Ready
          </div>
        </header>

        <div className="content-grid">
          <section className="input-panel">
            <div className="panel-title">
              <Stethoscope size={20} />
              <h2>Clinical Note</h2>
            </div>
            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Paste raw clinical notes here..."
            />
            <div className="actions">
              <button onClick={analyzeNote} disabled={loading || !note.trim()}>
                {loading ? <Loader2 className="spin" size={18} /> : <ClipboardCheck size={18} />}
                {loading ? "Analyzing" : "Analyze"}
              </button>
              <button className="secondary" onClick={() => setNote(SAMPLE_NOTE)}>
                Load Sample
              </button>
            </div>
            {error ? <p className="error">{error}</p> : null}
          </section>

          <section className="result-panel">
            <div className="panel-title">
              <FileText size={20} />
              <h2>Report Output</h2>
            </div>
            {!report ? (
              <div className="empty-state">
                <p>Run the analysis to see summary, extracted medical information, and recommended next steps.</p>
              </div>
            ) : (
              <ReportView report={report} />
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

function ReportView({ report }) {
  return (
    <div className="report-stack">
      <section className="report-block">
        <h3>Summary</h3>
        <p>{report.summary}</p>
      </section>

      <section className="report-block">
        <h3>Extracted Info</h3>
        <InfoGrid extracted={report.extracted_info} />
      </section>

      <section className="report-block">
        <h3>Recommendations</h3>
        <InfoGrid extracted={report.recommendations} />
      </section>

      <section className="report-block">
        <h3>Readable Report</h3>
        <pre>{report.report}</pre>
      </section>

      <details className="json-details">
        <summary>Structured JSON</summary>
        <pre>{JSON.stringify(report, null, 2)}</pre>
      </details>
    </div>
  );
}

function InfoGrid({ extracted }) {
  return (
    <div className="info-grid">
      {Object.entries(extracted).map(([key, value]) => (
        <div className="info-item" key={key}>
          <strong>{formatLabel(key)}</strong>
          {Array.isArray(value) ? (
            value.length ? (
              <ul>
                {value.map((item, index) => (
                  <li key={`${key}-${index}`}>{renderValue(item)}</li>
                ))}
              </ul>
            ) : (
              <span className="muted">None documented</span>
            )
          ) : (
            <span>{String(value)}</span>
          )}
        </div>
      ))}
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

createRoot(document.getElementById("root")).render(<App />);
