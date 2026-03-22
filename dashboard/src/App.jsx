import { useState, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REFRESH_INTERVAL = 5000;

function App() {
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/stats`);
      if (res.ok) setStats(await res.json());
    } catch {}
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/logs?limit=20`);
      if (res.ok) setLogs(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    fetchStats();
    fetchLogs();
    const interval = setInterval(() => {
      fetchStats();
      fetchLogs();
    }, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchStats, fetchLogs]);

  const sendQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
      setQuery("");
      fetchStats();
      fetchLogs();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>
          <span style={styles.logo}>⚡</span> ModelRouter
        </h1>
        <p style={styles.subtitle}>Intelligent LLM Routing Dashboard</p>
      </header>

      {/* Stats Cards */}
      {stats && stats.total_requests > 0 && (
        <div style={styles.statsGrid}>
          <StatCard
            label="Total Requests"
            value={stats.total_requests}
            color="#6366f1"
          />
          <StatCard
            label="Cost Savings"
            value={`${stats.savings_pct}%`}
            sub={`$${stats.savings_usd.toFixed(4)} saved`}
            color="#10b981"
          />
          <StatCard
            label="Avg Latency"
            value={`${stats.avg_latency_ms.toFixed(0)}ms`}
            color="#f59e0b"
          />
          <StatCard
            label="Error Rate"
            value={`${stats.error_rate_pct}%`}
            color={stats.error_rate_pct > 5 ? "#ef4444" : "#10b981"}
          />
        </div>
      )}

      {/* Distribution Bars */}
      {stats && stats.total_requests > 0 && (
        <div style={styles.row}>
          <DistributionCard
            title="Model Distribution"
            data={stats.model_distribution}
            total={stats.total_requests}
            colors={modelColors}
          />
          <DistributionCard
            title="Difficulty Distribution"
            data={stats.difficulty_distribution}
            total={stats.total_requests}
            colors={difficultyColors}
          />
        </div>
      )}

      {/* Query Input */}
      <div style={styles.card}>
        <h2 style={styles.cardTitle}>Test a Query</h2>
        <div style={styles.inputRow}>
          <input
            style={styles.input}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendQuery()}
            placeholder="Type a query to classify and route..."
            disabled={loading}
          />
          <button
            style={{
              ...styles.button,
              opacity: loading ? 0.6 : 1,
            }}
            onClick={sendQuery}
            disabled={loading}
          >
            {loading ? "Routing..." : "Send"}
          </button>
        </div>
        {error && <p style={styles.error}>Error: {error}</p>}
      </div>

      {/* Result */}
      {result && (
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Routing Result</h2>
          <div style={styles.resultGrid}>
            <ResultBadge
              label="Difficulty"
              value={result.routing.difficulty_label}
              sub={`Score: ${result.routing.difficulty_score}`}
              color={difficultyColors[result.routing.difficulty_label] || "#6b7280"}
            />
            <ResultBadge
              label="Model"
              value={result.routing.routed_to}
              sub={`Score: ${result.routing.routing_score}`}
              color={modelColors[result.routing.routed_to] || "#6b7280"}
            />
            <ResultBadge
              label="Domain"
              value={result.routing.domain}
              color="#8b5cf6"
            />
            <ResultBadge
              label="Latency"
              value={`${result.metadata.latency_ms.toFixed(0)}ms`}
              color="#f59e0b"
            />
            <ResultBadge
              label="Cost"
              value={`$${result.metadata.estimated_cost_usd.toFixed(6)}`}
              color="#10b981"
            />
            <ResultBadge
              label="Tokens"
              value={`${result.metadata.tokens_in} → ${result.metadata.tokens_out}`}
              color="#6366f1"
            />
          </div>

          {/* ML Confidence */}
          {result.routing.features.ml_probs && (
            <div style={{ marginTop: 16 }}>
              <p style={styles.smallLabel}>ML Confidence</p>
              <div style={styles.probsRow}>
                {["trivial", "easy", "medium", "hard"].map((label, i) => (
                  <div key={label} style={styles.probBar}>
                    <div style={styles.probLabel}>{label}</div>
                    <div style={styles.probTrack}>
                      <div
                        style={{
                          ...styles.probFill,
                          width: `${(result.routing.features.ml_probs[i] * 100).toFixed(0)}%`,
                          backgroundColor:
                            difficultyColors[label === "trivial" ? "easy" : label] || "#6b7280",
                        }}
                      />
                    </div>
                    <div style={styles.probVal}>
                      {(result.routing.features.ml_probs[i] * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={styles.responseBox}>
            <p style={styles.smallLabel}>Response</p>
            <p style={styles.responseText}>{result.response}</p>
          </div>
        </div>
      )}

      {/* Recent Logs */}
      {logs.length > 0 && (
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Recent Requests</h2>
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead>
                <tr>
                  {["Time", "Query", "Difficulty", "Model", "Latency", "Cost"].map(
                    (h) => (
                      <th key={h} style={styles.th}>{h}</th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {[...logs].reverse().map((log) => (
                  <tr key={log.query_id} style={styles.tr}>
                    <td style={styles.td}>
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </td>
                    <td style={{ ...styles.td, maxWidth: 300 }}>
                      {log.query.slice(0, 80)}
                      {log.query.length > 80 ? "..." : ""}
                    </td>
                    <td style={styles.td}>
                      <span
                        style={{
                          ...styles.badge,
                          backgroundColor:
                            difficultyColors[log.difficulty_label] || "#6b7280",
                        }}
                      >
                        {log.difficulty_label} ({log.difficulty_score})
                      </span>
                    </td>
                    <td style={styles.td}>
                      <span
                        style={{
                          ...styles.badge,
                          backgroundColor:
                            modelColors[log.routed_to] || "#6b7280",
                        }}
                      >
                        {log.routed_to.replace("-groq", "").replace("claude-sonnet-bedrock", "claude")}
                      </span>
                    </td>
                    <td style={styles.td}>{log.latency_ms.toFixed(0)}ms</td>
                    <td style={styles.td}>
                      ${log.estimated_cost_usd.toFixed(6)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <footer style={styles.footer}>
        ModelRouter v0.1 · DistilBERT Classifier · Groq + AWS Bedrock
      </footer>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────── */

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ ...styles.statCard, borderTop: `3px solid ${color}` }}>
      <p style={styles.statLabel}>{label}</p>
      <p style={{ ...styles.statValue, color }}>{value}</p>
      {sub && <p style={styles.statSub}>{sub}</p>}
    </div>
  );
}

function DistributionCard({ title, data, total, colors }) {
  return (
    <div style={{ ...styles.card, flex: 1 }}>
      <h3 style={styles.cardTitle}>{title}</h3>
      {Object.entries(data).map(([key, count]) => (
        <div key={key} style={{ marginBottom: 8 }}>
          <div style={styles.distHeader}>
            <span style={styles.distLabel}>{key}</span>
            <span style={styles.distCount}>
              {count} ({((count / total) * 100).toFixed(0)}%)
            </span>
          </div>
          <div style={styles.distTrack}>
            <div
              style={{
                ...styles.distFill,
                width: `${(count / total) * 100}%`,
                backgroundColor: colors[key] || "#6b7280",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ResultBadge({ label, value, sub, color }) {
  return (
    <div style={styles.resultBadge}>
      <p style={styles.smallLabel}>{label}</p>
      <p style={{ ...styles.resultValue, color }}>{value}</p>
      {sub && <p style={styles.resultSub}>{sub}</p>}
    </div>
  );
}

/* ── Colors ─────────────────────────────────────────────── */

const modelColors = {
  "llama-8b-groq": "#10b981",
  "llama-70b-groq": "#6366f1",
  "claude-sonnet-bedrock": "#f59e0b",
};

const difficultyColors = {
  easy: "#10b981",
  medium: "#f59e0b",
  hard: "#ef4444",
};

/* ── Styles ─────────────────────────────────────────────── */

const styles = {
  app: {
    maxWidth: 960,
    margin: "0 auto",
    padding: "24px 16px",
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: "#e2e8f0",
    backgroundColor: "#0f172a",
    minHeight: "100vh",
  },
  header: { textAlign: "center", marginBottom: 32 },
  title: { fontSize: 28, fontWeight: 700, margin: 0, color: "#f8fafc" },
  logo: { fontSize: 32 },
  subtitle: { color: "#94a3b8", margin: "4px 0 0", fontSize: 14 },
  statsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    backgroundColor: "#1e293b",
    borderRadius: 8,
    padding: 16,
  },
  statLabel: { margin: 0, fontSize: 12, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 0.5 },
  statValue: { margin: "4px 0", fontSize: 28, fontWeight: 700 },
  statSub: { margin: 0, fontSize: 12, color: "#64748b" },
  row: { display: "flex", gap: 12, marginBottom: 16 },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
  },
  cardTitle: { margin: "0 0 12px", fontSize: 16, fontWeight: 600, color: "#f8fafc" },
  inputRow: { display: "flex", gap: 8 },
  input: {
    flex: 1,
    padding: "10px 12px",
    borderRadius: 6,
    border: "1px solid #334155",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    fontSize: 14,
    outline: "none",
  },
  button: {
    padding: "10px 20px",
    borderRadius: 6,
    border: "none",
    backgroundColor: "#6366f1",
    color: "#fff",
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
  },
  error: { color: "#ef4444", fontSize: 13, margin: "8px 0 0" },
  resultGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
    gap: 10,
  },
  resultBadge: {
    backgroundColor: "#0f172a",
    borderRadius: 6,
    padding: 10,
    textAlign: "center",
  },
  resultValue: { margin: "2px 0", fontSize: 16, fontWeight: 700 },
  resultSub: { margin: 0, fontSize: 11, color: "#64748b" },
  smallLabel: {
    margin: 0,
    fontSize: 11,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  probsRow: { display: "flex", flexDirection: "column", gap: 4, marginTop: 6 },
  probBar: { display: "flex", alignItems: "center", gap: 8 },
  probLabel: { width: 56, fontSize: 12, color: "#94a3b8" },
  probTrack: {
    flex: 1,
    height: 8,
    backgroundColor: "#0f172a",
    borderRadius: 4,
    overflow: "hidden",
  },
  probFill: { height: "100%", borderRadius: 4, transition: "width 0.3s" },
  probVal: { width: 44, fontSize: 12, color: "#94a3b8", textAlign: "right" },
  responseBox: {
    marginTop: 16,
    padding: 12,
    backgroundColor: "#0f172a",
    borderRadius: 6,
  },
  responseText: {
    margin: "6px 0 0",
    fontSize: 13,
    lineHeight: 1.6,
    color: "#cbd5e1",
    whiteSpace: "pre-wrap",
  },
  tableWrap: { overflowX: "auto" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: {
    textAlign: "left",
    padding: "8px 10px",
    borderBottom: "1px solid #334155",
    color: "#94a3b8",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  tr: { borderBottom: "1px solid #1e293b" },
  td: { padding: "8px 10px", color: "#cbd5e1" },
  badge: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 11,
    fontWeight: 600,
    color: "#fff",
  },
  distHeader: { display: "flex", justifyContent: "space-between", marginBottom: 4 },
  distLabel: { fontSize: 13, color: "#e2e8f0" },
  distCount: { fontSize: 12, color: "#94a3b8" },
  distTrack: {
    height: 6,
    backgroundColor: "#0f172a",
    borderRadius: 3,
    overflow: "hidden",
  },
  distFill: { height: "100%", borderRadius: 3, transition: "width 0.5s" },
  footer: {
    textAlign: "center",
    fontSize: 12,
    color: "#475569",
    marginTop: 32,
    paddingTop: 16,
    borderTop: "1px solid #1e293b",
  },
};

export default App;