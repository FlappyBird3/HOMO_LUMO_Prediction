import { useState, useEffect } from "react";

const API = "http://127.0.0.1:8000";

const EXAMPLES = [
  { smiles: "OC1CC1O", label: "cyclopropane-1,2-diol" },
  { smiles: "c1ccccc1", label: "benzene" },
  { smiles: "C", label: "methane" },
  { smiles: "CC(=O)O", label: "acetic acid" },
];

const COLORS = { C: "#3a3a3a", H: "#b8b8b8", O: "#d64545",
                 N: "#3b6fb5", F: "#4a9d5f" };

function MoleculeGraph({ atoms, bonds }) {
  const pad = 40;
  const size = 420;

  const xs = atoms.map((a) => a.x);
  const ys = atoms.map((a) => a.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const span = Math.max(maxX - minX, maxY - minY, 1);
  const scale = (size - 2 * pad) / span;

  // RDKit's y axis points up, SVG's points down, so y is flipped.
  const px = (a) => pad + (a.x - minX) * scale + ((span - (maxX - minX)) * scale) / 2;
  const py = (a) => pad + (maxY - a.y) * scale + ((span - (maxY - minY)) * scale) / 2;

  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", maxWidth: 420 }}>
      {bonds.map((b, i) => {
        const a1 = atoms[b.source], a2 = atoms[b.target];
        const x1 = px(a1), y1 = py(a1), x2 = px(a2), y2 = py(a2);

        // Offset perpendicular to the bond so double bonds draw as two lines.
        const dx = x2 - x1, dy = y2 - y1;
        const len = Math.hypot(dx, dy) || 1;
        const ox = (-dy / len) * 3, oy = (dx / len) * 3;
        const n = b.order === "double" ? 2 : b.order === "triple" ? 3 : 1;

        return [...Array(n)].map((_, k) => {
          const shift = n === 1 ? 0 : k - (n - 1) / 2;
          return (
            <line key={`${i}-${k}`}
              x1={x1 + ox * shift * 2} y1={y1 + oy * shift * 2}
              x2={x2 + ox * shift * 2} y2={y2 + oy * shift * 2}
              stroke="#999" strokeWidth="1.5" />
          );
        });
      })}

      {atoms.map((a) => (
        <g key={a.index}>
          <circle cx={px(a)} cy={py(a)} r={a.element === "H" ? 9 : 13}
                  fill={COLORS[a.element] || "#888"} />
          <text x={px(a)} y={py(a)} textAnchor="middle" dominantBaseline="central"
                fill="#fff" fontSize={a.element === "H" ? 8 : 10}
                fontFamily="system-ui">
            {a.element}{a.index}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function App() {
  const [smiles, setSmiles] = useState("OC1CC1O");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Wake the API on page load so the first real request isn't slow.
  useEffect(() => {
    fetch(`${API}/health`).catch(() => {});
  }, []);

  async function predict(input) {
    const query = (input ?? smiles).trim();
    if (!query) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: query }),
      });
      const data = await res.json();
      if (data.ok) setResult(data);
      else setError(data.error);
    } catch {
      setError("Could not reach the server. Is the API running on port 8000?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={S.page}>
      <h1 style={S.title}>HOMO–LUMO Gap Prediction</h1>
      <p style={S.sub}>
        A graph neural network trained on 130,000 molecules from QM9.
        Enter a molecule as a SMILES string.
      </p>

      <div style={S.row}>
        <input
          style={S.input}
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && predict()}
          placeholder="e.g. OC1CC1O"
        />
        <button style={S.button} onClick={() => predict()} disabled={loading}>
          {loading ? "…" : "Predict"}
        </button>
      </div>

      <div style={S.examples}>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.smiles}
            style={S.chip}
            onClick={() => {
              setSmiles(ex.smiles);
              predict(ex.smiles);
            }}
          >
            {ex.label}
          </button>
        ))}
      </div>

      {error && <div style={S.error}>{error}</div>}

      {result && (
        <div style={S.result}>
          <div style={S.gap}>{result.gap_ev.toFixed(2)} eV</div>
          <div style={S.formula}>{result.structure.formula}</div>

          <div style={S.stats}>
            <span>{result.structure.num_atoms} atoms</span>
            <span>{result.structure.num_bonds} bonds</span>
            <span>{result.structure.num_rings} rings</span>
          </div>

          <div style={{ marginTop: 20 }}>
            <MoleculeGraph
              atoms={result.structure.atoms}
              bonds={result.structure.bonds}
            />
          </div>

          <p style={S.note}>{result.note}</p>
        </div>
      )}
    </div>
  );
}

const S = {
  page: { maxWidth: 720, margin: "0 auto", padding: "48px 24px",
          fontFamily: "system-ui, sans-serif", color: "#1a1a1a" },
  title: { fontSize: 28, fontWeight: 600, margin: "0 0 8px" },
  sub: { color: "#666", fontSize: 15, lineHeight: 1.5, margin: "0 0 28px" },
  row: { display: "flex", gap: 8 },
  input: { flex: 1, padding: "10px 12px", fontSize: 15, borderRadius: 6,
           border: "1px solid #ccc", fontFamily: "ui-monospace, monospace" },
  button: { padding: "10px 20px", fontSize: 15, borderRadius: 6, border: "none",
            background: "#1a1a1a", color: "#fff", cursor: "pointer" },
  examples: { display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 },
  chip: { padding: "5px 11px", fontSize: 13, borderRadius: 14,
          border: "1px solid #ddd", background: "#fff", cursor: "pointer",
          color: "#555" },
  error: { marginTop: 24, padding: "12px 14px", borderRadius: 6,
           background: "#fdf0ed", color: "#a03626", fontSize: 14 },
  result: { marginTop: 32, padding: 24, borderRadius: 10, border: "1px solid #e5e5e5" },
  gap: { fontSize: 42, fontWeight: 600, letterSpacing: -1, color: "#fff" },
  formula: { fontSize: 17, color: "#666", marginTop: 2 },
  stats: { display: "flex", gap: 18, marginTop: 16, fontSize: 13, color: "#888" },
  bondList: { marginTop: 20, display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 6 },
  bond: { fontSize: 13, fontFamily: "ui-monospace, monospace",
          display: "flex", justifyContent: "space-between",
          padding: "5px 9px", background: "#fafafa", borderRadius: 4 },
  order: { color: "#999", fontSize: 11 },
  note: { marginTop: 22, fontSize: 12, color: "#999", lineHeight: 1.5 },
};