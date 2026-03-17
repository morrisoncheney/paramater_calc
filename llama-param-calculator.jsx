import { useState, useMemo } from "react";

const DEFAULTS = {
  vocab_size: 128256,
  hidden_size: 4096,
  intermediate_size: 11008,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: null,
  tie_word_embeddings: false,
};

function formatNumber(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toLocaleString();
}

function Field({ label, value, onChange, defaultVal, hint, type = "number" }) {
  const isNull = value === null || value === "";
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
        <label style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: "#c5f467", letterSpacing: 0.3 }}>
          {label}
        </label>
        <span style={{ fontSize: 11, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace" }}>
          Default: {String(defaultVal)}
        </span>
      </div>
      {type === "toggle" ? (
        <button
          onClick={() => onChange(!value)}
          style={{
            background: value ? "#c5f467" : "#1e2028",
            color: value ? "#0d0f14" : "#6b7280",
            border: "1px solid " + (value ? "#c5f467" : "#2a2d38"),
            borderRadius: 6,
            padding: "6px 16px",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          {value ? "true" : "false"}
        </button>
      ) : (
        <input
          type="text"
          value={isNull ? "" : value}
          placeholder="null"
          onChange={(e) => {
            const v = e.target.value;
            if (v === "" || v === "null") onChange(null);
            else if (!isNaN(Number(v))) onChange(Number(v));
          }}
          style={{
            width: "100%",
            background: "#1e2028",
            border: "1px solid #2a2d38",
            borderRadius: 6,
            padding: "8px 12px",
            color: "#e2e8f0",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 14,
            outline: "none",
            boxSizing: "border-box",
            transition: "border-color 0.15s",
          }}
          onFocus={(e) => (e.target.style.borderColor = "#c5f467")}
          onBlur={(e) => (e.target.style.borderColor = "#2a2d38")}
        />
      )}
      {hint && (
        <div style={{ fontSize: 11, color: "#4b5563", marginTop: 3, fontStyle: "italic" }}>{hint}</div>
      )}
    </div>
  );
}

function BreakdownRow({ label, value, pct, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 12, color: "#9ca3af", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
        <span style={{ fontSize: 12, color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace" }}>
          {formatNumber(value)} <span style={{ color: "#6b7280" }}>({pct.toFixed(1)}%)</span>
        </span>
      </div>
      <div style={{ height: 4, background: "#1e2028", borderRadius: 2, overflow: "hidden" }}>
        <div
          style={{
            width: `${Math.max(pct, 0.5)}%`,
            height: "100%",
            background: color,
            borderRadius: 2,
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

export default function LlamaParamCalculator() {
  const [config, setConfig] = useState({ ...DEFAULTS });

  const set = (key) => (val) => setConfig((c) => ({ ...c, [key]: val }));

  const breakdown = useMemo(() => {
    const V = config.vocab_size || 0;
    const H = config.hidden_size || 0;
    const I = config.intermediate_size || 0;
    const L = config.num_hidden_layers || 0;
    const heads = config.num_attention_heads || 1;
    const kvHeads = config.num_key_value_heads ?? heads;
    const headDim = Math.floor(H / heads);
    const tied = config.tie_word_embeddings;

    const embedding = V * H;

    const qProj = H * H;
    const kProj = H * (kvHeads * headDim);
    const vProj = H * (kvHeads * headDim);
    const oProj = H * H;
    const attnPerLayer = qProj + kProj + vProj + oProj;

    const mlpPerLayer = 3 * H * I; // gate + up + down (SwiGLU)

    const normPerLayer = 2 * H; // input_layernorm + post_attention_layernorm

    const perLayer = attnPerLayer + mlpPerLayer + normPerLayer;
    const allLayers = L * perLayer;

    const finalNorm = H;

    const lmHead = tied ? 0 : V * H;

    const total = embedding + allLayers + finalNorm + lmHead;

    return {
      embedding,
      attnTotal: L * attnPerLayer,
      mlpTotal: L * mlpPerLayer,
      normTotal: L * normPerLayer + finalNorm,
      lmHead,
      total,
      headDim,
      kvHeads,
      perLayer,
    };
  }, [config]);

  const t = breakdown.total || 1;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0d0f14",
        color: "#e2e8f0",
        fontFamily: "'IBM Plex Sans', -apple-system, sans-serif",
        padding: "32px 20px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#c5f467",
                boxShadow: "0 0 8px #c5f46788",
              }}
            />
            <h1
              style={{
                margin: 0,
                fontSize: 20,
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 600,
                letterSpacing: -0.5,
              }}
            >
              LLaMA Parameter Calculator
            </h1>
          </div>
          <p style={{ margin: 0, fontSize: 13, color: "#6b7280", paddingLeft: 18 }}>
            Estimate parameter counts for LLaMA-style transformer architectures
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          {/* Config panel */}
          <div>
            <div
              style={{
                background: "#14161d",
                borderRadius: 10,
                padding: 20,
                border: "1px solid #1e2028",
              }}
            >
              <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 16 }}>
                Model Config
              </div>
              <Field label="vocab_size" value={config.vocab_size} onChange={set("vocab_size")} defaultVal={128256} />
              <Field label="hidden_size" value={config.hidden_size} onChange={set("hidden_size")} defaultVal={4096} />
              <Field label="intermediate_size" value={config.intermediate_size} onChange={set("intermediate_size")} defaultVal={11008} />
              <Field label="num_hidden_layers" value={config.num_hidden_layers} onChange={set("num_hidden_layers")} defaultVal={32} />
              <Field label="num_attention_heads" value={config.num_attention_heads} onChange={set("num_attention_heads")} defaultVal={32} />
              <Field
                label="num_key_value_heads"
                value={config.num_key_value_heads}
                onChange={set("num_key_value_heads")}
                defaultVal="null"
                hint="null → defaults to num_attention_heads (MHA)"
              />
              <Field
                label="tie_word_embeddings"
                value={config.tie_word_embeddings}
                onChange={set("tie_word_embeddings")}
                defaultVal="false"
                type="toggle"
                hint="Shares embedding and LM head weights"
              />

              <button
                onClick={() => setConfig({ ...DEFAULTS })}
                style={{
                  marginTop: 8,
                  background: "transparent",
                  border: "1px solid #2a2d38",
                  borderRadius: 6,
                  padding: "6px 14px",
                  color: "#6b7280",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  cursor: "pointer",
                }}
              >
                Reset to LLaMA 3.2 1B defaults
              </button>
            </div>
          </div>

          {/* Results panel */}
          <div>
            {/* Total */}
            <div
              style={{
                background: "#14161d",
                borderRadius: 10,
                padding: 20,
                border: "1px solid #c5f46733",
                marginBottom: 16,
              }}
            >
              <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 8 }}>
                Total Parameters
              </div>
              <div
                style={{
                  fontSize: 36,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontWeight: 700,
                  color: "#c5f467",
                  letterSpacing: -1,
                }}
              >
                {formatNumber(breakdown.total)}
              </div>
              <div style={{ fontSize: 12, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace", marginTop: 2 }}>
                {breakdown.total.toLocaleString()} exact
              </div>
            </div>

            {/* Breakdown */}
            <div
              style={{
                background: "#14161d",
                borderRadius: 10,
                padding: 20,
                border: "1px solid #1e2028",
                marginBottom: 16,
              }}
            >
              <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 16 }}>
                Breakdown
              </div>
              <BreakdownRow label="Embeddings" value={breakdown.embedding} pct={(breakdown.embedding / t) * 100} color="#c5f467" />
              <BreakdownRow label="Attention (all layers)" value={breakdown.attnTotal} pct={(breakdown.attnTotal / t) * 100} color="#67b8f4" />
              <BreakdownRow label="MLP / FFN (all layers)" value={breakdown.mlpTotal} pct={(breakdown.mlpTotal / t) * 100} color="#f4a867" />
              <BreakdownRow label="RMSNorm" value={breakdown.normTotal} pct={(breakdown.normTotal / t) * 100} color="#a78bfa" />
              {breakdown.lmHead > 0 && (
                <BreakdownRow label="LM Head (untied)" value={breakdown.lmHead} pct={(breakdown.lmHead / t) * 100} color="#f467a8" />
              )}
            </div>

            {/* Derived info */}
            <div
              style={{
                background: "#14161d",
                borderRadius: 10,
                padding: 20,
                border: "1px solid #1e2028",
              }}
            >
              <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 12 }}>
                Derived
              </div>
              {[
                ["head_dim", breakdown.headDim],
                ["kv_heads (effective)", breakdown.kvHeads],
                ["params_per_layer", formatNumber(breakdown.perLayer)],
                ["GQA", breakdown.kvHeads < (config.num_attention_heads || 1) ? "yes" : "no (MHA)"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 12, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace" }}>{k}</span>
                  <span style={{ fontSize: 12, color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
