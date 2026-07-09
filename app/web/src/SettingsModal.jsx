// Settings modal — edit provider credentials, endpoints, and per-provider
// model lists. Writes flow through /api/settings/* into both os.environ
// (immediate effect on /api/health and /api/runs) and `5 App/.env` (so
// changes survive a restart).

import { useEffect, useState } from 'react';
import {
  TOKENS, FONT_SANS, FONT_MONO, Icon, Button, ProviderDot, SectionLabel,
} from './shared.jsx';
import {
  fetchSettings, saveDefaultProvider, saveProviderSettings,
} from './api.js';

export function SettingsModal({ onClose, onSaved }) {
  const [data, setData]   = useState(null);
  const [err, setErr]     = useState(null);
  const [savingDefault, setSavingDefault] = useState(false);

  async function refresh() {
    try {
      setData(await fetchSettings());
    } catch (e) {
      setErr(String(e.message || e));
    }
  }
  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function changeDefault(provider) {
    setSavingDefault(true);
    try {
      const updated = await saveDefaultProvider(provider);
      setData(updated);
      onSaved && onSaved();
    } catch (e) {
      alert(`Couldn't switch default backend: ${e.message || e}`);
    } finally {
      setSavingDefault(false);
    }
  }

  async function saveProvider(provider, payload) {
    const updated = await saveProviderSettings(provider, payload);
    setData(updated);
    onSaved && onSaved();
  }

  const providerOrder = ['anthropic', 'azure', 'gemini', 'openrouter'];

  return (
    <div onClick={onClose} style={{
      position: "absolute", inset: 0, zIndex: 130,
      background: "rgba(15,26,36,0.40)",
      display: "grid", placeItems: "center", padding: 24,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: "min(840px, 100%)", height: "min(720px, 88vh)",
        background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 10,
        boxShadow: "0 16px 48px rgba(15,26,36,0.18)",
        display: "grid", gridTemplateRows: "auto 1fr auto", overflow: "hidden",
      }}>
        <header style={{
          display:"flex", alignItems:"center", gap: 10,
          padding: "14px 18px", borderBottom: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper,
        }}>
          <h3 style={{ margin: 0, fontFamily: FONT_SANS, fontSize: 15, fontWeight: 700, color: TOKENS.ink }}>
            Settings
          </h3>
          <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>
            providers · models · credentials
          </span>
          <div style={{ flex: 1 }}/>
          <button onClick={onClose} style={{
            width: 26, height: 26, padding: 0, background: "transparent",
            border: `1px solid ${TOKENS.rule}`, borderRadius: 4, cursor: "pointer",
            fontFamily: FONT_SANS, fontWeight: 700, color: TOKENS.muted, fontSize: 14,
          }}>×</button>
        </header>

        <div style={{ overflowY: "auto", padding: "16px 18px 20px" }}>
          {err && <div style={{ color: TOKENS.err, fontFamily: FONT_SANS, padding: 10 }}>Couldn't load settings: {err}</div>}
          {!err && data == null && (
            <div style={{ padding: 24, textAlign:"center", color: TOKENS.muted, fontFamily: FONT_SANS, fontSize: 13 }}>Loading…</div>
          )}
          {!err && data && (
            <>
              <DefaultBackendRow
                value={data.default_provider}
                providers={data.providers}
                onChange={changeDefault}
                saving={savingDefault}
              />
              <div style={{ height: 12 }}/>
              {providerOrder.map(prov => (
                <ProviderSection
                  key={prov}
                  provider={prov}
                  config={data.providers[prov]}
                  onSave={(payload) => saveProvider(prov, payload)}
                />
              ))}
            </>
          )}
        </div>

        <footer style={{
          padding: "10px 18px", borderTop: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper,
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted,
          textAlign: "right",
        }}>changes persist to .env · esc to close</footer>
      </div>
    </div>
  );
}

function DefaultBackendRow({ value, providers, onChange, saving }) {
  const opts = Object.keys(providers || {});
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12,
      padding: 12,
      background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
    }}>
      <div style={{ display:"flex", flexDirection:"column", gap: 2 }}>
        <span style={{ fontFamily: FONT_SANS, fontSize: 11, fontWeight: 700, color: TOKENS.muted, textTransform:"uppercase", letterSpacing: 0.5 }}>
          Default backend
        </span>
        <span style={{ fontFamily: FONT_SANS, fontSize: 12.5, color: TOKENS.ink2 }}>
          Used when a request doesn't specify a provider.
        </span>
      </div>
      <div style={{ flex: 1 }}/>
      <select value={value} disabled={saving}
              onChange={(e) => onChange(e.target.value)}
              style={{
        padding: "6px 10px",
        fontFamily: FONT_SANS, fontSize: 13, color: TOKENS.ink,
        border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
        background: "white", outline: "none",
      }}>
        {opts.map(p => (
          <option key={p} value={p}>{providers[p].label}</option>
        ))}
      </select>
    </div>
  );
}

function ProviderSection({ provider, config, onSave }) {
  const [apiKey, setApiKey]   = useState("");           // empty = no change
  const [showKey, setShowKey] = useState(false);
  const [endpoint, setEndpoint] = useState(config.endpoint || "");
  const [baseUrl, setBaseUrl]   = useState(config.base_url || "");
  const [defaultModel, setDefaultModel] = useState(config.default_model || "");
  const [models, setModels]     = useState(config.models || []);
  const [busy, setBusy]         = useState(false);
  const [msg, setMsg]           = useState(null);
  const [expanded, setExpanded] = useState(config.api_key_set);  // auto-collapse providers with no key

  const cfgKey = JSON.stringify([config.endpoint, config.base_url, config.default_model, config.models, config.api_key_preview]);
  // Re-sync local form state if the upstream config changed (e.g. after Save returned a refreshed payload).
  useEffect(() => {
    setEndpoint(config.endpoint || "");
    setBaseUrl(config.base_url || "");
    setDefaultModel(config.default_model || "");
    setModels(config.models || []);
    setApiKey("");
    setShowKey(false);
  }, [cfgKey]);

  function addModel() {
    setModels([...models, { id: "", label: "" }]);
  }
  function updateModel(i, patch) {
    setModels(models.map((m, j) => j === i ? { ...m, ...patch } : m));
  }
  function removeModel(i) {
    setModels(models.filter((_, j) => j !== i));
  }

  async function doSave(payload) {
    setBusy(true); setMsg(null);
    try {
      await onSave(payload);
      setMsg({ kind: "ok", text: "Saved." });
      setTimeout(() => setMsg(null), 2200);
    } catch (e) {
      setMsg({ kind: "err", text: String(e.message || e) });
    } finally {
      setBusy(false);
    }
  }
  function save() {
    const payload = {
      ...(apiKey.trim() ? { api_key: apiKey } : {}),
      default_model: defaultModel,
      models: models.filter(m => (m.id || "").trim()),
    };
    if (config.endpoint_field === "endpoint")  payload.endpoint = endpoint;
    if (config.endpoint_field === "base_url")  payload.base_url = baseUrl;
    doSave(payload);
  }
  function clearKey() {
    if (!confirm(`Clear the API key for ${config.label}?`)) return;
    doSave({ clear_api_key: true });
  }

  return (
    <details
      open={expanded}
      onToggle={(e) => setExpanded(e.currentTarget.open)}
      style={{
        marginTop: 10,
        background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
        overflow: "hidden",
    }}>
      <summary style={{
        listStyle: "none", cursor: "pointer",
        display: "flex", alignItems: "center", gap: 10,
        padding: "10px 14px", borderBottom: expanded ? `1px solid ${TOKENS.rule}` : "none",
        background: TOKENS.paper,
      }}>
        <Icon.chev style={{ transform: expanded ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform .15s" }}/>
        <ProviderDot provider={provider}/>
        <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink }}>{config.label}</span>
        <span style={{
          fontFamily: FONT_SANS, fontSize: 10.5, fontWeight: 700,
          color: config.api_key_set ? TOKENS.pos : TOKENS.muted,
          background: config.api_key_set ? TOKENS.pos2 : TOKENS.paper,
          border: `1px solid ${config.api_key_set ? TOKENS.pos + "55" : TOKENS.rule}`,
          padding: "1px 7px", borderRadius: 3,
          textTransform: "uppercase", letterSpacing: 0.4,
        }}>
          {config.api_key_set ? "configured" : "not set"}
        </span>
        <span style={{ flex: 1 }}/>
        <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>
          {(config.models || []).length} {(config.models || []).length === 1 ? "model" : "models"}
        </span>
      </summary>

      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
        {/* API key */}
        <Field label="API key">
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={config.api_key_set ? config.api_key_preview : "Paste your key…"}
              style={inputStyle}
              autoComplete="off"
              spellCheck={false}
            />
            <button onClick={() => setShowKey(s => !s)} title={showKey ? "Hide" : "Show"} style={tinyBtn}>
              {showKey ? "🙈" : "👁"}
            </button>
            {config.api_key_set && (
              <button onClick={clearKey} title="Clear stored key"
                      style={{ ...tinyBtn, color: TOKENS.err, borderColor: TOKENS.err + "55" }}>×</button>
            )}
          </div>
        </Field>

        {/* Endpoint (Azure) */}
        {config.endpoint_field === "endpoint" && (
          <Field label="Endpoint">
            <input value={endpoint}
                   onChange={(e) => setEndpoint(e.target.value)}
                   placeholder="https://your-resource.services.ai.azure.com/anthropic/"
                   style={inputStyle} spellCheck={false} autoComplete="off"/>
          </Field>
        )}
        {config.endpoint_field === "base_url" && (
          <Field label="Base URL"
                 hint="Defaults to https://openrouter.ai/api/v1 if blank.">
            <input value={baseUrl}
                   onChange={(e) => setBaseUrl(e.target.value)}
                   placeholder="https://openrouter.ai/api/v1"
                   style={inputStyle} spellCheck={false} autoComplete="off"/>
          </Field>
        )}

        {/* Default model */}
        <Field label="Default model"
               hint={`Used when no model is selected in the picker. Fallback: ${config.default_model_fallback}.`}>
          <input value={defaultModel}
                 onChange={(e) => setDefaultModel(e.target.value)}
                 placeholder={config.default_model_fallback}
                 style={{ ...inputStyle, fontFamily: FONT_MONO }} spellCheck={false} autoComplete="off"/>
        </Field>

        {/* Models list */}
        <Field label="Models in picker"
               hint="Add every model you want to appear in the dropdown. Label is optional — defaults to the id.">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {(models || []).map((m, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 28px", gap: 6 }}>
                <input
                  value={m.id}
                  onChange={(e) => updateModel(i, { id: e.target.value })}
                  placeholder={modelIdPlaceholder(provider)}
                  style={{ ...inputStyle, fontFamily: FONT_MONO }}
                  spellCheck={false} autoComplete="off"
                />
                <input
                  value={m.label}
                  onChange={(e) => updateModel(i, { label: e.target.value })}
                  placeholder="Display label (optional)"
                  style={inputStyle}
                  spellCheck={false} autoComplete="off"
                />
                <button onClick={() => removeModel(i)} title="Remove" style={tinyBtn}>×</button>
              </div>
            ))}
            <button onClick={addModel} style={{
              alignSelf: "flex-start",
              background: "transparent", border: `1px dashed ${TOKENS.rule}`,
              padding: "5px 10px", borderRadius: 4,
              fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
              color: TOKENS.muted, cursor: "pointer",
            }}>+ Add model</button>
          </div>
        </Field>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Button primary disabled={busy} onClick={save}>
            {busy ? "Saving…" : "Save"}
          </Button>
          {msg && (
            <span style={{
              fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
              color: msg.kind === "ok" ? TOKENS.pos : TOKENS.err,
            }}>{msg.text}</span>
          )}
        </div>
      </div>
    </details>
  );
}

function modelIdPlaceholder(provider) {
  return {
    anthropic:  "claude-opus-4-7",
    azure:      "claude-opus-4-7 (deployment name)",
    gemini:     "gemini-2.5-pro",
    openrouter: "anthropic/claude-opus-4-7",
  }[provider] || "model-id";
}

const inputStyle = {
  width: "100%", padding: "7px 10px",
  fontFamily: FONT_SANS, fontSize: 13, color: TOKENS.ink,
  border: `1px solid ${TOKENS.rule}`, borderRadius: 5,
  background: "white", outline: "none",
};

const tinyBtn = {
  width: 28, height: 28, padding: 0, flex: "0 0 auto",
  background: "white", border: `1px solid ${TOKENS.rule}`,
  color: TOKENS.muted, borderRadius: 4, cursor: "pointer",
  fontSize: 13,
};

function Field({ label, hint, children }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{
        fontFamily: FONT_SANS, fontSize: 11, fontWeight: 700,
        color: TOKENS.muted, textTransform: "uppercase", letterSpacing: 0.5,
      }}>{label}</span>
      {children}
      {hint && (
        <span style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.muted }}>{hint}</span>
      )}
    </div>
  );
}
