// Shared primitives, copied from the Claude Design export and ES-modulized.
// MolStructurePlaceholder is replaced with a real backend-rendered <img>.

import {
  useState,
  useEffect,
  useRef,
  useMemo,
  useContext,
  createContext,
  Fragment,
} from 'react';

// ── Tokens ──────────────────────────────────────────────────────────────────
export const TOKENS = {
  paper:        "#FDFBF5",
  panel:        "#FFFFFF",
  ink:          "#0F1A24",
  ink2:         "#3A4855",
  muted:        "#7B8794",
  rule:         "#E6E1D3",
  rule2:        "#EFEAD9",
  accent:       "#1E5F8F",
  accent2:      "#0E4F75",
  pos:          "#1F7A4D",
  pos2:         "#E6F4EC",
  warn:         "#A36A1A",
  warn2:        "#FBEED1",
  err:          "#A23030",
  err2:         "#F6E2E0",
  catalyst:     "#7A4DA8",
  base:         "#1F7A4D",
  solvent:      "#236B8A",
  additive:     "#A36A1A",
};

export const FONT_SANS  = '"Inter Tight", -apple-system, "Helvetica Neue", system-ui, sans-serif';
export const FONT_SERIF = '"Source Serif 4", "Iowan Old Style", Georgia, serif';
export const FONT_MONO  = '"JetBrains Mono", ui-monospace, Menlo, monospace';

// ── Density scale ───────────────────────────────────────────────────────────
export const DENSITY = {
  compact:  { pad: 8,  gap: 6,  font: 13, tileH: 110, rowH: 36 },
  comfy:    { pad: 12, gap: 10, font: 14, tileH: 138, rowH: 44 },
  spacious: { pad: 18, gap: 14, font: 15, tileH: 168, rowH: 52 },
};
export function useDensity(d) { return DENSITY[d] || DENSITY.comfy; }

// ── Icons (16px) ────────────────────────────────────────────────────────────
export const Icon = {
  copy:    () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>,
  ext:     () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>,
  check:   () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>,
  edit:    () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>,
  flag:    () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>,
  chev:    (props) => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" {...props}><polyline points="6 9 12 15 18 9"/></svg>,
  search:  () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="7"/><line x1="20" y1="20" x2="16.65" y2="16.65"/></svg>,
  download:() => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  upload:  () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
  spark:   () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>,
  hex:     () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 22 7 22 17 12 22 2 17 2 7"/></svg>,
  warn:    () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
  arrow:   () => <svg width="100%" height="14" viewBox="0 0 100 14" preserveAspectRatio="none"><line x1="0" y1="7" x2="92" y2="7" stroke={TOKENS.ink2} strokeWidth="1.4"/><polygon points="100,7 92,3 92,11" fill={TOKENS.ink2}/></svg>,
};

// ── Molecule renderer ───────────────────────────────────────────────────────
// Backed by `GET /api/mol?smiles=...&w=...&h=...` (RDKit, transparent RGBA PNG).
// Falls back to a SMILES text label if the backend can't render the SMILES.
// The fallback is clickable to retry — useful if the backend was momentarily
// unreachable when the tile first mounted.
export function MolStructurePlaceholder({ smiles, width = 200, height = 110 }) {
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  useEffect(() => { setFailed(false); setAttempt(0); }, [smiles]);

  if (!smiles) {
    return (
      <div style={{ width, height, display:"grid", placeItems:"center",
                    color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 11 }}>
        (generic)
      </div>
    );
  }
  if (failed) {
    return (
      <button
        type="button"
        title="Couldn't render — click to retry"
        onClick={(e) => { e.stopPropagation(); setFailed(false); setAttempt(a => a + 1); }}
        style={{
          width, height, display:"grid", placeItems:"center",
          color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 10,
          padding: 4, textAlign: "center", wordBreak: "break-all", lineHeight: 1.2,
          background: "transparent", border: `1px dashed ${TOKENS.rule}`,
          borderRadius: 4, cursor: "pointer",
        }}>
        {smiles}
      </button>
    );
  }
  // Render at 2x for crisp HiDPI; CSS sizes back down. `attempt` busts the
  // browser cache after a manual retry.
  const cacheKey = attempt ? `&r=${attempt}` : "";
  const src = `/api/mol?smiles=${encodeURIComponent(smiles)}&w=${Math.max(64, Math.round(width*2))}&h=${Math.max(48, Math.round(height*2))}${cacheKey}`;
  return (
    <img
      src={src}
      width={width}
      height={height}
      alt={smiles}
      loading="lazy"
      onError={() => setFailed(true)}
      style={{ display: "block", maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
    />
  );
}

// ── MolTile — clean paper card ──────────────────────────────────────────────
export function MolTile({ mol, kind, dens, onEdit, compactWidth }) {
  const sub =
    kind === "product" && mol.yield_pct != null ? `${mol.yield_pct}%` :
    kind === "product" && mol.yield_note         ? mol.yield_note :
    kind === "reactant" && mol.equiv              ? mol.equiv : null;

  const subTone =
    kind === "product" && mol.yield_pct != null
      ? (mol.yield_pct >= 75 ? TOKENS.pos : mol.yield_pct >= 50 ? TOKENS.warn : TOKENS.err)
      : TOKENS.muted;

  return (
    <div style={{
      flex: "0 0 auto",
      width: compactWidth || 188,
      background: TOKENS.panel,
      border: `1px solid ${TOKENS.rule}`,
      borderRadius: 8,
      padding: dens.pad,
      display: "flex",
      flexDirection: "column",
      gap: 6,
    }}>
      <div style={{
        height: dens.tileH,
        background: TOKENS.paper,
        borderRadius: 6,
        display: "grid",
        placeItems: "center",
        position: "relative",
      }}>
        <MolStructurePlaceholder smiles={mol.smiles} width={compactWidth ? compactWidth - dens.pad*2 - 4 : 170} height={dens.tileH - 8}/>
        {mol.label && (
          <span style={{
            position: "absolute", top: 6, left: 6,
            fontFamily: FONT_MONO, fontSize: 10, color: TOKENS.muted,
            background: "rgba(255,255,255,0.85)", padding: "1px 6px", borderRadius: 3, border: `1px solid ${TOKENS.rule2}`,
          }}>{mol.label}</span>
        )}
      </div>
      <div style={{
        display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 6,
        fontFamily: FONT_SANS, fontSize: dens.font - 1, color: TOKENS.ink, fontWeight: 600,
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
      }}>
        <span style={{ overflow:"hidden", textOverflow:"ellipsis" }}>{mol.name || "—"}</span>
        {sub && <span style={{ color: subTone, fontVariantNumeric: "tabular-nums", flex: "0 0 auto", fontSize: dens.font - 2 }}>{sub}</span>}
      </div>
      <div style={{ display:"flex", alignItems:"center", gap: 4 }}>
        <code style={{
          fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted,
          background: TOKENS.paper, padding: "2px 6px", borderRadius: 4, border: `1px solid ${TOKENS.rule2}`,
          overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", flex: 1, minWidth: 0,
        }}>{mol.smiles || "—"}</code>
        <CopyButton text={mol.smiles}/>
      </div>
    </div>
  );
}

export function CopyButton({ text }) {
  const [done, setDone] = useState(false);
  if (!text) return null;
  return (
    <button
      onClick={(e) => { e.stopPropagation(); navigator.clipboard?.writeText(text); setDone(true); setTimeout(()=>setDone(false), 900); }}
      title="Copy SMILES"
      style={{
        flex: "0 0 auto", display:"grid", placeItems:"center",
        width: 22, height: 22, padding: 0,
        background: done ? TOKENS.pos2 : "transparent",
        border: `1px solid ${done ? TOKENS.pos : TOKENS.rule2}`,
        color: done ? TOKENS.pos : TOKENS.muted,
        borderRadius: 4, cursor: "pointer",
      }}
    >{done ? <Icon.check/> : <Icon.copy/>}</button>
  );
}

// ── Role chip ──────────────────────────────────────────────────────────────
export function RoleChip({ role, children, tone }) {
  const c = tone || TOKENS[role] || TOKENS.muted;
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap: 4,
      fontFamily: FONT_SANS, fontSize: 11, fontWeight: 600,
      letterSpacing: 0.2,
      color: c,
      background: "white",
      border: `1px solid ${c}33`,
      padding: "2px 7px", borderRadius: 999,
    }}>
      <span style={{ width: 5, height: 5, borderRadius: 999, background: c }}/>
      {children || role}
    </span>
  );
}

// ── Condition row ──────────────────────────────────────────────────────────
export function ConditionRow({ conditions, reagents, compact }) {
  const items = [];
  for (const r of (reagents || [])) {
    items.push({
      k: r.role || "reagent",
      tone: TOKENS[r.role] || TOKENS.ink2,
      label: (r.name || r.label || r.smiles || "?") + (r.loading ? ` · ${r.loading}` : ""),
    });
  }
  const c = conditions || {};
  if (c.solvent)     items.push({ k:"solvent",  tone: TOKENS.solvent, label: c.solvent });
  if (c.temperature) items.push({ k:"temp",     tone: TOKENS.ink2,    label: c.temperature });
  if (c.time)        items.push({ k:"time",     tone: TOKENS.ink2,    label: c.time });
  if (c.atmosphere)  items.push({ k:"atm",      tone: TOKENS.ink2,    label: c.atmosphere });

  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap: 6 }}>
      {items.map((it, i) => (
        <span key={i} style={{
          display:"inline-flex", alignItems:"center", gap: 6,
          fontFamily: FONT_SANS, fontSize: compact ? 11 : 12,
          color: TOKENS.ink2,
          background: "white",
          border: `1px solid ${TOKENS.rule}`,
          padding: compact ? "2px 7px" : "3px 9px", borderRadius: 4,
        }}>
          <span style={{ width: 5, height: 5, borderRadius: 999, background: it.tone, flex: "0 0 auto" }}/>
          {it.label}
        </span>
      ))}
    </div>
  );
}

// ── Yield badge ────────────────────────────────────────────────────────────
export function YieldBadge({ value, note, big }) {
  if (value == null && !note) return null;
  const tone = value == null ? TOKENS.muted
             : value >= 75 ? TOKENS.pos
             : value >= 50 ? TOKENS.warn
             : TOKENS.err;
  const bg   = value == null ? "#F3F0E6"
             : value >= 75 ? TOKENS.pos2
             : value >= 50 ? TOKENS.warn2
             : TOKENS.err2;
  return (
    <span style={{
      display:"inline-flex", alignItems:"baseline", gap: 3,
      fontFamily: FONT_SANS, fontWeight: 700, color: tone,
      background: bg, padding: big ? "4px 10px" : "2px 8px",
      borderRadius: 999, fontVariantNumeric: "tabular-nums",
      fontSize: big ? 14 : 12, border: `1px solid ${tone}33`,
    }}>
      {value != null ? <span style={{ display:"inline-flex", alignItems:"baseline", gap: 3 }}><span>{value}</span><span style={{ fontSize: big ? 11 : 9, fontWeight: 600 }}>%</span></span> : <span style={{ fontSize: big ? 12 : 11, fontWeight: 600 }}>{note}</span>}
    </span>
  );
}

// ── Structured Rxn-INSIGHT panel ───────────────────────────────────────────
// Resilient to fields that the backend Rxn-INSIGHT row may not carry
// (suggested_solvents, hazards) — only renders rows for fields with content.
export function StructuredInsightPanel({ analyses, dens }) {
  if (!analyses || Object.keys(analyses).length === 0) return null;
  const entries = Object.entries(analyses);
  return (
    <div style={{
      background: "white",
      border: `1px solid ${TOKENS.rule}`,
      borderRadius: 8,
      overflow: "hidden",
    }}>
      <div style={{
        display:"flex", alignItems:"center", gap: 8,
        padding: `${dens.pad}px ${dens.pad+4}px`,
        borderBottom: `1px solid ${TOKENS.rule}`,
        background: TOKENS.paper,
      }}>
        <Icon.spark/>
        <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: dens.font, color: TOKENS.ink }}>
          Rxn-INSIGHT enrichment
        </span>
        <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted, marginLeft: "auto" }}>
          {entries.length} {entries.length === 1 ? "reaction" : "reactions"} analysed
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: entries.length > 1 ? "repeat(auto-fill, minmax(280px, 1fr))" : "1fr", gap: 1, background: TOKENS.rule }}>
        {entries.map(([id, a]) => (
          <div key={id} style={{ background: "white", padding: dens.pad+4 }}>
            <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom: 8 }}>
              <span style={{
                fontFamily: FONT_MONO, fontSize: 10, color: TOKENS.accent,
                background: "#E8F1F8", padding: "1px 6px", borderRadius: 3,
                border: `1px solid ${TOKENS.accent}33`,
              }}>{id}</span>
              <span style={{ fontFamily: FONT_SANS, fontSize: dens.font - 1, fontWeight: 600, color: TOKENS.ink }}>
                {a.reaction_class || a.error || "—"}
              </span>
            </div>
            <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "auto 1fr", rowGap: 4, columnGap: 10, fontFamily: FONT_SANS, fontSize: dens.font - 2 }}>
              {a.named_reaction && (<><dt style={{ color: TOKENS.muted }}>Named</dt><dd style={{ margin: 0, color: TOKENS.ink, fontStyle: "italic" }}>{a.named_reaction}</dd></>)}
              {a.rxn_class_id && (<><dt style={{ color: TOKENS.muted }}>Tag</dt><dd style={{ margin: 0, fontFamily: FONT_MONO, color: TOKENS.ink2 }}>{a.rxn_class_id}</dd></>)}
              {a.functional_groups?.length > 0 && (
                <>
                  <dt style={{ color: TOKENS.muted }}>FG</dt>
                  <dd style={{ margin: 0 }}>
                    {a.functional_groups.map((fg, i) => (
                      <span key={i} style={{ display:"inline-block", marginRight: 4, padding:"1px 6px", border:`1px solid ${TOKENS.rule}`, borderRadius: 3, fontSize: 11, color: TOKENS.ink2 }}>{fg}</span>
                    ))}
                  </dd>
                </>
              )}
              {a.byproducts?.length > 0 && (
                <>
                  <dt style={{ color: TOKENS.muted }}>Byprod</dt>
                  <dd style={{ margin: 0, fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2 }}>{a.byproducts.join(", ")}</dd>
                </>
              )}
              {a.template && (<><dt style={{ color: TOKENS.muted }}>Template</dt><dd style={{ margin: 0, fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2, wordBreak:"break-all" }}>{a.template}</dd></>)}
              {a.scaffold && (<><dt style={{ color: TOKENS.muted }}>Scaffold</dt><dd style={{ margin: 0, fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2, wordBreak:"break-all" }}>{a.scaffold}</dd></>)}
              {a.suggested_solvents?.length > 0 && (
                <>
                  <dt style={{ color: TOKENS.muted }}>Suggest</dt>
                  <dd style={{ margin: 0, fontSize: 11, color: TOKENS.ink2 }}>{a.suggested_solvents.join(" / ")}</dd>
                </>
              )}
              {a.hazards?.length > 0 && a.hazards[0] !== "—" && (
                <>
                  <dt style={{ color: TOKENS.muted }}>Hazards</dt>
                  <dd style={{ margin: 0, fontSize: 11, color: TOKENS.warn, display:"inline-flex", alignItems:"center", gap:4 }}>
                    <Icon.warn/>{a.hazards.join(", ")}
                  </dd>
                </>
              )}
            </dl>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Activity drawer ────────────────────────────────────────────────────────
export function ActivityDrawer({ log, open, onToggle, visible, position = "left", meta }) {
  if (!visible) return null;
  const stepCount = log.length;
  const wallTime = meta?.wall_time_s != null ? `${meta.wall_time_s.toFixed(1)} s` : null;
  return (
    <div style={{
      borderTop: `1px solid ${TOKENS.rule}`,
      background: TOKENS.paper,
    }}>
      <button
        onClick={onToggle}
        style={{
          display: "flex", alignItems: "center", gap: 8,
          width: "100%", padding: "8px 14px", background: "transparent", border: "none",
          color: TOKENS.muted, cursor: "pointer", fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
          textAlign: "left", letterSpacing: 0.3, textTransform: "uppercase",
        }}
      >
        <Icon.chev style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform .15s" }}/>
        Agent activity
        <span style={{
          color: TOKENS.ink2, textTransform: "none", letterSpacing: 0,
          fontWeight: 500,
        }}>· {stepCount} {stepCount === 1 ? "step" : "steps"}{wallTime ? ` · ${wallTime}` : ""}</span>
      </button>
      {open && (
        <ol style={{
          margin: 0, padding: "0 14px 12px 14px", listStyle: "none",
          fontFamily: FONT_MONO, fontSize: 12, color: TOKENS.ink2,
          display: "grid", gap: 4,
          maxHeight: 200, overflowY: "auto",
        }}>
          {log.map((s) => (
            <li key={s.n} style={{ display: "grid", gridTemplateColumns: "26px 150px 1fr 14px", gap: 8, alignItems: "center", padding: "2px 0" }}>
              <span style={{ color: TOKENS.muted }}>#{s.n}</span>
              <span style={{ color: TOKENS.accent, fontWeight: 600 }}>{s.tool}</span>
              <span style={{ color: TOKENS.ink2, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{s.summary || s.args_preview || ""}</span>
              <span style={{ color: s.ok === false ? TOKENS.err : (s.ok === true ? TOKENS.pos : TOKENS.muted) }}>{s.ok === false ? "✗" : (s.ok === true ? "✓" : "…")}</span>
            </li>
          ))}
          {log.length === 0 && (
            <li style={{ color: TOKENS.muted, padding: "8px 4px" }}>No activity yet.</li>
          )}
        </ol>
      )}
    </div>
  );
}

// ── Stat cell ──────────────────────────────────────────────────────────────
export function Stat({ label, value, mono, accent }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap: 2, minWidth: 0 }}>
      <span style={{ fontFamily: FONT_SANS, fontSize: 10, textTransform:"uppercase", letterSpacing: 0.6, color: TOKENS.muted, fontWeight: 600 }}>{label}</span>
      <span style={{
        fontFamily: mono ? FONT_MONO : FONT_SANS,
        fontSize: 16, fontWeight: 600,
        color: accent || TOKENS.ink,
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap", overflow:"hidden", textOverflow:"ellipsis",
      }}>{value}</span>
    </div>
  );
}

// ── Model picker (prominent card form) ─────────────────────────────────────
export function ModelPickerCard({ models, value, onChange, compact }) {
  const cur = models.find(m => m.id === value) || models[0];
  const [open, setOpen] = useState(false);
  if (!cur) {
    return (
      <div style={{
        minWidth: 220, padding: "8px 12px",
        background: "white", border: `1px solid ${TOKENS.rule}`,
        borderRadius: 6, color: TOKENS.muted,
        fontFamily: FONT_SANS, fontSize: 12,
      }}>No model configured</div>
    );
  }
  return (
    <div style={{ position:"relative", minWidth: 220 }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: "100%", display:"flex", alignItems:"center", gap: 8,
        padding: compact ? "6px 10px" : "8px 12px",
        background: "white", border: `1px solid ${TOKENS.rule}`,
        borderRadius: 6, cursor:"pointer", textAlign:"left",
      }}>
        <ProviderDot provider={cur.provider}/>
        <div style={{ display:"flex", flexDirection:"column", gap: 1, flex: 1, minWidth: 0 }}>
          <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13, color: TOKENS.ink, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{cur.label}</span>
          <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>{cur.provider}{cur.context ? ` · ${cur.context}` : ""}</span>
        </div>
        <Icon.chev style={{ transform: open ? "rotate(180deg)" : "none", color: TOKENS.muted }}/>
      </button>
      {open && (
        <div style={{
          position:"absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 5,
          background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
          boxShadow: "0 4px 16px rgba(15,26,36,0.08)",
          overflow: "hidden",
        }}>
          {models.map(m => (
            <div key={m.id} onClick={() => { onChange(m.id); setOpen(false); }} style={{
              display:"flex", alignItems:"center", gap: 8, padding: "7px 10px",
              cursor: "pointer", borderBottom: `1px solid ${TOKENS.rule2}`,
              background: m.id === value ? TOKENS.paper : "white",
            }}>
              <ProviderDot provider={m.provider}/>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display:"flex", alignItems:"baseline", gap: 6 }}>
                  <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13, color: TOKENS.ink }}>{m.label}</span>
                  {m.recommended && <span style={{ fontFamily: FONT_SANS, fontSize: 10, color: TOKENS.pos, background: TOKENS.pos2, padding: "0 5px", borderRadius: 3, fontWeight: 600 }}>RECOMMENDED</span>}
                </div>
                <div style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>{m.provider}{m.context ? ` · ctx ${m.context}` : ""}</div>
              </div>
              {m.id === value && <Icon.check/>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ProviderDot({ provider }) {
  const C = { anthropic:"#D97757", openai:"#10A37F", google:"#4285F4", openrouter:"#7A4DA8", azure:"#0078D4", gemini:"#4285F4" };
  const c = C[provider] || TOKENS.muted;
  return <span style={{ display:"inline-block", width: 9, height: 9, borderRadius: 999, background: c, flex: "0 0 auto" }}/>;
}

// ── Primary button ─────────────────────────────────────────────────────────
export function Button({ children, primary, onClick, icon, small, disabled, as, href }) {
  const style = {
    display:"inline-flex", alignItems:"center", gap: 6,
    padding: small ? "5px 10px" : "7px 13px",
    fontFamily: FONT_SANS, fontWeight: 600, fontSize: small ? 12 : 13,
    borderRadius: 6, cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1,
    background: primary ? TOKENS.accent : "white",
    color: primary ? "white" : TOKENS.ink,
    border: `1px solid ${primary ? TOKENS.accent : TOKENS.rule}`,
    textDecoration: "none",
  };
  if (as === 'a') {
    return <a href={href} download style={style} onClick={onClick}>{icon}{children}</a>;
  }
  return (
    <button onClick={onClick} disabled={disabled} style={style}>
      {icon}
      {children}
    </button>
  );
}

// ── Ghent University attribution ──────────────────────────────────────────
// Loads `web/public/ghent_university.png` (served at /static/ghent_university.png
// by Vite). Falls back to a text-only "Ghent University" mark in the official
// UGent blue if the file isn't there.
export function GhentBadge({ height = 40, muted = false }) {
  const [pngOk, setPngOk] = useState(true);
  return (
    <a
      href="https://www.ugent.be"
      target="_blank"
      rel="noreferrer"
      title="Ghent University"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        textDecoration: "none",
        opacity: muted ? 0.85 : 1,
      }}
    >
      {pngOk ? (
        <img
          src="/static/logo_UGent_EN_RGB_2400_color.png"
          alt="Ghent University"
          onError={() => setPngOk(false)}
          style={{ height, width: "auto", display: "block" }}
        />
      ) : (
        <span style={{
          display: "inline-flex", flexDirection: "column",
          alignItems: "flex-start", lineHeight: 1.0,
          color: "#1E64C8",
          fontFamily: FONT_SANS, fontWeight: 700, letterSpacing: 0.6,
        }}>
          <span style={{ fontSize: Math.round(height * 0.34) }}>GHENT</span>
          <span style={{ fontSize: Math.round(height * 0.34) }}>UNIVERSITY</span>
        </span>
      )}
    </a>
  );
}

// ── Section header ─────────────────────────────────────────────────────────
export function SectionLabel({ children, right }) {
  return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", margin: "0 0 8px" }}>
      <h3 style={{
        margin: 0, fontFamily: FONT_SANS, fontSize: 11, fontWeight: 700,
        textTransform: "uppercase", letterSpacing: 0.8, color: TOKENS.muted,
      }}>{children}</h3>
      {right}
    </div>
  );
}
