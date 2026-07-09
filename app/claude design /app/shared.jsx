// Shared primitives for both ReactionMiner redesign directions.
// Light, paper-feel theme; chemistry-native typography.

const { useState, useEffect, useRef, useMemo, useContext, createContext, Fragment } = React;

// ── Tokens ──────────────────────────────────────────────────────────────────
const TOKENS = {
  paper:        "#FDFBF5",  // off-white, lab-notebook
  panel:        "#FFFFFF",  // pure white card
  ink:          "#0F1A24",  // near-black
  ink2:         "#3A4855",
  muted:        "#7B8794",
  rule:         "#E6E1D3",  // warm rule line
  rule2:        "#EFEAD9",
  accent:       "#1E5F8F",  // ink-blue
  accent2:      "#0E4F75",
  pos:          "#1F7A4D",  // verified, good yield
  pos2:         "#E6F4EC",
  warn:         "#A36A1A",
  warn2:        "#FBEED1",
  err:          "#A23030",
  err2:         "#F6E2E0",
  // chem accent palette for role chips
  catalyst:     "#7A4DA8",
  base:         "#1F7A4D",
  solvent:      "#236B8A",
  additive:     "#A36A1A",
};

const FONT_SANS  = '"Inter Tight", -apple-system, "Helvetica Neue", system-ui, sans-serif';
const FONT_SERIF = '"Source Serif 4", "Iowan Old Style", Georgia, serif';
const FONT_MONO  = '"JetBrains Mono", ui-monospace, Menlo, monospace';

// ── Density scale ───────────────────────────────────────────────────────────
const DENSITY = {
  compact:  { pad: 8,  gap: 6,  font: 13, tileH: 110, rowH: 36 },
  comfy:    { pad: 12, gap: 10, font: 14, tileH: 138, rowH: 44 },
  spacious: { pad: 18, gap: 14, font: 15, tileH: 168, rowH: 52 },
};
function useDensity(d) { return DENSITY[d] || DENSITY.comfy; }

// ── Icons (16px) ────────────────────────────────────────────────────────────
const Icon = {
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

// ── Molecule structure placeholder ──────────────────────────────────────────
// Deterministic SVG sketch based on simple SMILES features.
// (In production the backend renders this server-side with RDKit.)
function MolStructurePlaceholder({ smiles, width = 200, height = 110 }) {
  if (!smiles) {
    return (
      <div style={{ width, height, display:"grid", placeItems:"center", color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 11 }}>
        (generic)
      </div>
    );
  }

  const aromatic = /c1cc|c1nc|c1oc|c1sc|c1ccccc1|c[1-9].*?c[1-9]/i.test(smiles);
  const hetero   = /N1|O1|S1|n1|o1/.test(smiles);
  const boron    = /B\(/.test(smiles);
  const branched = (smiles.match(/\(/g) || []).length >= 3 && !aromatic;
  const halogen  = /Cl|Br|F\b|I\b/.test(smiles);
  const carbonyl = /=O|C=O|C\(=O\)/.test(smiles);
  const chiral   = /@/.test(smiles);

  const cx = width/2, cy = height/2;
  const ink = TOKENS.ink2;

  // Pick template
  let template;
  if (boron && !aromatic)     template = "boron";
  else if (aromatic)          template = halogen ? "aryl_x" : (carbonyl ? "aryl_co" : "aryl");
  else if (hetero)            template = "ring_n";
  else if (branched)          template = "branched";
  else                        template = "chain";

  // Templates render around (cx, cy) at scale ~0.7 * min(w,h)
  const R = Math.min(width, height) * 0.28;

  function hexPoints(cx, cy, r) {
    return [0,1,2,3,4,5].map(i => {
      const a = -Math.PI/2 + i * Math.PI/3;
      return [cx + r*Math.cos(a), cy + r*Math.sin(a)];
    });
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%" style={{ display:"block" }}>
      {template === "aryl" && (() => {
        const p = hexPoints(cx, cy, R);
        return (
          <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
            <polygon points={p.map(([x,y])=>`${x},${y}`).join(" ")} />
            {/* inner aromatic circle */}
            <circle cx={cx} cy={cy} r={R*0.55} stroke={ink} strokeWidth="1" fill="none" opacity="0.7"/>
          </g>
        );
      })()}

      {template === "aryl_x" && (() => {
        const p = hexPoints(cx-R*0.6, cy, R);
        return (
          <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
            <polygon points={p.map(([x,y])=>`${x},${y}`).join(" ")}/>
            <circle cx={cx-R*0.6} cy={cy} r={R*0.55} stroke={ink} strokeWidth="1" opacity="0.7"/>
            <line x1={p[1][0]} y1={p[1][1]} x2={p[1][0]+R*0.7} y2={p[1][1]-R*0.2}/>
            <text x={p[1][0]+R*0.85} y={p[1][1]-R*0.18} fontFamily={FONT_SANS} fontSize="11" fill={ink} dominantBaseline="middle">Cl</text>
            <line x1={p[5][0]} y1={p[5][1]} x2={p[5][0]+R*0.7} y2={p[5][1]+R*0.2}/>
            <text x={p[5][0]+R*0.85} y={p[5][1]+R*0.22} fontFamily={FONT_SANS} fontSize="11" fill={ink} dominantBaseline="middle">Cl</text>
          </g>
        );
      })()}

      {template === "aryl_co" && (() => {
        const p = hexPoints(cx-R*0.6, cy, R);
        return (
          <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
            <polygon points={p.map(([x,y])=>`${x},${y}`).join(" ")}/>
            <circle cx={cx-R*0.6} cy={cy} r={R*0.55} stroke={ink} strokeWidth="1" opacity="0.7"/>
            <line x1={p[0][0]} y1={p[0][1]} x2={p[0][0]+R*0.4} y2={p[0][1]-R*0.7}/>
            <text x={p[0][0]+R*0.55} y={p[0][1]-R*0.78} fontFamily={FONT_SANS} fontSize="11" fill={ink} dominantBaseline="middle">O</text>
          </g>
        );
      })()}

      {template === "ring_n" && (() => {
        const p = hexPoints(cx, cy, R);
        return (
          <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
            <polygon points={p.map(([x,y])=>`${x},${y}`).join(" ")}/>
            <circle cx={p[3][0]} cy={p[3][1]} r="9" fill={TOKENS.panel}/>
            <text x={p[3][0]} y={p[3][1]+1} fontFamily={FONT_SANS} fontSize="12" fontWeight="600" fill={ink} textAnchor="middle" dominantBaseline="middle">N</text>
            {carbonyl && (
              <g>
                <line x1={p[0][0]} y1={p[0][1]} x2={p[0][0]+R*0.35} y2={p[0][1]-R*0.6}/>
                <text x={p[0][0]+R*0.5} y={p[0][1]-R*0.7} fontFamily={FONT_SANS} fontSize="11" fill={ink}>O</text>
              </g>
            )}
          </g>
        );
      })()}

      {template === "boron" && (() => {
        return (
          <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
            <circle cx={cx} cy={cy} r="11" fill={TOKENS.panel}/>
            <text x={cx} y={cy+1} fontFamily={FONT_SANS} fontSize="13" fontWeight="700" fill={ink} textAnchor="middle" dominantBaseline="middle">B</text>
            <line x1={cx} y1={cy-12} x2={cx} y2={cy-R}/>
            <text x={cx} y={cy-R-8} fontFamily={FONT_SANS} fontSize="11" fill={ink} textAnchor="middle">OH</text>
            <line x1={cx-10} y1={cy+8} x2={cx-R*0.95} y2={cy+R*0.6}/>
            <text x={cx-R*0.95-12} y={cy+R*0.65} fontFamily={FONT_SANS} fontSize="11" fill={ink} textAnchor="end">OH</text>
            <line x1={cx+10} y1={cy+8} x2={cx+R*0.95} y2={cy+R*0.6}/>
            <text x={cx+R*0.95+4} y={cy+R*0.65} fontFamily={FONT_SANS} fontSize="11" fill={ink}>OH</text>
          </g>
        );
      })()}

      {template === "branched" && (() => (
        <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
          <line x1={cx} y1={cy} x2={cx-R} y2={cy-R*0.6}/>
          <line x1={cx} y1={cy} x2={cx+R} y2={cy-R*0.6}/>
          <line x1={cx} y1={cy} x2={cx} y2={cy+R}/>
          <line x1={cx-R} y1={cy-R*0.6} x2={cx-R*1.6} y2={cy-R*0.2}/>
          <line x1={cx+R} y1={cy-R*0.6} x2={cx+R*1.6} y2={cy-R*0.2}/>
        </g>
      ))()}

      {template === "chain" && (() => {
        const seg = R*0.55;
        const start = cx - 2.2*seg;
        const pts = [];
        for (let i=0; i<6; i++) {
          pts.push([start + i*seg, cy + (i % 2 ? -seg*0.6 : seg*0.6)]);
        }
        return (
          <g fill="none" stroke={ink} strokeWidth="1.4" strokeLinecap="round">
            {pts.slice(0,-1).map(([x1,y1], i) => (
              <line key={i} x1={x1} y1={y1} x2={pts[i+1][0]} y2={pts[i+1][1]}/>
            ))}
            {carbonyl && (
              <g>
                <line x1={pts[5][0]} y1={pts[5][1]} x2={pts[5][0]+seg*0.6} y2={pts[5][1]-seg*1.0}/>
                <text x={pts[5][0]+seg*0.7} y={pts[5][1]-seg*1.05} fontFamily={FONT_SANS} fontSize="11" fill={ink}>O</text>
              </g>
            )}
          </g>
        );
      })()}

      {chiral && (
        <text x={width-6} y={height-6} fontFamily={FONT_MONO} fontSize="9" fill={TOKENS.muted} textAnchor="end">{smiles.match(/@@/) ? "(R)" : "(S)"}</text>
      )}
    </svg>
  );
}

// ── MolTile — clean paper card ──────────────────────────────────────────────
function MolTile({ mol, kind, dens, onEdit, compactWidth }) {
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

function CopyButton({ text }) {
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
function RoleChip({ role, children, tone }) {
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
function ConditionRow({ conditions, reagents, compact }) {
  const items = [];
  for (const r of (reagents || [])) {
    items.push({
      k: r.role || "reagent",
      tone: TOKENS[r.role] || TOKENS.ink2,
      label: r.name + (r.loading ? ` · ${r.loading}` : ""),
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
function YieldBadge({ value, note, big }) {
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
function StructuredInsightPanel({ analyses, dens }) {
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
                {a.reaction_class}
              </span>
            </div>
            <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "auto 1fr", rowGap: 4, columnGap: 10, fontFamily: FONT_SANS, fontSize: dens.font - 2 }}>
              {a.named_reaction && (<><dt style={{ color: TOKENS.muted }}>Named</dt><dd style={{ margin: 0, color: TOKENS.ink, fontStyle: "italic" }}>{a.named_reaction}</dd></>)}
              <dt style={{ color: TOKENS.muted }}>Class ID</dt><dd style={{ margin: 0, fontFamily: FONT_MONO, color: TOKENS.ink2 }}>{a.rxn_class_id}</dd>
              <dt style={{ color: TOKENS.muted }}>Functional</dt>
              <dd style={{ margin: 0 }}>
                {a.functional_groups.map((fg, i) => (
                  <span key={i} style={{ display:"inline-block", marginRight: 4, padding:"1px 6px", border:`1px solid ${TOKENS.rule}`, borderRadius: 3, fontSize: 11, color: TOKENS.ink2 }}>{fg}</span>
                ))}
              </dd>
              <dt style={{ color: TOKENS.muted }}>Byproducts</dt>
              <dd style={{ margin: 0, fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2 }}>{a.byproducts.join(", ")}</dd>
              <dt style={{ color: TOKENS.muted }}>Suggest</dt>
              <dd style={{ margin: 0, fontSize: 11, color: TOKENS.ink2 }}>{a.suggested_solvents.join(" / ")}</dd>
              {a.hazards && a.hazards[0] !== "—" && (
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
function ActivityDrawer({ log, open, onToggle, visible, position = "left" }) {
  if (!visible) return null;
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
        }}>· {log.length} steps · 38.4 s</span>
      </button>
      {open && (
        <ol style={{
          margin: 0, padding: "0 14px 12px 14px", listStyle: "none",
          fontFamily: FONT_MONO, fontSize: 12, color: TOKENS.ink2,
          display: "grid", gap: 4,
        }}>
          {log.map((s) => (
            <li key={s.n} style={{ display: "grid", gridTemplateColumns: "26px 150px 1fr 14px", gap: 8, alignItems: "center", padding: "2px 0" }}>
              <span style={{ color: TOKENS.muted }}>#{s.n}</span>
              <span style={{ color: TOKENS.accent, fontWeight: 600 }}>{s.tool}</span>
              <span style={{ color: TOKENS.ink2, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{s.summary}</span>
              <span style={{ color: TOKENS.pos }}>{s.ok ? "✓" : "✗"}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

// ── Stat cell ──────────────────────────────────────────────────────────────
function Stat({ label, value, mono, accent }) {
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
function ModelPickerCard({ models, value, onChange, compact }) {
  const cur = models.find(m => m.id === value) || models[0];
  const [open, setOpen] = useState(false);
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
          <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>{cur.provider} · {cur.context}</span>
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
                <div style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>{m.provider} · ctx {m.context}</div>
              </div>
              {m.id === value && <Icon.check/>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ProviderDot({ provider }) {
  const C = { anthropic:"#D97757", openai:"#10A37F", google:"#4285F4", openrouter:"#7A4DA8", azure:"#0078D4" };
  const c = C[provider] || TOKENS.muted;
  return <span style={{ display:"inline-block", width: 9, height: 9, borderRadius: 999, background: c, flex: "0 0 auto" }}/>;
}

// ── Primary button ─────────────────────────────────────────────────────────
function Button({ children, primary, onClick, icon, small, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      display:"inline-flex", alignItems:"center", gap: 6,
      padding: small ? "5px 10px" : "7px 13px",
      fontFamily: FONT_SANS, fontWeight: 600, fontSize: small ? 12 : 13,
      borderRadius: 6, cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.45 : 1,
      background: primary ? TOKENS.accent : "white",
      color: primary ? "white" : TOKENS.ink,
      border: `1px solid ${primary ? TOKENS.accent : TOKENS.rule}`,
    }}>
      {icon}
      {children}
    </button>
  );
}

// ── Section header ─────────────────────────────────────────────────────────
function SectionLabel({ children, right }) {
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

Object.assign(window, {
  TOKENS, FONT_SANS, FONT_SERIF, FONT_MONO,
  Icon, useDensity, DENSITY,
  MolStructurePlaceholder, MolTile, CopyButton,
  RoleChip, ConditionRow, YieldBadge,
  StructuredInsightPanel, ActivityDrawer, Stat,
  ModelPickerCard, ProviderDot, Button, SectionLabel,
});
