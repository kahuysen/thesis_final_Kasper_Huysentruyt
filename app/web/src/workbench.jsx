// Workbench layout — ES-modulized from claude design /app/workbench.jsx.
// All MOCK_* references replaced with props. Real backend wires sit in App.jsx.

import {
  useState,
  useEffect,
  useRef,
  useMemo,
  Fragment,
} from 'react';

import {
  TOKENS, FONT_SANS, FONT_SERIF, FONT_MONO,
  DENSITY, useDensity,
  Icon, MolStructurePlaceholder, MolTile, CopyButton,
  RoleChip, ConditionRow, YieldBadge,
  StructuredInsightPanel, ActivityDrawer, Stat,
  ModelPickerCard, ProviderDot, Button, SectionLabel,
  GhentBadge,
} from './shared.jsx';
import {
  createFolder, deleteFolder, fetchFolders, fetchRuns, fileUrl,
  moveRun, renameFolder,
} from './api.js';
import { SettingsModal } from './SettingsModal.jsx';

export function WorkbenchApp({
  extraction,
  agentLog = [],
  insight,                  // { [entry_id]: { reaction_class, ... } } — adapted from /api/runs/{id}/insight
  metadata,                 // { steps, input_tokens, output_tokens, wall_time_s? }
  models = [],
  modelId,
  onModelChange,
  sourceImageUrl,
  runId,
  onUploadFile,
  onOpenExistingRun,        // (item) => void  — picks a run from "My runs"
  onSettingsSaved,          // () => void      — App.jsx refetches health
  onRunInsight,
  rxnInsightAvailable = false,
  downloads,                // { csv?: string, json?: string, source?: string }  (URLs)
  phase = 'idle',           // 'idle' | 'uploading' | 'streaming' | 'done' | 'error'
  errorMessage,
  backendInfo,
}) {
  const dens = useDensity('comfy');

  const data = useMemo(() => ({
    source_image: sourceImageUrl || null,
    figure_caption: extraction?.figure_caption || null,
    extraction_notes: extraction?.extraction_notes || null,
    reactions: extraction?.reactions || [],
  }), [extraction, sourceImageUrl]);

  const reactions = data.reactions;
  const hasRun = phase !== 'idle';

  const [layout, setLayout]         = useState('table');
  const [logOpen, setLogOpen]       = useState(true);
  const [detailRxn, setDetailRxn]   = useState(null);
  const [lightboxOpen, setLightbox] = useState(false);
  const [showKbd, setShowKbd]       = useState(false);
  const [showMyRuns, setShowMyRuns] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [signedIn, setSignedIn]     = useState(true);

  const [query, setQuery]           = useState("");
  const [sort, setSort]             = useState({ key: null, dir: "asc" });
  const [verified, setVerified]     = useState(new Set());
  const [selected, setSelected]     = useState(new Set());

  useEffect(() => {
    setVerified(new Set());
    setSelected(new Set());
    setDetailRxn(null);
    setSort({ key: null, dir: "asc" });
  }, [runId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let rows = reactions.map((r, i) => ({ r, i }));
    if (q) {
      rows = rows.filter(({ r }) => {
        const hay = [
          r.entry_id, r.title,
          ...(r.reactants||[]).flatMap(m => [m.name, m.smiles, m.label]),
          ...(r.reagents||[]).flatMap(m => [m.name, m.smiles, m.role]),
          ...(r.products||[]).flatMap(m => [m.name, m.smiles, m.label]),
          r.conditions?.solvent, r.conditions?.temperature, r.conditions?.time,
        ].filter(Boolean).join(" ").toLowerCase();
        return hay.includes(q);
      });
    }
    if (sort.key) {
      rows = [...rows].sort((a, b) => {
        const av = sortVal(a.r, sort.key), bv = sortVal(b.r, sort.key);
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        const cmp = av < bv ? -1 : av > bv ? 1 : 0;
        return sort.dir === "desc" ? -cmp : cmp;
      });
    }
    return rows;
  }, [reactions, query, sort]);

  const insights = useMemo(() => {
    if (!insight) return {};
    const out = {};
    for (const r of reactions) if (insight[r.entry_id]) out[r.entry_id] = insight[r.entry_id];
    return out;
  }, [reactions, insight]);

  const totalProducts = reactions.reduce((a,r) => a + (r.products||[]).length, 0);
  const yields = reactions.flatMap(r => (r.products||[]).map(p => p.yield_pct)).filter(v => v != null);
  const avgYield = yields.length ? Math.round(yields.reduce((a,b)=>a+b,0) / yields.length) : null;

  function toggleSelected(i) {
    const s = new Set(selected); s.has(i) ? s.delete(i) : s.add(i); setSelected(s);
  }
  function selectAll() {
    if (selected.size === filtered.length) setSelected(new Set());
    else setSelected(new Set(filtered.map(x => x.i)));
  }
  function bulkVerify(state) {
    const v = new Set(verified);
    selected.forEach(i => state ? v.add(i) : v.delete(i));
    setVerified(v);
  }

  return (
    <div style={{
      position: "relative",
      width: "100%", height: "100%",
      background: TOKENS.paper, color: TOKENS.ink,
      fontFamily: FONT_SANS, fontSize: 14, lineHeight: 1.45,
      display: "grid", gridTemplateRows: "auto 1fr auto",
      overflow: "hidden",
    }}>
      <TopBar
        models={models}
        modelId={modelId} onModelChange={onModelChange}
        reactions={reactions} verified={verified}
        onShowKbd={() => setShowKbd(v => !v)} kbdActive={showKbd}
        onSignOut={() => setSignedIn(false)}
        onUploadFile={onUploadFile}
        onOpenMyRuns={() => setShowMyRuns(true)}
        onOpenSettings={() => setShowSettings(true)}
        runId={runId}
        phase={phase}
        backendInfo={backendInfo}
      />

      <div style={{
        display: "grid", gridTemplateColumns: `320px 1fr ${detailRxn != null ? "440px" : "0px"}`,
        gap: 0, minHeight: 0, transition: "grid-template-columns 0.18s ease",
      }}>
        <LeftRail
          data={data} reactions={reactions}
          totalProducts={totalProducts} avgYield={avgYield}
          verifiedCount={verified.size}
          metadata={metadata}
          downloads={downloads}
          phase={phase}
          onUploadFile={onUploadFile}
          onRunInsight={onRunInsight}
          rxnInsightAvailable={rxnInsightAvailable}
          insightDone={!!insight}
          onOpenLightbox={() => data.source_image && setLightbox(true)}
        />

        <main style={{ overflow: "auto", padding: "16px 22px 24px", minWidth: 0 }}>
          {phase === 'error' && (
            <ErrorBanner message={errorMessage}/>
          )}
          {!hasRun && (
            <Landing onUploadFile={onUploadFile}/>
          )}
          {hasRun && phase !== 'done' && reactions.length === 0 && (
            <BusyState phase={phase} agentLog={agentLog}/>
          )}

          {hasRun && reactions.length > 0 && (
            <>
              <ReactionsToolbar
                layout={layout} onLayout={setLayout}
                query={query} onQuery={setQuery}
                selectedCount={selected.size}
                filteredCount={filtered.length} totalCount={reactions.length}
                onBulkVerify={() => bulkVerify(true)}
                onBulkUnverify={() => bulkVerify(false)}
                onBulkClear={() => setSelected(new Set())}
              />

              {layout === "table" && (
                <ReactionTable
                  rows={filtered} dens={dens}
                  detailRxn={detailRxn} onOpenDetail={setDetailRxn}
                  verified={verified}
                  onToggleVerified={(i) => {
                    const v = new Set(verified); v.has(i) ? v.delete(i) : v.add(i); setVerified(v);
                  }}
                  selected={selected}
                  onToggleSelected={toggleSelected}
                  onSelectAll={selectAll}
                  allSelected={filtered.length > 0 && selected.size === filtered.length}
                  sort={sort} onSort={setSort}
                />
              )}
              {layout === "equation" && <ReactionEquations rows={filtered} dens={dens} onOpenDetail={setDetailRxn}/>}

              {filtered.length === 0 && (
                <EmptyState query={query} onClear={() => setQuery("")}/>
              )}

              {Object.keys(insights).length > 0 && (
                <div style={{ marginTop: 22 }}>
                  <SectionLabel right={<span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>via rxn-insight</span>}>
                    Enrichment
                  </SectionLabel>
                  <StructuredInsightPanel analyses={insights} dens={dens}/>
                </div>
              )}
            </>
          )}
        </main>

        {detailRxn != null && reactions[detailRxn] && (
          <DetailDrawer
            rxn={reactions[detailRxn]}
            insight={insight?.[reactions[detailRxn]?.entry_id]}
            verified={verified.has(detailRxn)}
            onClose={() => setDetailRxn(null)}
            onToggleVerified={() => {
              const v = new Set(verified);
              v.has(detailRxn) ? v.delete(detailRxn) : v.add(detailRxn);
              setVerified(v);
            }}
            dens={dens}
          />
        )}
      </div>

      <ActivityDrawer
        log={agentLog} open={logOpen}
        onToggle={() => setLogOpen(o => !o)}
        visible={hasRun}
        meta={metadata}
      />

      {lightboxOpen && data.source_image && (
        <FigureLightbox src={data.source_image} caption={data.figure_caption} onClose={() => setLightbox(false)}/>
      )}
      {showKbd      && <KeyboardCheatSheet onClose={() => setShowKbd(false)}/>}
      {showMyRuns   && (
        <MyRunsModal
          currentRunId={runId}
          onPick={(item) => { setShowMyRuns(false); onOpenExistingRun?.(item); }}
          onClose={() => setShowMyRuns(false)}
        />
      )}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onSaved={() => onSettingsSaved?.()}
        />
      )}
      {!signedIn    && <SignInScreen user={MOCK_USER} onSignIn={() => setSignedIn(true)}/>}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────
function sortVal(r, key) {
  if (key === "entry") return r.entry_id || "";
  if (key === "yield") {
    const ys = (r.products||[]).map(p => p.yield_pct).filter(v => v != null);
    return ys.length ? Math.max(...ys) : null;
  }
  if (key === "title") return r.title || "";
  return null;
}

// ── Top bar ──────────────────────────────────────────────────────────────
function TopBar({
  models, modelId, onModelChange, reactions, verified,
  onShowKbd, kbdActive, onSignOut,
  onUploadFile,
  onOpenMyRuns, onOpenSettings,
  runId, phase, backendInfo,
}) {
  const fileRef = useRef(null);
  function triggerUpload() { fileRef.current?.click(); }
  function onFileChange(e) {
    const f = e.target.files?.[0];
    if (f) onUploadFile?.(f);
    e.target.value = "";
  }

  const busy = phase === 'uploading' || phase === 'streaming';

  return (
    <header style={{
      display: "flex", alignItems: "center", gap: 14,
      padding: "10px 18px",
      borderBottom: `1px solid ${TOKENS.rule}`,
      background: "white",
    }}>
      <div style={{ display:"flex", alignItems:"center", gap: 10 }}>
        <img
          src="/static/rxn_extraction_logo.png"
          alt="Rxn-EXTRACTION"
          style={{
            width: 36, height: 36, borderRadius: 8,
            objectFit: "cover", objectPosition: "center 28%",
            border: `1px solid ${TOKENS.rule}`,
            display: "block",
          }}
        />
        <div style={{ display:"flex", flexDirection:"column", gap: 0 }}>
          <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink, letterSpacing: 0.1 }}>Rxn-EXTRACTION</span>
          {runId && (
            <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>
              run · {runId}
            </span>
          )}
        </div>
      </div>

      <div style={{ flex: 1 }}/>

      {phase === 'done' && (
        <div style={{
          display:"inline-flex", alignItems:"center", gap: 6,
          padding: "4px 10px", borderRadius: 999,
          background: TOKENS.pos2, border: `1px solid ${TOKENS.pos}33`,
          color: TOKENS.pos, fontFamily: FONT_SANS, fontSize: 11.5, fontWeight: 600,
        }}>
          <span style={{ width: 7, height: 7, borderRadius:999, background: TOKENS.pos }}/>
          {reactions.length} reactions · {verified.size} verified
        </div>
      )}
      {busy && (
        <div style={{
          display:"inline-flex", alignItems:"center", gap: 6,
          padding: "4px 10px", borderRadius: 999,
          background: TOKENS.warn2, border: `1px solid ${TOKENS.warn}33`,
          color: TOKENS.warn, fontFamily: FONT_SANS, fontSize: 11.5, fontWeight: 600,
        }}>
          <span style={{ width: 7, height: 7, borderRadius:999, background: TOKENS.warn,
                          animation: "rxn-pulse 1.2s ease-in-out infinite" }}/>
          {phase === 'uploading' ? 'Uploading…' : 'Extracting…'}
        </div>
      )}

      <ModelPickerCard models={models} value={modelId} onChange={onModelChange} compact/>

      <button onClick={onShowKbd} title="Keyboard shortcuts (?)" style={{
        width: 28, height: 28, padding: 0,
        background: kbdActive ? TOKENS.ink : "white",
        color: kbdActive ? "white" : TOKENS.ink2,
        border: `1px solid ${kbdActive ? TOKENS.ink : TOKENS.rule}`,
        borderRadius: 6, cursor: "pointer",
        fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14,
      }}>?</button>

      <input ref={fileRef} type="file" accept="image/*,application/pdf,.pdf" onChange={onFileChange} style={{ display: 'none' }}/>
      <Button icon={<Icon.upload/>} onClick={triggerUpload} disabled={busy}>
        {busy ? "Working…" : "Upload figure or PDF"}
      </Button>

      <span style={{ width: 1, height: 22, background: TOKENS.rule, margin: "0 2px" }}/>
      <UserMenu onSignOut={onSignOut} onOpenMyRuns={onOpenMyRuns} onOpenSettings={onOpenSettings}/>

      <style>{`@keyframes rxn-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }`}</style>
    </header>
  );
}

// ── User menu (avatar + popover) ────────────────────────────────────────
const MOCK_USER = {
  name:    "Researcher",
  email:   "you@lab.org",
  role:    "Researcher",
  org:     "ReactionMiner",
  initials:"R",
  tone:    "#7A4DA8",
  runs_today: 0,
  quota:   { used: 0, total: 500 },
};

function UserMenu({ onSignOut, onOpenMyRuns, onOpenSettings }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const onDown = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, []);

  const u = MOCK_USER;
  const pct = Math.round((u.quota.used / u.quota.total) * 100);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button onClick={() => setOpen(o => !o)} title={u.name} style={{
        width: 30, height: 30, borderRadius: 999,
        background: u.tone, color: "white",
        border: `2px solid ${open ? TOKENS.ink : "white"}`,
        boxShadow: open ? "none" : `0 0 0 1px ${TOKENS.rule}`,
        display: "grid", placeItems: "center", cursor: "pointer",
        fontFamily: FONT_SANS, fontWeight: 700, fontSize: 11.5, letterSpacing: 0.4,
        padding: 0,
      }}>{u.initials}</button>

      {open && (
        <div style={{
          position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 50,
          width: 280,
          background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
          boxShadow: "0 8px 28px rgba(15,26,36,0.14)",
          overflow: "hidden",
        }}>
          <div style={{ display:"flex", alignItems:"center", gap: 10, padding: 14, borderBottom: `1px solid ${TOKENS.rule2}` }}>
            <div style={{
              width: 40, height: 40, borderRadius: 999,
              background: u.tone, color: "white",
              display: "grid", placeItems: "center",
              fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14,
            }}>{u.initials}</div>
            <div style={{ display:"flex", flexDirection:"column", minWidth: 0, flex: 1 }}>
              <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{u.name}</span>
              <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{u.email}</span>
            </div>
          </div>

          <div style={{ padding: 6 }}>
            <MenuItem onClick={() => { setOpen(false); onOpenMyRuns && onOpenMyRuns(); }}>My runs</MenuItem>
            <MenuItem onClick={() => { setOpen(false); onOpenSettings && onOpenSettings(); }}>Settings</MenuItem>
            <div style={{ height: 1, background: TOKENS.rule2, margin: "4px 0" }}/>
            <MenuItem tone={TOKENS.err} onClick={() => { setOpen(false); onSignOut && onSignOut(); }}>Sign out</MenuItem>
          </div>
        </div>
      )}
    </div>
  );
}

function MenuItem({ children, tone, onClick }) {
  return (
    <button onClick={onClick} style={{
      display: "block", width: "100%", textAlign: "left",
      padding: "7px 10px", border: "none", background: "transparent",
      borderRadius: 4, cursor: "pointer",
      fontFamily: FONT_SANS, fontSize: 13, color: tone || TOKENS.ink2,
      fontWeight: 500,
    }} onMouseOver={(e) => e.currentTarget.style.background = TOKENS.paper}
       onMouseOut={(e) => e.currentTarget.style.background = "transparent"}>
      {children}
    </button>
  );
}

// ── Left rail ────────────────────────────────────────────────────────────
function LeftRail({
  data, reactions, totalProducts, avgYield, verifiedCount,
  metadata, downloads, phase, onUploadFile, onOpenLightbox,
  onRunInsight, rxnInsightAvailable, insightDone,
}) {
  const canRunInsight = !!rxnInsightAvailable && phase === 'done';
  return (
    <aside style={{
      borderRight: `1px solid ${TOKENS.rule}`,
      background: "white",
      padding: 16, display: "flex", flexDirection: "column", gap: 14,
      overflow: "auto", minWidth: 0,
    }}>
      <SectionLabel right={data.source_image ? <button onClick={onOpenLightbox} style={{
        background:"transparent", border:"none", padding: 0, cursor:"pointer",
        fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.accent, fontWeight: 600,
      }}>Enlarge ↗</button> : null}>Source figure</SectionLabel>

      {data.source_image ? (
        <button onClick={onOpenLightbox} className="figure-thumb" style={{
          background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`,
          borderRadius: 6, padding: 8, position: "relative", cursor: "zoom-in",
          display: "block", width: "100%", textAlign: "left",
        }}>
          <img src={data.source_image} alt="source figure"
               style={{ width: "100%", display: "block", borderRadius: 3 }}/>
        </button>
      ) : (
        <SidebarDropzone onUploadFile={onUploadFile} disabled={phase === 'uploading' || phase === 'streaming'}/>
      )}
      {data.figure_caption && (
        <p style={{
          margin: 0, fontFamily: FONT_SERIF, fontStyle: "italic",
          fontSize: 13, color: TOKENS.ink2, lineHeight: 1.4,
        }}>{data.figure_caption}</p>
      )}

      {(reactions.length > 0 || metadata) && (
        <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
          <SectionLabel>Run metadata</SectionLabel>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(2, 1fr)", gap: 10, marginBottom: 10 }}>
            <Stat label="Reactions" value={reactions.length}/>
            <Stat label="Products"  value={totalProducts}/>
            <Stat label="Avg yield" value={avgYield != null ? `${avgYield}%` : "—"} accent={avgYield != null && avgYield >= 75 ? TOKENS.pos : null}/>
            <Stat label="Verified"  value={`${verifiedCount} / ${reactions.length}`}/>
            {metadata?.steps != null && <Stat label="Steps"     value={metadata.steps} mono/>}
            {metadata?.wall_time_s != null && <Stat label="Wall time" value={`${metadata.wall_time_s.toFixed(1)}s`} mono/>}
          </div>
          {metadata?.input_tokens != null && (
            <div style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted, display:"flex", justifyContent:"space-between" }}>
              <span>tokens</span>
              <span>{(metadata.input_tokens || 0).toLocaleString()} in · {(metadata.output_tokens || 0).toLocaleString()} out</span>
            </div>
          )}
        </div>
      )}

      {data.extraction_notes && (
        <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
          <SectionLabel>Extractor note</SectionLabel>
          <p style={{
            margin: 0, padding: "8px 10px",
            background: TOKENS.warn2 + "55", borderLeft: `2px solid ${TOKENS.warn}`,
            fontFamily: FONT_SERIF, fontSize: 13, color: TOKENS.ink2, fontStyle: "italic", lineHeight: 1.5,
          }}>{data.extraction_notes}</p>
        </div>
      )}

      {(canRunInsight || (downloads && (downloads.csv || downloads.json || downloads.source || downloads.insight))) && (
        <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
          <SectionLabel>Export</SectionLabel>
          <div style={{ display:"grid", gap: 6 }}>
            {canRunInsight && (
              <Button primary icon={<Icon.spark/>}
                      onClick={onRunInsight}
                      disabled={insightDone}>
                {insightDone ? "Rxn-INSIGHT done" : "Postprocess with Rxn-INSIGHT"}
              </Button>
            )}
            {downloads?.csv     && <Button as="a" href={downloads.csv}     icon={<Icon.download/>}>reactions.csv</Button>}
            {downloads?.json    && <Button as="a" href={downloads.json}    icon={<Icon.download/>}>extraction.json</Button>}
            {downloads?.insight && <Button as="a" href={downloads.insight} icon={<Icon.download/>}>insight.csv</Button>}
            {downloads?.source  && <Button as="a" href={downloads.source}  icon={<Icon.download/>}>source figure</Button>}
          </div>
        </div>
      )}

      <div style={{ flex: 1 }}/>
      <ThesisAttribution/>
    </aside>
  );
}

function ThesisAttribution() {
  return (
    <div style={{
      borderTop: `1px solid ${TOKENS.rule}`,
      paddingTop: 14, marginTop: 12,
      display: "grid",
      gridTemplateColumns: "1fr auto",
      alignItems: "center",
      gap: 10,
    }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
        <span style={{
          fontFamily: FONT_SANS, fontSize: 10, fontWeight: 700,
          color: TOKENS.muted, textTransform: "uppercase", letterSpacing: 0.6,
        }}>Master's thesis · 2025</span>
        <a
          href="https://www.linkedin.com/in/kasper-huysentruyt-a972a6236"
          target="_blank"
          rel="noreferrer"
          title="Kasper Huysentruyt on LinkedIn"
          style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            fontFamily: FONT_SERIF, fontStyle: "italic",
            fontSize: 14.5, color: TOKENS.accent,
            textDecoration: "none", lineHeight: 1.2,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = TOKENS.accent2}
          onMouseLeave={(e) => e.currentTarget.style.color = TOKENS.accent}
        >
          Kasper Huysentruyt
          <Icon.ext/>
        </a>
      </div>
      <GhentBadge height={56}/>
    </div>
  );
}

function SidebarDropzone({ onUploadFile, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault(); setDrag(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !disabled) onUploadFile?.(f);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        cursor: disabled ? "not-allowed" : "pointer",
        background: drag ? TOKENS.pos2 : TOKENS.paper,
        border: `2px dashed ${drag ? TOKENS.pos : TOKENS.rule}`,
        borderRadius: 6,
        padding: 24,
        display: "grid",
        placeItems: "center",
        gap: 8,
        textAlign: "center",
        color: TOKENS.muted,
        fontFamily: FONT_SANS,
        fontSize: 12.5,
      }}
    >
      <Icon.upload/>
      <span style={{ fontWeight: 600, color: TOKENS.ink2 }}>Drop a figure here</span>
      <span style={{ fontSize: 11 }}>or click to choose</span>
      <input ref={inputRef} type="file" accept="image/*,application/pdf,.pdf" style={{ display: "none" }}
             onChange={(e) => {
               const f = e.target.files?.[0];
               if (f) onUploadFile?.(f);
               e.target.value = "";
             }}/>
    </div>
  );
}

// ── Reactions toolbar ───────────────────────────────────────────────────
function ReactionsToolbar({
  layout, onLayout, query, onQuery,
  selectedCount, filteredCount, totalCount,
  onBulkVerify, onBulkUnverify, onBulkClear,
}) {
  const bulk = selectedCount > 0;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "1fr auto",
      alignItems: "center", gap: 14, marginBottom: 12,
    }}>
      {bulk ? (
        <div style={{
          display:"flex", alignItems:"center", gap: 10,
          padding: "8px 12px",
          background: "#EEF6FF", border: `1px solid ${TOKENS.accent}55`,
          borderRadius: 6, color: TOKENS.accent,
          fontFamily: FONT_SANS, fontSize: 13, fontWeight: 600,
        }}>
          <span>{selectedCount} selected</span>
          <span style={{ width: 1, height: 16, background: TOKENS.accent + "33" }}/>
          <button onClick={onBulkVerify}   style={bulkBtn}>✓ Verify</button>
          <button onClick={onBulkUnverify} style={bulkBtn}>↺ Unverify</button>
          <div style={{ flex: 1 }}/>
          <button onClick={onBulkClear} style={{ ...bulkBtn, color: TOKENS.muted }}>Clear</button>
        </div>
      ) : (
        <div style={{ display:"flex", alignItems:"center", gap: 10 }}>
          <h2 style={{
            margin: 0, fontFamily: FONT_SANS, fontWeight: 700, fontSize: 18, color: TOKENS.ink,
          }}>Extracted reactions</h2>
          <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted, marginTop: 4 }}>
            {query ? `${filteredCount} of ${totalCount}` : `${totalCount} total`}
          </span>
          <div style={{ flex: 1 }}/>
          <div style={{
            position:"relative",
            display:"flex", alignItems:"center", gap: 6,
            background: "white", border: `1px solid ${TOKENS.rule}`,
            borderRadius: 6, padding: "0 8px", width: 280,
          }}>
            <Icon.search/>
            <input
              value={query}
              onChange={(e) => onQuery(e.target.value)}
              placeholder="Filter by name, SMILES, conditions…"
              style={{
                flex: 1, minWidth: 0,
                border: "none", outline: "none",
                padding: "7px 0",
                background: "transparent",
                fontFamily: FONT_SANS, fontSize: 12.5, color: TOKENS.ink,
              }}
            />
            <span style={{
              fontFamily: FONT_MONO, fontSize: 10, color: TOKENS.muted,
              border: `1px solid ${TOKENS.rule}`, borderRadius: 3, padding: "1px 5px",
            }}>/</span>
          </div>
        </div>
      )}
      <div style={{ display:"flex", alignItems:"center", gap: 8 }}>
        <SegmentedControl value={layout} onChange={onLayout}
          options={[{v:"table", l:"Table"}, {v:"equation", l:"Equation"}]}/>
      </div>
    </div>
  );
}
const bulkBtn = {
  background: "transparent", border: "none",
  fontFamily: FONT_SANS, fontWeight: 600, fontSize: 12.5,
  color: TOKENS.accent, cursor: "pointer", padding: "2px 4px",
};

// ── Segmented control ───────────────────────────────────────────────────
function SegmentedControl({ value, onChange, options }) {
  return (
    <div style={{
      display:"inline-flex", padding: 2, background: TOKENS.paper,
      border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
    }}>
      {options.map(o => (
        <button key={o.v} onClick={() => onChange(o.v)} style={{
          padding: "5px 12px", border: "none",
          background: value === o.v ? "white" : "transparent",
          color: value === o.v ? TOKENS.ink : TOKENS.muted,
          fontFamily: FONT_SANS, fontWeight: 600, fontSize: 12,
          borderRadius: 4, cursor: "pointer",
          boxShadow: value === o.v ? `0 1px 2px rgba(15,26,36,0.06), inset 0 0 0 1px ${TOKENS.rule}` : "none",
        }}>{o.l}</button>
      ))}
    </div>
  );
}

// ── Reaction table ──────────────────────────────────────────────────────
function ReactionTable({
  rows, dens, detailRxn, onOpenDetail,
  verified, onToggleVerified,
  selected, onToggleSelected, onSelectAll, allSelected,
  sort, onSort,
}) {
  const headerSort = (key, label, align) => {
    const active = sort.key === key;
    const dir = active ? sort.dir : null;
    return (
      <button onClick={() => onSort({ key, dir: active && sort.dir === "asc" ? "desc" : "asc" })}
              style={{
                display:"inline-flex", alignItems:"center", gap: 4,
                background:"transparent", border:"none", padding: 0, cursor:"pointer",
                fontFamily: FONT_SANS, fontWeight: 600, fontSize: 11,
                textTransform: "uppercase", letterSpacing: 0.6,
                color: active ? TOKENS.ink : TOKENS.muted,
              }}>
        {label}
        <span style={{ fontSize: 10, opacity: active ? 1 : 0.3 }}>
          {dir === "desc" ? "▼" : "▲"}
        </span>
      </button>
    );
  };
  return (
    <div style={{
      background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
      overflow: "hidden",
    }}>
      <table style={{
        width: "100%", borderCollapse: "collapse",
        fontFamily: FONT_SANS, fontSize: dens.font - 1,
      }}>
        <thead>
          <tr style={{ background: TOKENS.paper, textAlign: "left" }}>
            <Th w="32px">
              <input type="checkbox" checked={allSelected} onChange={onSelectAll}
                     style={{ accentColor: TOKENS.accent, margin: 0 }}/>
            </Th>
            <Th w="28px"></Th>
            <Th w="90px">{headerSort("entry", "Entry")}</Th>
            <Th>Reactants</Th>
            <Th>Reagents · conditions</Th>
            <Th>Products</Th>
            <Th w="76px" align="right">{headerSort("yield", "Yield", "right")}</Th>
            <Th w="68px"></Th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ r, i }) => {
            const ver = verified.has(i);
            const sel = selected.has(i);
            const open = detailRxn === i;
            return (
              <tr key={i}
                  onClick={() => onOpenDetail(open ? null : i)}
                  className="rxn-row"
                  style={{
                    borderTop: `1px solid ${TOKENS.rule}`,
                    background: open ? "#EEF6FF" : (sel ? "#F4F2E8" : "white"),
                    cursor: "pointer", position: "relative",
                  }}>
                <Td>
                  <input type="checkbox" checked={sel}
                         onClick={(e) => e.stopPropagation()}
                         onChange={() => onToggleSelected(i)}
                         style={{ accentColor: TOKENS.accent, margin: 0 }}/>
                </Td>
                <Td>
                  <button onClick={(e) => { e.stopPropagation(); onToggleVerified(i); }} style={{
                    width: 18, height: 18, padding: 0,
                    background: ver ? TOKENS.pos : "white",
                    border: `1.5px solid ${ver ? TOKENS.pos : TOKENS.rule}`,
                    borderRadius: 4, cursor: "pointer", color: "white",
                    display:"grid", placeItems:"center",
                  }} title={ver ? "Verified" : "Mark verified"}>
                    {ver && <Icon.check/>}
                  </button>
                </Td>
                <Td>
                  <code style={{
                    fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
                    background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
                    padding: "1px 6px", borderRadius: 3,
                  }}>{r.entry_id}</code>
                </Td>
                <Td>
                  <div style={{ display:"flex", flexWrap:"wrap", gap: 4, alignItems:"center" }}>
                    {(r.reactants || []).map((m, j) => <MolChip key={j} mol={m}/>)}
                  </div>
                </Td>
                <Td><ConditionRow conditions={r.conditions} reagents={r.reagents} compact/></Td>
                <Td>
                  <div style={{ display:"flex", flexWrap:"wrap", gap: 4, alignItems:"center" }}>
                    {(r.products || []).map((m, j) => <MolChip key={j} mol={m}/>)}
                  </div>
                </Td>
                <Td align="right">
                  <div style={{ display:"flex", justifyContent:"flex-end", gap: 4, flexWrap:"wrap" }}>
                    {(r.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note}/>)}
                  </div>
                </Td>
                <Td align="right">
                  <div className="row-actions" style={{
                    display: "flex", gap: 4, justifyContent: "flex-end", opacity: open ? 1 : 0.45,
                  }}>
                    <RowIconBtn title="Copy entry SMILES">  <Icon.copy/></RowIconBtn>
                    <RowIconBtn title="Open detail">  <Icon.ext/></RowIconBtn>
                  </div>
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RowIconBtn({ title, children }) {
  return (
    <button title={title} onClick={(e) => e.stopPropagation()} style={{
      width: 22, height: 22, padding: 0,
      background: "transparent", border: `1px solid ${TOKENS.rule}`,
      color: TOKENS.muted, borderRadius: 4, cursor: "pointer",
      display:"grid", placeItems:"center",
    }}>{children}</button>
  );
}

function Th({ children, w, align }) {
  return (
    <th style={{
      padding: "9px 12px", fontWeight: 600, fontSize: 11,
      textTransform: "uppercase", letterSpacing: 0.6, color: TOKENS.muted,
      borderBottom: `1px solid ${TOKENS.rule}`,
      width: w, textAlign: align || "left", whiteSpace: "nowrap",
    }}>{children}</th>
  );
}
function Td({ children, align }) {
  return <td style={{ padding: "10px 12px", verticalAlign: "middle", textAlign: align || "left", color: TOKENS.ink2 }}>{children}</td>;
}

// ── MolChip ─────────────────────────────────────────────────────────────
function MolChip({ mol }) {
  return (
    <div style={{
      display:"inline-flex", alignItems:"center", gap: 6,
      background: "white", border: `1px solid ${TOKENS.rule}`,
      borderRadius: 4, padding: "3px 7px 3px 4px",
    }}>
      <div style={{ width: 26, height: 22, display:"grid", placeItems:"center", flex:"0 0 auto" }}>
        <MolStructurePlaceholder smiles={mol.smiles} width={26} height={22}/>
      </div>
      <span style={{ fontFamily: FONT_SANS, fontSize: 12, color: TOKENS.ink, fontWeight: 500, whiteSpace:"nowrap" }}>
        {mol.label && <span style={{ color: TOKENS.accent, fontWeight: 700, marginRight: 4 }}>{mol.label}</span>}
        {mol.name || "—"}
      </span>
    </div>
  );
}

// ── Equation view ────────────────────────────────────────────────────────
function ReactionEquations({ rows, dens, onOpenDetail }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap: 14 }}>
      {rows.map(({ r, i }) => (
        <article key={i} onClick={() => onOpenDetail(i)} style={{
          background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
          overflow: "hidden", cursor: "pointer",
        }}>
          <header style={{
            display:"flex", alignItems:"center", gap: 10,
            padding: `${dens.pad}px ${dens.pad+4}px`,
            borderBottom: `1px solid ${TOKENS.rule}`,
            background: TOKENS.paper,
          }}>
            <code style={{
              fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
              background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
              padding: "1px 6px", borderRadius: 3,
            }}>{r.entry_id}</code>
            <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 14, color: TOKENS.ink, flex: 1 }}>{r.title}</span>
            {(r.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note} big/>)}
          </header>
          <EquationRow rxn={r} dens={dens}/>
          {r.notes && (
            <p style={{ margin: 0, padding: `0 ${dens.pad+4}px ${dens.pad}px`, fontFamily: FONT_SERIF, fontSize: 13, color: TOKENS.ink2, fontStyle: "italic" }}>{r.notes}</p>
          )}
        </article>
      ))}
    </div>
  );
}

// ── Equation row + parts (shared with detail drawer) ────────────────────
function EquationRow({ rxn, dens }) {
  const tileW = 168;
  return (
    <div style={{
      display:"flex", alignItems:"center", gap: 8,
      padding: dens.pad+2, overflowX: "auto",
    }}>
      {(rxn.reactants || []).map((m, i) => (
        <Fragment key={i}>
          {i > 0 && <PlusSep/>}
          <MolTile mol={m} kind="reactant" dens={dens} compactWidth={tileW}/>
        </Fragment>
      ))}
      {(!rxn.reactants || rxn.reactants.length === 0) && <EmptyTile dens={dens} w={tileW}/>}
      <ArrowBlock rxn={rxn}/>
      {(rxn.products || []).map((m, i) => (
        <Fragment key={i}>
          {i > 0 && <PlusSep/>}
          <MolTile mol={m} kind="product" dens={dens} compactWidth={tileW}/>
        </Fragment>
      ))}
      {(!rxn.products || rxn.products.length === 0) && <EmptyTile dens={dens} w={tileW}/>}
    </div>
  );
}
function PlusSep() { return <span style={{ fontSize: 22, color: TOKENS.muted, fontWeight: 300, padding: "0 4px" }}>+</span>; }
function EmptyTile({ dens, w }) {
  return <div style={{ width: w, height: dens.tileH + 64, display:"grid", placeItems:"center", color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 11, border: `1px dashed ${TOKENS.rule}`, borderRadius: 6 }}>(none)</div>;
}
function ArrowBlock({ rxn }) {
  const top = (rxn.reagents || []).map(r => `${r.label ? r.label + ": " : ""}${r.name || r.smiles || "?"}${r.loading ? ` (${r.loading})` : ""}`).join(", ");
  const c = rxn.conditions || {};
  const bottom = [c.solvent, c.temperature, c.time, c.atmosphere].filter(Boolean).join(" · ");
  return (
    <div style={{ flex:"0 0 auto", minWidth: 170, maxWidth: 240, display:"flex", flexDirection:"column", alignItems:"center", gap: 2, padding: "0 4px" }}>
      <div style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.ink, textAlign:"center", lineHeight: 1.3, wordBreak:"break-word" }}>{top}</div>
      <div style={{ width:"100%", height: 14 }}><Icon.arrow/></div>
      <div style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.muted, textAlign:"center", lineHeight: 1.3, wordBreak:"break-word" }}>{bottom}</div>
    </div>
  );
}

// ── Detail drawer ───────────────────────────────────────────────────────
function DetailDrawer({ rxn, insight, verified, onClose, onToggleVerified, dens }) {
  if (!rxn) return null;
  return (
    <aside style={{
      borderLeft: `1px solid ${TOKENS.rule}`,
      background: "white",
      display: "grid", gridTemplateRows: "auto 1fr auto",
      overflow: "hidden", minWidth: 0,
    }}>
      <header style={{
        display:"flex", alignItems:"center", gap: 10,
        padding: "12px 16px", borderBottom: `1px solid ${TOKENS.rule}`,
        background: TOKENS.paper,
      }}>
        <code style={{
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
          background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
          padding: "2px 7px", borderRadius: 3,
        }}>{rxn.entry_id}</code>
        <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink, flex: 1, minWidth: 0, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{rxn.title}</span>
        <button onClick={onClose} style={{
          width: 24, height: 24, padding: 0, background: "transparent",
          border: `1px solid ${TOKENS.rule}`, borderRadius: 4, cursor: "pointer",
          fontFamily: FONT_SANS, fontWeight: 700, color: TOKENS.muted, fontSize: 13,
        }}>×</button>
      </header>

      <div style={{ overflow: "auto", padding: 16, display:"flex", flexDirection:"column", gap: 16 }}>
        <div style={{
          background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
          overflow: "hidden",
        }}>
          <EquationRow rxn={rxn} dens={dens}/>
        </div>

        <DrawerSection label="Reactants">
          {(rxn.reactants || []).map((m, j) => <EditableSMILES key={j} mol={m} kind="reactant"/>)}
        </DrawerSection>

        <DrawerSection label="Reagents">
          {(rxn.reagents || []).map((m, j) => <EditableSMILES key={j} mol={m} kind="reagent"/>)}
        </DrawerSection>

        <DrawerSection label="Conditions">
          <div style={{ display:"grid", gridTemplateColumns:"100px 1fr", gap: 8, fontFamily: FONT_SANS, fontSize: 13 }}>
            <Dt>solvent</Dt><EditableText value={rxn.conditions?.solvent}/>
            <Dt>temp</Dt><EditableText value={rxn.conditions?.temperature}/>
            <Dt>time</Dt><EditableText value={rxn.conditions?.time}/>
            <Dt>atm</Dt><EditableText value={rxn.conditions?.atmosphere}/>
          </div>
        </DrawerSection>

        <DrawerSection label="Products">
          {(rxn.products || []).map((m, j) => <EditableSMILES key={j} mol={m} kind="product"/>)}
        </DrawerSection>

        {rxn.notes && (
          <DrawerSection label="Notes">
            <p style={{ margin: 0, fontFamily: FONT_SERIF, fontStyle:"italic", color: TOKENS.ink2, fontSize: 13.5, lineHeight: 1.5 }}>{rxn.notes}</p>
          </DrawerSection>
        )}

        {insight && (
          <DrawerSection label="Rxn-INSIGHT">
            <dl style={{ margin: 0, display:"grid", gridTemplateColumns:"100px 1fr", rowGap: 6, columnGap: 10, fontFamily: FONT_SANS, fontSize: 12.5 }}>
              {insight.reaction_class && (<><Dt>Class</Dt><Dd>{insight.reaction_class}</Dd></>)}
              {insight.named_reaction && (<><Dt>Named</Dt><Dd italic>{insight.named_reaction}</Dd></>)}
              {insight.rxn_class_id && (<><Dt>Tag</Dt><Dd mono>{insight.rxn_class_id}</Dd></>)}
              {insight.functional_groups?.length > 0 && (<><Dt>Functional</Dt><Dd>{insight.functional_groups.join(", ")}</Dd></>)}
              {insight.byproducts?.length > 0 && (<><Dt>Byproducts</Dt><Dd mono>{insight.byproducts.join(", ")}</Dd></>)}
              {insight.template && (<><Dt>Template</Dt><Dd mono>{insight.template}</Dd></>)}
              {insight.scaffold && (<><Dt>Scaffold</Dt><Dd mono>{insight.scaffold}</Dd></>)}
              {insight.error && (<><Dt>Error</Dt><Dd>{insight.error}</Dd></>)}
            </dl>
          </DrawerSection>
        )}
      </div>

      <footer style={{
        display:"flex", alignItems:"center", gap: 8,
        padding: "10px 14px", borderTop: `1px solid ${TOKENS.rule}`,
        background: TOKENS.paper,
      }}>
        <Button small icon={verified ? <Icon.check/> : null} onClick={onToggleVerified}>
          {verified ? "Verified" : "Mark verified"}
        </Button>
        <div style={{ flex: 1 }}/>
        <Button small icon={<Icon.copy/>} onClick={() => {
          const rxnSmiles = buildRxnSmiles(rxn);
          if (rxnSmiles) navigator.clipboard?.writeText(rxnSmiles);
        }}>Copy RXN-SMILES</Button>
      </footer>
    </aside>
  );
}

function buildRxnSmiles(rxn) {
  const left  = (rxn.reactants || []).map(m => m.smiles).filter(s => s && !s.includes('*')).join('.');
  const right = (rxn.products  || []).map(m => m.smiles).filter(s => s && !s.includes('*')).join('.');
  if (!left || !right) return null;
  return `${left}>>${right}`;
}

function DrawerSection({ label, children }) {
  return (
    <section>
      <h4 style={{
        margin: "0 0 8px", fontFamily: FONT_SANS, fontSize: 11, fontWeight: 700,
        textTransform: "uppercase", letterSpacing: 0.6, color: TOKENS.muted,
      }}>{label}</h4>
      <div style={{ display:"flex", flexDirection:"column", gap: 6 }}>{children}</div>
    </section>
  );
}

function EditableSMILES({ mol, kind }) {
  const sub =
    kind === "product"  && mol.yield_pct != null ? `${mol.yield_pct}%` :
    kind === "product"  && mol.yield_note         ? mol.yield_note :
    kind === "reactant" && mol.equiv              ? mol.equiv :
    kind === "reagent"  && mol.loading            ? mol.loading : null;
  return (
    <div style={{
      display:"grid", gridTemplateColumns:"56px 1fr auto", gap: 8, alignItems:"center",
      padding: 8, background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
    }}>
      <div style={{ width: 56, height: 44, display:"grid", placeItems:"center" }}>
        <MolStructurePlaceholder smiles={mol.smiles} width={56} height={44}/>
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ display:"flex", alignItems:"baseline", gap: 6 }}>
          {mol.label && <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent, fontWeight: 700 }}>{mol.label}</span>}
          <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13, color: TOKENS.ink, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{mol.name || "—"}</span>
        </div>
        <code style={{
          display:"block", fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted,
          padding: "2px 0", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis",
        }}>{mol.smiles || "—"}</code>
      </div>
      <div style={{ display:"flex", alignItems:"center", gap: 4 }}>
        {sub && <span style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.muted, fontVariantNumeric:"tabular-nums" }}>{sub}</span>}
        <CopyButton text={mol.smiles}/>
      </div>
    </div>
  );
}

function EditableText({ value }) {
  return (
    <span style={{
      fontFamily: FONT_SANS, fontSize: 13, color: TOKENS.ink,
      padding: "3px 6px", borderRadius: 4,
      border: `1px dashed transparent`, cursor: "text",
    }} onMouseOver={(e) => e.currentTarget.style.border = `1px dashed ${TOKENS.rule}`}
       onMouseOut={(e) => e.currentTarget.style.border = `1px dashed transparent`}>
      {value || <em style={{ color: TOKENS.muted }}>—</em>}
    </span>
  );
}

function Dt({ children }) { return <dt style={{ color: TOKENS.muted, fontFamily: FONT_SANS, fontSize: 12 }}>{children}</dt>; }
function Dd({ children, mono, italic }) {
  return <dd style={{ margin: 0, color: TOKENS.ink, fontFamily: mono ? FONT_MONO : FONT_SANS, fontStyle: italic ? "italic" : "normal", fontSize: 12.5, fontVariantNumeric:"tabular-nums" }}>{children}</dd>;
}

// ── Sign-in (kept as a UI affordance, not wired to real auth) ──────────
function SignInScreen({ user, onSignIn }) {
  const [email, setEmail] = useState(user?.email || "");
  const [submitting, setSubmitting] = useState(false);
  function submit(e) {
    e?.preventDefault?.();
    if (!email) return;
    setSubmitting(true);
    setTimeout(() => { onSignIn(); setSubmitting(false); }, 400);
  }
  return (
    <div style={{
      position: "absolute", inset: 0, zIndex: 200,
      background: TOKENS.paper, display: "grid", placeItems: "center", padding: 24,
    }}>
      <div style={{
        width: 380, background: "white",
        border: `1px solid ${TOKENS.rule}`, borderRadius: 10, padding: 28,
        boxShadow: "0 12px 40px rgba(15,26,36,0.08)",
        display: "flex", flexDirection: "column", gap: 18,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: TOKENS.ink, color: "white",
            display: "grid", placeItems: "center",
            fontFamily: FONT_SERIF, fontStyle: "italic", fontWeight: 700, fontSize: 18,
          }}>R</div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 15, color: TOKENS.ink }}>Rxn-EXTRACTION</span>
            <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>Sign in to continue</span>
          </div>
        </div>

        <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontFamily: FONT_SANS, fontSize: 11, fontWeight: 700, color: TOKENS.muted, textTransform: "uppercase", letterSpacing: 0.6 }}>Email</span>
            <input
              type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              autoFocus
              placeholder="you@lab.org"
              style={{
                padding: "9px 11px",
                fontFamily: FONT_SANS, fontSize: 13.5,
                border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
                background: "white", outline: "none", color: TOKENS.ink,
              }}
            />
          </label>

          <button type="submit" disabled={!email || submitting} style={{
            padding: "10px 14px",
            background: TOKENS.accent, color: "white",
            border: `1px solid ${TOKENS.accent}`,
            borderRadius: 6, cursor: email && !submitting ? "pointer" : "not-allowed",
            fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13.5,
            opacity: !email ? 0.5 : 1,
          }}>
            {submitting ? "Signing in…" : "Continue with email"}
          </button>
        </form>

        <p style={{
          margin: 0, fontFamily: FONT_SANS, fontSize: 11.5, color: TOKENS.muted,
          textAlign: "center", lineHeight: 1.5,
        }}>Local demo — any email will sign in.</p>
      </div>
    </div>
  );
}

// ── Figure lightbox ─────────────────────────────────────────────────────
function FigureLightbox({ src, caption, onClose }) {
  const [zoom, setZoom] = useState(1);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "+" || e.key === "=") setZoom(z => Math.min(z * 1.4, 6));
      else if (e.key === "-")                  setZoom(z => Math.max(z / 1.4, 0.5));
      else if (e.key === "0")                  setZoom(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div onClick={onClose} style={{
      position: "absolute", inset: 0, zIndex: 100,
      background: "rgba(15,26,36,0.78)",
      display: "grid", gridTemplateRows: "auto 1fr auto", gap: 0,
      padding: 0,
      cursor: "zoom-out",
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "10px 18px", color: "white",
        background: "rgba(0,0,0,0.25)",
        cursor: "default",
      }}>
        <span style={{ fontFamily: FONT_MONO, fontSize: 12 }}>source figure</span>
        <div style={{ flex: 1 }}/>
        <div style={{ display:"flex", alignItems:"center", gap: 4, background:"rgba(255,255,255,0.1)", borderRadius: 6, padding: 2 }}>
          <ZoomBtn onClick={() => setZoom(z => Math.max(z / 1.4, 0.5))}>−</ZoomBtn>
          <button onClick={() => setZoom(1)} style={{
            background: "transparent", border: "none", color: "white",
            fontFamily: FONT_MONO, fontSize: 12, padding: "4px 10px", cursor: "pointer", minWidth: 56,
          }}>{Math.round(zoom * 100)}%</button>
          <ZoomBtn onClick={() => setZoom(z => Math.min(z * 1.4, 6))}>+</ZoomBtn>
        </div>
        <button onClick={onClose} style={{
          background: "transparent", border: "1px solid rgba(255,255,255,0.3)",
          color: "white", borderRadius: 4, padding: "4px 10px", cursor: "pointer",
          fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
        }}>Close · esc</button>
      </div>

      <div onClick={onClose} style={{
        overflow: "auto", display: "grid", placeItems: zoom <= 1 ? "center" : "start center",
        padding: 20,
      }}>
        <img src={src} alt="source figure" onClick={(e) => { e.stopPropagation(); setZoom(z => z >= 2 ? 1 : 2); }}
             style={{
               display: "block",
               width: zoom <= 1 ? "auto" : `${zoom * 100}%`,
               maxWidth: zoom <= 1 ? "100%" : "none",
               maxHeight: zoom <= 1 ? "100%" : "none",
               background: "white", borderRadius: 6, padding: 12,
               cursor: zoom >= 2 ? "zoom-out" : "zoom-in",
               boxShadow: "0 12px 40px rgba(0,0,0,0.4)",
             }}/>
      </div>

      <div onClick={(e) => e.stopPropagation()} style={{
        padding: "10px 24px", textAlign: "center",
        background: "rgba(0,0,0,0.25)", cursor: "default",
      }}>
        <p style={{ margin: 0, fontFamily: FONT_SERIF, fontStyle:"italic", fontSize: 14, color: "rgba(255,255,255,0.9)" }}>{caption || ""}</p>
      </div>
    </div>
  );
}
function ZoomBtn({ children, onClick, title }) {
  return (
    <button onClick={onClick} title={title} style={{
      background: "transparent", border: "none", color: "white",
      width: 28, height: 28, padding: 0, cursor: "pointer",
      fontFamily: FONT_SANS, fontSize: 16, fontWeight: 600,
      display: "grid", placeItems: "center",
      borderRadius: 4,
    }}>{children}</button>
  );
}

// ── Keyboard cheat sheet ────────────────────────────────────────────────
function KeyboardCheatSheet({ onClose }) {
  const rows = [
    ["/",       "Focus filter"],
    ["enter",   "Open detail drawer"],
    ["v",       "Toggle verified"],
    ["x",       "Toggle selected"],
    ["esc",     "Close drawer / clear selection"],
  ];
  return (
    <div onClick={onClose} style={{
      position: "absolute", inset: 0, zIndex: 90,
      background: "rgba(15,26,36,0.30)",
      display: "grid", placeItems: "start center", padding: "80px 20px",
      cursor: "default",
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: "white", border: `1px solid ${TOKENS.rule}`,
        borderRadius: 8, padding: 18, minWidth: 320,
        boxShadow: "0 16px 40px rgba(15,26,36,0.2)",
      }}>
        <div style={{ display:"flex", alignItems:"baseline", justifyContent:"space-between", marginBottom: 12 }}>
          <h3 style={{ margin: 0, fontFamily: FONT_SANS, fontSize: 14, fontWeight: 700, color: TOKENS.ink }}>Keyboard shortcuts</h3>
          <button onClick={onClose} style={{
            background: "transparent", border: "none", color: TOKENS.muted, cursor: "pointer",
            fontSize: 18, padding: 0,
          }}>×</button>
        </div>
        <dl style={{ margin: 0, display:"grid", gridTemplateColumns:"auto 1fr", rowGap: 6, columnGap: 14 }}>
          {rows.map(([k, v]) => (
            <Fragment key={k}>
              <dt><Kbd>{k}</Kbd></dt>
              <dd style={{ margin: 0, fontFamily: FONT_SANS, fontSize: 13, color: TOKENS.ink2 }}>{v}</dd>
            </Fragment>
          ))}
        </dl>
      </div>
    </div>
  );
}
function Kbd({ children }) {
  return (
    <kbd style={{
      fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2,
      background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`,
      padding: "2px 7px", borderRadius: 4, fontWeight: 600,
      boxShadow: `inset 0 -1px 0 ${TOKENS.rule2}`,
    }}>{children}</kbd>
  );
}

// ── Empty / landing / busy / error states ───────────────────────────────
function EmptyState({ query, onClear }) {
  return (
    <div style={{
      background: "white", border: `1px dashed ${TOKENS.rule}`, borderRadius: 8,
      padding: 32, display:"grid", placeItems:"center", gap: 8,
      color: TOKENS.muted, fontFamily: FONT_SANS, fontSize: 13,
    }}>
      <span>No reactions match <code style={{ fontFamily: FONT_MONO, color: TOKENS.ink2, background: TOKENS.paper, padding: "1px 6px", borderRadius: 3 }}>{query}</code>.</span>
      <button onClick={onClear} style={{
        background: "transparent", border: "none", color: TOKENS.accent,
        fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13, cursor: "pointer",
      }}>Clear filter</button>
    </div>
  );
}

function Landing({ onUploadFile }) {
  const fileRef = useRef(null);
  const [drag, setDrag] = useState(false);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault(); setDrag(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onUploadFile?.(f);
      }}
      style={{
        background: "white", border: `2px dashed ${drag ? TOKENS.accent : TOKENS.rule}`,
        borderRadius: 12, padding: "56px 24px",
        display: "grid", placeItems: "center", gap: 14,
        color: TOKENS.ink2, fontFamily: FONT_SANS,
        minHeight: 320,
      }}
    >
      <div style={{
        width: 56, height: 56, borderRadius: 999,
        background: TOKENS.paper, color: TOKENS.accent,
        display:"grid", placeItems:"center",
        border: `1px solid ${TOKENS.rule}`,
      }}>
        <Icon.upload/>
      </div>
      <h2 style={{ margin: 0, fontFamily: FONT_SANS, fontSize: 18, fontWeight: 700, color: TOKENS.ink }}>
        Upload a chemistry figure or PDF
      </h2>
      <p style={{ margin: 0, fontFamily: FONT_SERIF, fontStyle: "italic", fontSize: 14, color: TOKENS.ink2, textAlign: "center", maxWidth: 520, lineHeight: 1.5 }}>
        Drop a PNG / JPG of a reaction scheme to extract reactions directly, or drop a full PDF — VisualHeist will detect every figure and the agent runs on each one.
      </p>
      <Button primary icon={<Icon.upload/>} onClick={() => fileRef.current?.click()}>Choose file</Button>
      <input ref={fileRef} type="file" accept="image/*,application/pdf,.pdf" style={{ display: "none" }}
             onChange={(e) => {
               const f = e.target.files?.[0];
               if (f) onUploadFile?.(f);
               e.target.value = "";
             }}/>
    </div>
  );
}

function BusyState({ phase, agentLog }) {
  return (
    <div style={{
      background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
      padding: 32, display:"flex", flexDirection: "column", gap: 12,
      color: TOKENS.ink2, fontFamily: FONT_SANS, fontSize: 13,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{
          width: 12, height: 12, borderRadius: 999,
          background: TOKENS.warn,
          animation: "rxn-pulse 1.2s ease-in-out infinite",
        }}/>
        <strong style={{ color: TOKENS.ink, fontSize: 15 }}>
          {phase === 'uploading' ? 'Uploading figure…' : 'Agent is extracting…'}
        </strong>
      </div>
      <p style={{ margin: 0, color: TOKENS.muted, fontFamily: FONT_SERIF, fontStyle: "italic" }}>
        {agentLog.length === 0
          ? "Waiting for the first tool call…"
          : `${agentLog.length} tool call${agentLog.length === 1 ? '' : 's'} so far. Live updates in the Agent activity panel below.`}
      </p>
    </div>
  );
}

// ── My runs modal (folder picker + drag-to-organize) ──────────────────
// Special sidebar entries:
//   "__all__"      → every run
//   "__unfiled__"  → runs with folder_id == null
//   <folder-uuid>  → runs whose folder_id matches
function MyRunsModal({ currentRunId, onPick, onClose }) {
  const [items, setItems]       = useState(null);
  const [folders, setFolders]   = useState(null);
  const [active, setActive]     = useState("__all__");
  const [err, setErr]           = useState(null);
  const [dragRun, setDragRun]   = useState(null);    // run_id while dragging
  const [creating, setCreating] = useState(false);
  const [newName, setNewName]   = useState("");
  const [renaming, setRenaming] = useState(null);    // folder_id being renamed
  const [renameText, setRenameText] = useState("");
  const [busy, setBusy]         = useState(false);

  async function refresh() {
    try {
      const [r, f] = await Promise.all([fetchRuns(), fetchFolders()]);
      setItems(r);
      setFolders(f);
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

  const filtered = useMemo(() => {
    if (!items) return [];
    if (active === "__all__")     return items;
    if (active === "__unfiled__") return items.filter(it => !it.folder_id);
    return items.filter(it => it.folder_id === active);
  }, [items, active]);

  const folderCounts = useMemo(() => {
    const counts = { __all__: items?.length || 0, __unfiled__: 0 };
    for (const f of (folders || [])) counts[f.id] = 0;
    for (const it of (items || [])) {
      if (it.folder_id && counts[it.folder_id] != null) counts[it.folder_id]++;
      if (!it.folder_id) counts.__unfiled__++;
    }
    return counts;
  }, [items, folders]);

  async function doCreate(name) {
    const v = (name || "").trim();
    if (!v) { setCreating(false); setNewName(""); return; }
    setBusy(true);
    try {
      const f = await createFolder(v);
      setCreating(false); setNewName("");
      setActive(f.id);
      await refresh();
    } catch (e) {
      alert(`Couldn't create folder: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function doRename(folderId, name) {
    const v = (name || "").trim();
    if (!v) { setRenaming(null); return; }
    setBusy(true);
    try {
      await renameFolder(folderId, v);
      setRenaming(null);
      await refresh();
    } catch (e) {
      alert(`Couldn't rename folder: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function doDelete(folderId, name) {
    if (!confirm(`Delete folder "${name}"? It must be empty.`)) return;
    setBusy(true);
    try {
      await deleteFolder(folderId);
      if (active === folderId) setActive("__all__");
      await refresh();
    } catch (e) {
      alert(`Couldn't delete folder: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function doMove(runId, folderId) {
    setBusy(true);
    try {
      await moveRun(runId, folderId);
      await refresh();
    } catch (e) {
      alert(`Couldn't move run: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div onClick={onClose} style={{
      position: "absolute", inset: 0, zIndex: 120,
      background: "rgba(15,26,36,0.40)",
      display: "grid", placeItems: "center", padding: 24,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: "min(1040px, 100%)", height: "min(680px, 85vh)",
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
            My runs
          </h3>
          <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>
            {items == null ? "loading…" : `${filtered.length} of ${items.length}`}
          </span>
          <div style={{ flex: 1 }}/>
          <button onClick={onClose} style={{
            width: 26, height: 26, padding: 0, background: "transparent",
            border: `1px solid ${TOKENS.rule}`, borderRadius: 4, cursor: "pointer",
            fontFamily: FONT_SANS, fontWeight: 700, color: TOKENS.muted, fontSize: 14,
          }}>×</button>
        </header>

        <div style={{
          display: "grid", gridTemplateColumns: "220px 1fr",
          minHeight: 0,
        }}>
          {/* Left rail: folders */}
          <aside style={{
            borderRight: `1px solid ${TOKENS.rule}`, background: TOKENS.paper,
            display: "flex", flexDirection: "column", minHeight: 0,
          }}>
            <nav style={{ flex: 1, overflowY: "auto", padding: 6 }}>
              <FolderItem
                label="All runs"
                count={folderCounts.__all__}
                active={active === "__all__"}
                onClick={() => setActive("__all__")}
              />
              <FolderItem
                label="Unfiled"
                count={folderCounts.__unfiled__}
                active={active === "__unfiled__"}
                onClick={() => setActive("__unfiled__")}
                onDrop={(runId) => doMove(runId, null)}
                dragRun={dragRun}
              />
              <div style={{ height: 1, background: TOKENS.rule2, margin: "6px 4px" }}/>
              {(folders || []).map(f => (
                <FolderItem
                  key={f.id}
                  folder={f}
                  count={folderCounts[f.id] || 0}
                  active={active === f.id}
                  onClick={() => setActive(f.id)}
                  onDrop={(runId) => doMove(runId, f.id)}
                  dragRun={dragRun}
                  renaming={renaming === f.id}
                  onStartRename={() => { setRenaming(f.id); setRenameText(f.name); }}
                  onCommitRename={(v) => doRename(f.id, v)}
                  renameText={renameText}
                  onRenameTextChange={setRenameText}
                  onDelete={() => doDelete(f.id, f.name)}
                />
              ))}
              {creating ? (
                <input
                  autoFocus
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onBlur={() => doCreate(newName)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") doCreate(newName);
                    if (e.key === "Escape") { setCreating(false); setNewName(""); }
                  }}
                  placeholder="Folder name"
                  style={{
                    width: "calc(100% - 8px)", margin: "4px",
                    padding: "6px 9px", fontFamily: FONT_SANS, fontSize: 13,
                    border: `1px solid ${TOKENS.accent}`, borderRadius: 4,
                    background: "white", color: TOKENS.ink, outline: "none",
                  }}/>
              ) : (
                <button onClick={() => setCreating(true)} style={{
                  width: "calc(100% - 8px)", margin: "4px",
                  padding: "6px 9px",
                  textAlign: "left", background: "transparent",
                  border: `1px dashed ${TOKENS.rule}`, borderRadius: 4,
                  color: TOKENS.muted, cursor: "pointer",
                  fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
                }}>+ New folder</button>
              )}
            </nav>
            <div style={{
              padding: "8px 12px", borderTop: `1px solid ${TOKENS.rule}`,
              fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted, lineHeight: 1.4,
            }}>Drag a run onto a folder to file it.</div>
          </aside>

          {/* Right: runs in active folder */}
          <div style={{ overflowY: "auto", padding: 12, minWidth: 0 }}>
            {err && <div style={{ color: TOKENS.err, fontFamily: FONT_SANS, padding: 10 }}>Couldn't load runs: {err}</div>}
            {!err && items == null && (
              <div style={{ padding: 24, textAlign:"center", color: TOKENS.muted, fontFamily: FONT_SANS, fontSize: 13 }}>Loading…</div>
            )}
            {!err && items != null && filtered.length === 0 && (
              <div style={{ padding: 36, textAlign:"center", color: TOKENS.muted, fontFamily: FONT_SERIF, fontStyle: "italic" }}>
                {active === "__all__"     ? "No runs yet. Upload a figure to get started." :
                 active === "__unfiled__" ? "Every run is in a folder." :
                                            "Drop a run here to file it in this folder."}
              </div>
            )}
            {!err && items != null && filtered.length > 0 && (
              <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 8 }}>
                {filtered.map(item => (
                  <RunCard
                    key={item.run_id}
                    item={item}
                    active={item.run_id === currentRunId}
                    folders={folders || []}
                    onClick={() => onPick(item)}
                    onDragStart={() => setDragRun(item.run_id)}
                    onDragEnd={() => setDragRun(null)}
                    onMoveTo={(folderId) => doMove(item.run_id, folderId)}
                  />
                ))}
              </ul>
            )}
          </div>
        </div>

        <footer style={{
          padding: "10px 18px", borderTop: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper,
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted,
          display: "flex", justifyContent: "space-between",
        }}>
          <span>{busy ? "saving…" : ""}</span>
          <span>esc to close</span>
        </footer>
      </div>
    </div>
  );
}

function FolderItem({
  folder, label, count, active, onClick,
  onDrop, dragRun,
  renaming, renameText, onRenameTextChange, onStartRename, onCommitRename,
  onDelete,
}) {
  const [hover, setHover] = useState(false);
  const [over, setOver]   = useState(false);
  const isDropTarget = !!onDrop && dragRun;

  return (
    <div
      onClick={onClick}
      onDoubleClick={folder && onStartRename ? onStartRename : undefined}
      onDragOver={isDropTarget ? (e) => { e.preventDefault(); setOver(true); } : undefined}
      onDragLeave={isDropTarget ? () => setOver(false) : undefined}
      onDrop={isDropTarget ? (e) => {
        e.preventDefault(); setOver(false);
        const id = e.dataTransfer.getData("text/run-id");
        if (id) onDrop(id);
      } : undefined}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "flex", alignItems: "center", gap: 6,
        padding: "5px 8px", margin: "1px 2px", cursor: "pointer",
        borderRadius: 4,
        background: over ? "#E8F1F8" : (active ? "white" : "transparent"),
        border: `1px solid ${over ? TOKENS.accent : (active ? TOKENS.rule : "transparent")}`,
        fontFamily: FONT_SANS, fontSize: 13,
        color: active ? TOKENS.ink : TOKENS.ink2,
        fontWeight: active ? 600 : 500,
      }}>
      <span style={{ width: 14, color: TOKENS.muted, flex: "0 0 auto" }}>
        {folder ? "📁" : (label === "All runs" ? "▦" : "·")}
      </span>
      {renaming ? (
        <input
          autoFocus
          value={renameText}
          onChange={(e) => onRenameTextChange(e.target.value)}
          onBlur={() => onCommitRename(renameText)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onCommitRename(renameText);
            if (e.key === "Escape") onCommitRename(folder.name);
          }}
          onClick={(e) => e.stopPropagation()}
          style={{
            flex: 1, minWidth: 0, padding: "2px 4px",
            fontFamily: FONT_SANS, fontSize: 13,
            border: `1px solid ${TOKENS.accent}`, borderRadius: 3,
            outline: "none", color: TOKENS.ink, background: "white",
          }}/>
      ) : (
        <span style={{
          flex: 1, minWidth: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
        }}>{folder ? folder.name : label}</span>
      )}
      <span style={{
        fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted,
        flex: "0 0 auto",
      }}>{count}</span>
      {folder && hover && !renaming && (
        <button onClick={(e) => { e.stopPropagation(); onDelete?.(); }}
          title="Delete folder (must be empty)"
          style={{
            background: "transparent", border: "none", color: TOKENS.muted,
            cursor: "pointer", padding: "0 2px", fontSize: 13,
          }}>×</button>
      )}
    </div>
  );
}

function RunCard({ item, active, folders, onClick, onDragStart, onDragEnd, onMoveTo }) {
  const thumb = item.input_name ? fileUrl(item.run_id, item.input_name) : null;
  const when  = formatWhen(item.created);
  const [menuOpen, setMenuOpen] = useState(false);
  const folderName = (folders || []).find(f => f.id === item.folder_id)?.name;

  return (
    <li
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData("text/run-id", item.run_id);
        e.dataTransfer.effectAllowed = "move";
        onDragStart?.();
      }}
      onDragEnd={() => onDragEnd?.()}
    >
      <div style={{ position: "relative" }}>
        <button onClick={(e) => { if (menuOpen) return; onClick(); }}
                disabled={!item.has_extraction} style={{
          width: "100%", textAlign: "left", cursor: item.has_extraction ? "pointer" : "not-allowed",
          display: "grid", gridTemplateColumns: "84px 1fr auto", gap: 14,
          padding: 10, alignItems: "center",
          background: active ? "#EEF6FF" : "white",
          border: `1px solid ${active ? TOKENS.accent + "55" : TOKENS.rule}`,
          borderRadius: 8,
          opacity: item.has_extraction ? 1 : 0.55,
        }}>
          <div style={{
            width: 84, height: 60, background: TOKENS.paper,
            border: `1px solid ${TOKENS.rule}`, borderRadius: 4,
            display: "grid", placeItems: "center", overflow: "hidden",
          }}>
            {thumb
              ? <img src={thumb} alt="" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" }}/>
              : <span style={{ color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 10 }}>no image</span>}
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ display:"flex", alignItems:"baseline", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
              <code style={{
                fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
                background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
                padding: "1px 6px", borderRadius: 3,
              }}>{item.run_id}</code>
              <span style={{ fontFamily: FONT_SANS, fontSize: 11.5, color: TOKENS.muted }}>{when}</span>
              {folderName && (
                <span style={{ fontFamily: FONT_SANS, fontSize: 10, fontWeight: 700,
                                color: TOKENS.ink2, background: TOKENS.paper,
                                border: `1px solid ${TOKENS.rule}`,
                                padding: "1px 6px", borderRadius: 3 }}>📁 {folderName}</span>
              )}
              {!item.has_extraction && (
                <span style={{ fontFamily: FONT_SANS, fontSize: 10, fontWeight: 700,
                                color: TOKENS.warn, background: TOKENS.warn2,
                                padding: "1px 6px", borderRadius: 3 }}>incomplete</span>
              )}
              {item.has_insight && (
                <span style={{ fontFamily: FONT_SANS, fontSize: 10, fontWeight: 700,
                                color: TOKENS.pos, background: TOKENS.pos2,
                                padding: "1px 6px", borderRadius: 3 }}>INSIGHT</span>
              )}
              {active && (
                <span style={{ fontFamily: FONT_SANS, fontSize: 10, fontWeight: 700,
                                color: TOKENS.accent, background: "#E8F1F8",
                                padding: "1px 6px", borderRadius: 3 }}>OPEN</span>
              )}
            </div>
            <div style={{
              fontFamily: FONT_SERIF, fontStyle: "italic", fontSize: 13,
              color: TOKENS.ink2, lineHeight: 1.35,
              display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}>
              {item.figure_caption || (item.has_extraction ? "(no caption)" : "—")}
            </div>
          </div>
          <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap: 4, fontFamily: FONT_SANS, fontSize: 12, color: TOKENS.ink2 }}>
            <span style={{ fontWeight: 700, fontSize: 18, color: TOKENS.ink, fontVariantNumeric: "tabular-nums" }}>
              {item.n_reactions ?? "—"}
            </span>
            <span style={{ fontSize: 10.5, color: TOKENS.muted, textTransform: "uppercase", letterSpacing: 0.5 }}>
              {item.n_reactions === 1 ? "reaction" : "reactions"}
            </span>
          </div>
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); setMenuOpen(o => !o); }}
          title="Move to folder"
          style={{
            position: "absolute", top: 6, right: 6,
            background: "white", border: `1px solid ${TOKENS.rule}`,
            color: TOKENS.muted, borderRadius: 4, padding: "1px 6px",
            fontFamily: FONT_SANS, fontWeight: 700, fontSize: 12,
            cursor: "pointer",
          }}>⋯</button>
        {menuOpen && (
          <div style={{
            position: "absolute", top: 32, right: 6, zIndex: 5,
            background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
            boxShadow: "0 4px 16px rgba(15,26,36,0.12)",
            minWidth: 180, padding: 4,
          }}>
            <MoveMenuItem onClick={() => { setMenuOpen(false); onMoveTo(null); }}
                          active={!item.folder_id}>Unfiled</MoveMenuItem>
            {(folders || []).length > 0 && (
              <div style={{ height: 1, background: TOKENS.rule2, margin: "2px 0" }}/>
            )}
            {(folders || []).map(f => (
              <MoveMenuItem key={f.id} onClick={() => { setMenuOpen(false); onMoveTo(f.id); }}
                            active={item.folder_id === f.id}>📁 {f.name}</MoveMenuItem>
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

function MoveMenuItem({ children, onClick, active }) {
  return (
    <button onClick={onClick} style={{
      display: "block", width: "100%", textAlign: "left",
      padding: "5px 9px", background: active ? TOKENS.paper : "transparent",
      border: "none", borderRadius: 3, cursor: "pointer",
      fontFamily: FONT_SANS, fontSize: 12.5, color: TOKENS.ink2,
      fontWeight: active ? 700 : 500,
    }}>{children}</button>
  );
}

function formatWhen(ts) {
  if (!ts) return "";
  const ms = ts * 1000;
  const diff = Date.now() - ms;
  const min = Math.floor(diff / 60000);
  if (min < 1)         return "just now";
  if (min < 60)        return `${min} min ago`;
  const hr = Math.floor(min / 60);
  if (hr  < 24)        return `${hr} h ago`;
  const days = Math.floor(hr / 24);
  if (days < 7)        return `${days} d ago`;
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function ErrorBanner({ message }) {
  return (
    <div style={{
      background: TOKENS.err2, border: `1px solid ${TOKENS.err}55`, borderRadius: 6,
      padding: "12px 14px", color: TOKENS.err,
      fontFamily: FONT_SANS, fontSize: 13, fontWeight: 600,
      marginBottom: 16,
      display: "flex", alignItems: "center", gap: 8,
    }}>
      <Icon.warn/>
      <span style={{ flex: 1 }}>{message || "Something went wrong."}</span>
    </div>
  );
}
