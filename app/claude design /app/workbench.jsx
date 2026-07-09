// Direction A — Workbench (v2)
// Adds: reactions toolbar (search, sort, bulk actions), per-row hover actions,
// right-side detail drawer, figure lightbox, keyboard-shortcut hint.

function WorkbenchApp({ tweaks }) {
  const data =
    tweaks.dataset === "small"  ? MOCK_SMALL :
    tweaks.dataset === "large"  ? MOCK_LARGE : MOCK_MEDIUM;
  const dens = useDensity(tweaks.density);

  const [modelId, setModelId]     = useState(MOCK_MODELS[0].id);
  const [layout, setLayout]       = useState(tweaks.layout);
  useEffect(() => setLayout(tweaks.layout), [tweaks.layout]);

  const [logOpen, setLogOpen]     = useState(false);
  const [detailRxn, setDetailRxn] = useState(null);    // index of row showing in drawer
  const [lightboxOpen, setLightbox] = useState(false);
  const [showKbd, setShowKbd]     = useState(false);
  const [signedIn, setSignedIn]   = useState(true);

  const [query, setQuery]         = useState("");
  const [sort, setSort]           = useState({ key: null, dir: "asc" });
  const [verified, setVerified]   = useState(() => initialVerified(data));
  const [selected, setSelected]   = useState(new Set());

  useEffect(() => {
    setVerified(initialVerified(data));
    setSelected(new Set());
    setDetailRxn(null);
    setSort({ key: null, dir: "asc" });
  }, [tweaks.dataset]);

  const reactions = data.reactions;

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
    const out = {};
    for (const r of reactions) if (MOCK_INSIGHT[r.entry_id]) out[r.entry_id] = MOCK_INSIGHT[r.entry_id];
    return out;
  }, [reactions]);

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
        modelId={modelId} onModelChange={setModelId}
        reactions={reactions} verified={verified}
        onShowKbd={() => setShowKbd(v => !v)} kbdActive={showKbd}
        onSignOut={() => setSignedIn(false)}
      />

      <div style={{
        display: "grid", gridTemplateColumns: `320px 1fr ${detailRxn != null ? "440px" : "0px"}`,
        gap: 0, minHeight: 0, transition: "grid-template-columns 0.18s ease",
      }}>
        <LeftRail
          data={data} reactions={reactions}
          totalProducts={totalProducts} avgYield={avgYield}
          verifiedCount={verified.size}
          onOpenLightbox={() => setLightbox(true)}
        />

        <main style={{ overflow: "auto", padding: "16px 22px 24px", minWidth: 0 }}>
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
          {layout === "card"     && <ReactionCards     rows={filtered} dens={dens} onOpenDetail={setDetailRxn}/>}
          {layout === "equation" && <ReactionEquations rows={filtered} dens={dens} onOpenDetail={setDetailRxn}/>}

          {filtered.length === 0 && (
            <EmptyState query={query} onClear={() => setQuery("")}/>
          )}

          <div style={{ marginTop: 22 }}>
            <SectionLabel right={<span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>via rxn-insight v0.2.1</span>}>
              Enrichment
            </SectionLabel>
            <StructuredInsightPanel analyses={insights} dens={dens}/>
          </div>
        </main>

        {detailRxn != null && (
          <DetailDrawer
            rxn={reactions[detailRxn]}
            insight={MOCK_INSIGHT[reactions[detailRxn]?.entry_id]}
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

      <ActivityDrawer log={MOCK_AGENT_LOG} open={logOpen} onToggle={() => setLogOpen(o => !o)} visible={tweaks.showLog}/>

      {lightboxOpen && <FigureLightbox src={data.source_image} caption={data.figure_caption} onClose={() => setLightbox(false)}/>}
      {showKbd      && <KeyboardCheatSheet onClose={() => setShowKbd(false)}/>}
      {!signedIn    && <SignInScreen user={MOCK_USER} onSignIn={() => setSignedIn(true)}/>}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────
function initialVerified(data) {
  return new Set(data.reactions.map((r,i) => r.verified ? i : null).filter(v => v !== null));
}
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
function TopBar({ modelId, onModelChange, reactions, verified, onShowKbd, kbdActive, onSignOut }) {
  return (
    <header style={{
      display: "flex", alignItems: "center", gap: 14,
      padding: "10px 18px",
      borderBottom: `1px solid ${TOKENS.rule}`,
      background: "white",
    }}>
      <div style={{ display:"flex", alignItems:"center", gap: 10 }}>
        <div style={{
          width: 26, height: 26, borderRadius: 6,
          background: TOKENS.ink, color: "white",
          display:"grid", placeItems:"center",
          fontFamily: FONT_SERIF, fontStyle: "italic", fontWeight: 700, fontSize: 16,
        }}>R</div>
        <div style={{ display:"flex", flexDirection:"column", gap: 0 }}>
          <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink, letterSpacing: 0.1 }}>Rxn-EXTRACTION</span>
          <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>run · 13415d0010f8 · 2 min ago</span>
        </div>
      </div>

      <div style={{ flex: 1 }}/>

      <div style={{
        display:"inline-flex", alignItems:"center", gap: 6,
        padding: "4px 10px", borderRadius: 999,
        background: TOKENS.pos2, border: `1px solid ${TOKENS.pos}33`,
        color: TOKENS.pos, fontFamily: FONT_SANS, fontSize: 11.5, fontWeight: 600,
      }}>
        <span style={{ width: 7, height: 7, borderRadius:999, background: TOKENS.pos }}/>
        {reactions.length} reactions · {verified.size} verified
      </div>

      <ModelPickerCard models={MOCK_MODELS} value={modelId} onChange={onModelChange} compact/>

      <button onClick={onShowKbd} title="Keyboard shortcuts (?)" style={{
        width: 28, height: 28, padding: 0,
        background: kbdActive ? TOKENS.ink : "white",
        color: kbdActive ? "white" : TOKENS.ink2,
        border: `1px solid ${kbdActive ? TOKENS.ink : TOKENS.rule}`,
        borderRadius: 6, cursor: "pointer",
        fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14,
      }}>?</button>

      <Button icon={<Icon.upload/>}>New figure</Button>
      <Button primary icon={<Icon.spark/>}>Re-run with INSIGHT</Button>

      <span style={{ width: 1, height: 22, background: TOKENS.rule, margin: "0 2px" }}/>
      <UserMenu onSignOut={onSignOut}/>
    </header>
  );
}

// ── User menu (avatar + popover) ────────────────────────────────────────
const MOCK_USER = {
  name:    "Mira Tanaka",
  email:   "mira.tanaka@reactgrid.bio",
  role:    "Data engineer",
  org:     "ReactGrid · Corpus team",
  initials:"MT",
  tone:    "#7A4DA8",
  runs_today: 14,
  quota:   { used: 142, total: 500 },
};

function UserMenu({ onSignOut }) {
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

          <div style={{ padding: "10px 14px", display:"flex", flexDirection:"column", gap: 8, borderBottom: `1px solid ${TOKENS.rule2}` }}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap: 8 }}>
              <span style={{ fontFamily: FONT_SANS, fontSize: 11.5, color: TOKENS.muted, fontWeight: 600, textTransform:"uppercase", letterSpacing: 0.5 }}>{u.role}</span>
              <span style={{ fontFamily: FONT_SANS, fontSize: 11.5, color: TOKENS.ink2 }}>{u.org}</span>
            </div>
            <div>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom: 4 }}>
                <span style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.muted }}>Monthly quota</span>
                <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2 }}>{u.quota.used} / {u.quota.total}</span>
              </div>
              <div style={{ height: 6, background: TOKENS.paper, borderRadius: 999, overflow:"hidden" }}>
                <div style={{ width: `${pct}%`, height: "100%", background: pct < 75 ? TOKENS.pos : pct < 90 ? TOKENS.warn : TOKENS.err }}/>
              </div>
            </div>
            <div style={{ display:"flex", justifyContent:"space-between", fontFamily: FONT_SANS, fontSize: 12, color: TOKENS.ink2 }}>
              <span>Runs today</span>
              <span style={{ fontFamily: FONT_MONO, color: TOKENS.ink, fontWeight: 600 }}>{u.runs_today}</span>
            </div>
          </div>

          <div style={{ padding: 6 }}>
            <MenuItem>My runs</MenuItem>
            <MenuItem>API keys</MenuItem>
            <MenuItem>Team &amp; sharing</MenuItem>
            <MenuItem>Settings</MenuItem>
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
function LeftRail({ data, reactions, totalProducts, avgYield, verifiedCount, onOpenLightbox }) {
  return (
    <aside style={{
      borderRight: `1px solid ${TOKENS.rule}`,
      background: "white",
      padding: 16, display: "flex", flexDirection: "column", gap: 14,
      overflow: "auto", minWidth: 0,
    }}>
      <SectionLabel right={<button onClick={onOpenLightbox} style={{
        background:"transparent", border:"none", padding: 0, cursor:"pointer",
        fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.accent, fontWeight: 600,
      }}>Enlarge ↗</button>}>Source figure</SectionLabel>

      <button onClick={onOpenLightbox} className="figure-thumb" style={{
        background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`,
        borderRadius: 6, padding: 8, position: "relative", cursor: "zoom-in",
        display: "block", width: "100%", textAlign: "left",
      }}>
        <img src={data.source_image} alt="source figure"
             style={{ width: "100%", display: "block", borderRadius: 3 }}/>
        <div style={{
          position:"absolute", top: 10, left: 10,
          background: "rgba(255,255,255,0.9)", padding: "2px 6px", borderRadius: 3,
          fontFamily: FONT_MONO, fontSize: 10, color: TOKENS.ink2, border: `1px solid ${TOKENS.rule}`,
        }}>input.png · 1240×860</div>
        <div className="figure-zoom-btn" style={{
          position: "absolute", top: 10, right: 10,
          width: 28, height: 28, borderRadius: 4,
          background: "rgba(15,26,36,0.78)", color: "white",
          display: "grid", placeItems: "center",
          boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
            <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
          </svg>
        </div>
        <div className="figure-hover-hint" style={{
          position:"absolute", inset: 8, borderRadius: 3,
          background: "rgba(15,26,36,0.0)",
          display:"grid", placeItems:"center",
          pointerEvents: "none",
          transition: "background .15s ease",
        }}>
          <span className="figure-hover-label" style={{
            background: "rgba(15,26,36,0.85)", color:"white",
            padding: "5px 10px", borderRadius: 4,
            fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
            opacity: 0, transition: "opacity .15s ease",
            display: "inline-flex", alignItems: "center", gap: 6,
          }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
            </svg>
            Click to enlarge
          </span>
        </div>
      </button>
      <p style={{
        margin: 0, fontFamily: FONT_SERIF, fontStyle: "italic",
        fontSize: 13, color: TOKENS.ink2, lineHeight: 1.4,
      }}>{data.figure_caption}</p>

      <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
        <SectionLabel>Run metadata</SectionLabel>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(2, 1fr)", gap: 10, marginBottom: 10 }}>
          <Stat label="Reactions" value={reactions.length}/>
          <Stat label="Products"  value={totalProducts}/>
          <Stat label="Avg yield" value={avgYield != null ? `${avgYield}%` : "—"} accent={avgYield != null && avgYield >= 75 ? TOKENS.pos : null}/>
          <Stat label="Verified"  value={`${verifiedCount} / ${reactions.length}`}/>
          <Stat label="Steps"     value={MOCK_METADATA.steps} mono/>
          <Stat label="Wall time" value={`${MOCK_METADATA.wall_time_s}s`} mono/>
        </div>
        <div style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted, display:"flex", justifyContent:"space-between" }}>
          <span>tokens</span>
          <span>{MOCK_METADATA.input_tokens.toLocaleString()} in · {MOCK_METADATA.output_tokens.toLocaleString()} out</span>
        </div>
      </div>

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

      <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
        <SectionLabel>Export</SectionLabel>
        <div style={{ display:"grid", gap: 6 }}>
          <Button icon={<Icon.download/>}>reactions.csv</Button>
          <Button icon={<Icon.download/>}>extraction.json</Button>
          <Button icon={<Icon.download/>}>insight.csv</Button>
        </div>
      </div>
    </aside>
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
          <button onClick={() => {}}       style={bulkBtn}>↓ Export selected</button>
          <button onClick={() => {}}       style={{ ...bulkBtn, color: TOKENS.err }}>✗ Delete</button>
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
          options={[{v:"table", l:"Table"}, {v:"card", l:"Cards"}, {v:"equation", l:"Equation"}]}/>
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
                    <RowIconBtn title="Copy SMILES">  <Icon.copy/></RowIconBtn>
                    <RowIconBtn title="Flag for review"><Icon.flag/></RowIconBtn>
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

// ── Cards / Equations views ─────────────────────────────────────────────
function ReactionCards({ rows, dens, onOpenDetail }) {
  return (
    <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(340px, 1fr))", gap: 14 }}>
      {rows.map(({ r, i }) => <ReactionCard key={i} rxn={r} dens={dens} onClick={() => onOpenDetail(i)}/>)}
    </div>
  );
}
function ReactionCard({ rxn, dens, onClick }) {
  return (
    <article onClick={onClick} style={{
      background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
      padding: dens.pad + 4, display:"flex", flexDirection:"column", gap: 10,
      cursor: "pointer",
    }}>
      <header style={{ display:"flex", alignItems:"center", gap: 8 }}>
        <code style={{
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
          background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
          padding: "1px 6px", borderRadius: 3,
        }}>{rxn.entry_id}</code>
        <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 14, color: TOKENS.ink, flex: 1, minWidth: 0, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{rxn.title}</span>
        {(rxn.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note}/>)}
      </header>
      <div style={{ display:"flex", gap: 6, flexWrap:"wrap" }}>
        {(rxn.reactants || []).map((m, j) => <MolChip key={j} mol={m}/>)}
      </div>
      <ConditionRow conditions={rxn.conditions} reagents={rxn.reagents} compact/>
      <div style={{ display:"flex", gap: 6, flexWrap:"wrap" }}>
        {(rxn.products || []).map((m, j) => <MolChip key={j} mol={m}/>)}
      </div>
      {rxn.notes && (
        <p style={{ margin: 0, fontFamily: FONT_SERIF, fontSize: 12.5, color: TOKENS.ink2, fontStyle: "italic" }}>{rxn.notes}</p>
      )}
    </article>
  );
}

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
  const top = (rxn.reagents || []).map(r => `${r.label ? r.label + ": " : ""}${r.name}${r.loading ? ` (${r.loading})` : ""}`).join(", ");
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
              <Dt>Class</Dt><Dd>{insight.reaction_class}</Dd>
              {insight.named_reaction && (<><Dt>Named</Dt><Dd italic>{insight.named_reaction}</Dd></>)}
              <Dt>Class ID</Dt><Dd mono>{insight.rxn_class_id}</Dd>
              <Dt>Functional</Dt><Dd>{insight.functional_groups.join(", ")}</Dd>
              <Dt>Byproducts</Dt><Dd mono>{insight.byproducts.join(", ")}</Dd>
              <Dt>Hazards</Dt><Dd>{insight.hazards.join(", ")}</Dd>
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
        <Button small icon={<Icon.flag/>}>Flag</Button>
        <div style={{ flex: 1 }}/>
        <Button small icon={<Icon.copy/>}>Copy RXN-SMILES</Button>
      </footer>
    </aside>
  );
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
      display:"grid", gridTemplateColumns:"36px 1fr auto", gap: 8, alignItems:"center",
      padding: 8, background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
    }}>
      <div style={{ width: 36, height: 30 }}>
        <MolStructurePlaceholder smiles={mol.smiles} width={36} height={30}/>
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
        <button title="Edit" style={{
          width: 22, height: 22, padding: 0,
          background: "transparent", border: `1px solid ${TOKENS.rule}`,
          color: TOKENS.muted, borderRadius: 4, cursor: "pointer",
          display:"grid", placeItems:"center",
        }}><Icon.edit/></button>
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

// ── Sign-in screen ──────────────────────────────────────────────────────
function SignInScreen({ user, onSignIn }) {
  const [email, setEmail] = useState(user?.email || "");
  const [submitting, setSubmitting] = useState(false);

  function submit(e) {
    e?.preventDefault?.();
    if (!email) return;
    setSubmitting(true);
    setTimeout(() => { onSignIn(); setSubmitting(false); }, 600);
  }

  return (
    <div style={{
      position: "absolute", inset: 0, zIndex: 200,
      background: TOKENS.paper,
      display: "grid", placeItems: "center", padding: 24,
    }}>
      <div style={{
        width: 380, background: "white",
        border: `1px solid ${TOKENS.rule}`, borderRadius: 10,
        padding: 28,
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

        <div style={{ height: 1, background: TOKENS.rule2 }}/>

        <p style={{
          margin: 0, fontFamily: FONT_SERIF, fontStyle: "italic", fontSize: 14,
          color: TOKENS.ink2, lineHeight: 1.5,
        }}>You’ve been signed out. Sign back in to access your runs, API keys, and quota.</p>

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
              onFocus={(e) => e.target.style.borderColor = TOKENS.accent}
              onBlur={(e) => e.target.style.borderColor = TOKENS.rule}
            />
          </label>

          <button type="submit" disabled={!email || submitting} style={{
            padding: "10px 14px",
            background: TOKENS.accent, color: "white",
            border: `1px solid ${TOKENS.accent}`,
            borderRadius: 6, cursor: email && !submitting ? "pointer" : "not-allowed",
            fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13.5,
            opacity: !email ? 0.5 : 1,
            display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
          }}>
            {submitting && <span style={{
              width: 12, height: 12, borderRadius: 999,
              border: "2px solid rgba(255,255,255,0.4)", borderTopColor: "white",
              animation: "spin 0.6s linear infinite",
            }}/>}
            {submitting ? "Signing in…" : "Continue with email"}
          </button>
        </form>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ flex: 1, height: 1, background: TOKENS.rule2 }}/>
          <span style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.muted, letterSpacing: 0.5 }}>OR</span>
          <div style={{ flex: 1, height: 1, background: TOKENS.rule2 }}/>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <SSOButton onClick={() => { setSubmitting(true); setTimeout(onSignIn, 500); }}
                     icon={<svg width="14" height="14" viewBox="0 0 48 48"><path fill="#4285F4" d="M45.12 24.5c0-1.56-.14-3.06-.4-4.5H24v8.51h11.84c-.51 2.75-2.06 5.08-4.39 6.64v5.52h7.11c4.16-3.83 6.56-9.47 6.56-16.17z"/><path fill="#34A853" d="M24 46c5.94 0 10.92-1.97 14.56-5.33l-7.11-5.52c-1.97 1.32-4.49 2.1-7.45 2.1-5.73 0-10.58-3.87-12.31-9.07H4.34v5.7C7.96 41.07 15.4 46 24 46z"/><path fill="#FBBC05" d="M11.69 28.18c-.44-1.32-.69-2.73-.69-4.18s.25-2.86.69-4.18v-5.7H4.34A21.99 21.99 0 0 0 2 24c0 3.55.85 6.91 2.34 9.88l7.35-5.7z"/><path fill="#EA4335" d="M24 10.75c3.23 0 6.13 1.11 8.42 3.29l6.31-6.31C34.91 4.18 29.93 2 24 2 15.4 2 7.96 6.93 4.34 14.12l7.35 5.7c1.73-5.2 6.58-9.07 12.31-9.07z"/></svg>}>
            Continue with Google
          </SSOButton>
          <SSOButton onClick={() => { setSubmitting(true); setTimeout(onSignIn, 500); }}
                     icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="#0F1A24"><path d="M12 2C6.48 2 2 6.48 2 12c0 4.41 2.87 8.14 6.84 9.47.5.09.66-.22.66-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.89 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.99 1.03-2.69-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02.8-.22 1.65-.33 2.5-.33s1.7.11 2.5.33c1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.6 1.03 2.69 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.74c0 .27.16.58.67.48A10.01 10.01 0 0 0 22 12c0-5.52-4.48-10-10-10z"/></svg>}>
            Continue with GitHub
          </SSOButton>
          <SSOButton onClick={() => { setSubmitting(true); setTimeout(onSignIn, 500); }}
                     icon={<svg width="14" height="14" viewBox="0 0 24 24" fill="#0078D4"><path d="M11.4 24H0V12.6h11.4V24zM24 24H12.6V12.6H24V24zM11.4 11.4H0V0h11.4v11.4zM24 11.4H12.6V0H24v11.4z"/></svg>}>
            Continue with SSO (Microsoft)
          </SSOButton>
        </div>

        <p style={{
          margin: 0, fontFamily: FONT_SANS, fontSize: 11.5, color: TOKENS.muted,
          textAlign: "center", lineHeight: 1.5,
        }}>By signing in you agree to the <a href="#" style={{ color: TOKENS.accent, textDecoration: "none" }}>terms</a> and <a href="#" style={{ color: TOKENS.accent, textDecoration: "none" }}>privacy policy</a>.</p>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function SSOButton({ children, icon, onClick }) {
  return (
    <button onClick={onClick} style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
      padding: "8px 12px",
      background: "white", color: TOKENS.ink,
      border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
      fontFamily: FONT_SANS, fontWeight: 600, fontSize: 13,
      cursor: "pointer",
    }} onMouseOver={(e) => e.currentTarget.style.background = TOKENS.paper}
       onMouseOut={(e) => e.currentTarget.style.background = "white"}>
      {icon}{children}
    </button>
  );
}

// ── Figure lightbox ─────────────────────────────────────────────────────
function FigureLightbox({ src, caption, onClose }) {
  const [zoom, setZoom] = useState(1); // 1 = fit, 2 = 2x, etc.
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
      {/* Header */}
      <div onClick={(e) => e.stopPropagation()} style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "10px 18px", color: "white",
        background: "rgba(0,0,0,0.25)",
        cursor: "default",
      }}>
        <span style={{ fontFamily: FONT_MONO, fontSize: 12 }}>input.png · 1240×860</span>
        <div style={{ flex: 1 }}/>
        <div style={{ display:"flex", alignItems:"center", gap: 4, background:"rgba(255,255,255,0.1)", borderRadius: 6, padding: 2 }}>
          <ZoomBtn onClick={() => setZoom(z => Math.max(z / 1.4, 0.5))}>−</ZoomBtn>
          <button onClick={() => setZoom(1)} style={{
            background: "transparent", border: "none", color: "white",
            fontFamily: FONT_MONO, fontSize: 12, padding: "4px 10px", cursor: "pointer", minWidth: 56,
          }}>{Math.round(zoom * 100)}%</button>
          <ZoomBtn onClick={() => setZoom(z => Math.min(z * 1.4, 6))}>+</ZoomBtn>
          <span style={{ width: 1, height: 16, background: "rgba(255,255,255,0.2)", margin: "0 2px" }}/>
          <ZoomBtn onClick={() => setZoom(1)} title="Fit">⤢</ZoomBtn>
        </div>
        <button onClick={onClose} style={{
          background: "transparent", border: "1px solid rgba(255,255,255,0.3)",
          color: "white", borderRadius: 4, padding: "4px 10px", cursor: "pointer",
          fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
        }}>Close · esc</button>
      </div>

      {/* Image area */}
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

      {/* Caption */}
      <div onClick={(e) => e.stopPropagation()} style={{
        padding: "10px 24px", textAlign: "center",
        background: "rgba(0,0,0,0.25)",
        cursor: "default",
      }}>
        <p style={{ margin: 0, fontFamily: FONT_SERIF, fontStyle:"italic", fontSize: 14, color: "rgba(255,255,255,0.9)" }}>{caption}</p>
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
    ["j / k",   "Next / previous reaction"],
    ["enter",   "Open detail drawer"],
    ["v",       "Toggle verified"],
    ["x",       "Toggle selected"],
    ["esc",     "Close drawer / clear selection"],
    ["⌘ / ⌃ a", "Select all (filtered)"],
    ["⌘ / ⌃ e", "Export selected"],
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

// ── Empty state ─────────────────────────────────────────────────────────
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

Object.assign(window, { WorkbenchApp });
