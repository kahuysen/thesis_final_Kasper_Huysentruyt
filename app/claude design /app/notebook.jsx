// Direction B — Lab Notebook
// Paper-feel, serif headings, generous whitespace, equation-row dominant.
// Reads like a chemistry notebook entry; figure pinned as a sidebar plate.

function NotebookApp({ tweaks }) {
  const data =
    tweaks.dataset === "small"  ? MOCK_SMALL :
    tweaks.dataset === "large"  ? MOCK_LARGE : MOCK_MEDIUM;
  const dens = useDensity(tweaks.density);
  const [modelId, setModelId] = useState(MOCK_MODELS[0].id);
  const [layout, setLayout]   = useState(tweaks.layout);
  useEffect(() => setLayout(tweaks.layout), [tweaks.layout]);
  const [logOpen, setLogOpen] = useState(false);
  const [insightOn, setInsightOn] = useState(true);

  const reactions = data.reactions;
  const insights = useMemo(() => {
    if (!insightOn) return {};
    const out = {};
    for (const r of reactions) if (MOCK_INSIGHT[r.entry_id]) out[r.entry_id] = MOCK_INSIGHT[r.entry_id];
    return out;
  }, [reactions, insightOn]);

  const totalProducts = reactions.reduce((a,r) => a + (r.products||[]).length, 0);
  const yields = reactions.flatMap(r => (r.products||[]).map(p => p.yield_pct)).filter(v => v != null);
  const avgYield = yields.length ? Math.round(yields.reduce((a,b)=>a+b,0) / yields.length) : null;

  return (
    <div style={{
      width: "100%", height: "100%",
      background: TOKENS.paper, color: TOKENS.ink,
      fontFamily: FONT_SERIF, fontSize: 15, lineHeight: 1.5,
      display: "grid", gridTemplateRows: "auto 1fr auto",
      overflow: "hidden",
    }}>

      {/* ── Notebook header ──────────────────────────────────── */}
      <header style={{
        display:"flex", alignItems:"flex-end", gap: 16,
        padding: "14px 32px 12px",
        borderBottom: `1px solid ${TOKENS.rule}`,
        background: TOKENS.paper,
      }}>
        <div style={{ display:"flex", alignItems:"baseline", gap: 12 }}>
          <h1 style={{
            margin: 0, fontFamily: FONT_SERIF, fontWeight: 600, fontStyle:"italic",
            fontSize: 26, color: TOKENS.ink, letterSpacing: -0.4, lineHeight: 1,
          }}>Reaction Miner</h1>
          <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted, letterSpacing: 1, textTransform:"uppercase" }}>
            Lab notebook · Vol. 04 · entry 0218
          </span>
        </div>
        <div style={{ flex: 1 }}/>
        <div style={{ display:"flex", alignItems:"center", gap: 10 }}>
          <ModelPickerCard models={MOCK_MODELS} value={modelId} onChange={setModelId} compact/>
          <Button icon={<Icon.upload/>}>New figure</Button>
        </div>
      </header>

      {/* ── Sub-bar: date, run, status ───────────────────────── */}
      <div style={{ display: "contents" }}>
        <div style={{
          display:"grid", gridTemplateColumns:"320px 1fr",
          minHeight: 0, background: TOKENS.paper,
        }}>

          {/* Left plate: figure + caption + meta */}
          <aside style={{
            padding: "20px 24px 24px 32px",
            borderRight: `1px dashed ${TOKENS.rule}`,
            overflow: "auto",
            display:"flex", flexDirection:"column", gap: 14,
          }}>
            <div style={{
              fontFamily: FONT_SANS, fontSize: 10, color: TOKENS.muted,
              letterSpacing: 1.4, textTransform: "uppercase", fontWeight: 700,
              display:"flex", justifyContent:"space-between",
            }}>
              <span>Plate 1</span>
              <span>20 May 2026 · 14:08</span>
            </div>
            <figure style={{ margin: 0 }}>
              <div style={{
                background: "white", padding: 10, border: `1px solid ${TOKENS.rule}`,
                boxShadow: "0 1px 0 rgba(15,26,36,0.04), 0 6px 18px -8px rgba(15,26,36,0.12)",
              }}>
                <img src={data.source_image} alt="source figure" style={{ width:"100%", display:"block" }}/>
              </div>
              <figcaption style={{
                marginTop: 10, fontFamily: FONT_SERIF, fontSize: 13.5,
                color: TOKENS.ink2, fontStyle: "italic", lineHeight: 1.5,
              }}>{data.figure_caption}</figcaption>
            </figure>

            <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
              <SmallCapsLabel>Run · ƒ(figure)</SmallCapsLabel>
              <dl style={{ margin: "8px 0 0", display:"grid", gridTemplateColumns:"auto 1fr", rowGap: 4, columnGap: 12, fontFamily: FONT_SANS, fontSize: 12.5 }}>
                <Dt>Run id</Dt><Dd mono>13415d0010f8</Dd>
                <Dt>Model</Dt><Dd>{MOCK_MODELS.find(m=>m.id===modelId)?.label}</Dd>
                <Dt>Steps</Dt><Dd mono>{MOCK_METADATA.steps}</Dd>
                <Dt>Tokens</Dt><Dd mono>{MOCK_METADATA.input_tokens.toLocaleString()} / {MOCK_METADATA.output_tokens.toLocaleString()}</Dd>
                <Dt>Wall</Dt><Dd mono>{MOCK_METADATA.wall_time_s}s</Dd>
                <Dt>Reactions</Dt><Dd mono>{reactions.length}</Dd>
                <Dt>Products</Dt><Dd mono>{totalProducts}</Dd>
                {avgYield != null && (<><Dt>Avg yield</Dt><Dd mono>{avgYield}%</Dd></>)}
              </dl>
            </div>

            {data.extraction_notes && (
              <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
                <SmallCapsLabel>Extractor margin note</SmallCapsLabel>
                <p style={{
                  margin: "8px 0 0", fontFamily: FONT_SERIF, fontSize: 13.5,
                  color: TOKENS.ink2, fontStyle: "italic", lineHeight: 1.55,
                  borderLeft: `2px solid ${TOKENS.warn}`, paddingLeft: 10,
                }}>{data.extraction_notes}</p>
              </div>
            )}

            <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
              <SmallCapsLabel>Export</SmallCapsLabel>
              <div style={{ display:"grid", gap: 6, marginTop: 8 }}>
                <Button small icon={<Icon.download/>}>reactions.csv</Button>
                <Button small icon={<Icon.download/>}>extraction.json</Button>
                <Button small icon={<Icon.download/>}>insight.csv</Button>
              </div>
            </div>
          </aside>

          {/* Notebook body */}
          <main style={{ overflow:"auto", padding: "20px 36px 36px" }}>
            <div style={{ maxWidth: 880, margin: "0 auto" }}>
              <div style={{ display:"flex", alignItems:"baseline", justifyContent:"space-between", gap: 12, marginBottom: 18 }}>
                <h2 style={{
                  margin: 0, fontFamily: FONT_SERIF, fontWeight: 600, fontStyle:"italic",
                  fontSize: 22, color: TOKENS.ink, letterSpacing: -0.2,
                }}>
                  Extracted reactions
                  <span style={{ fontFamily: FONT_SANS, fontStyle:"normal", fontSize: 12, color: TOKENS.muted, marginLeft: 10, fontWeight: 500, letterSpacing: 0.4 }}>
                    {reactions.length} entries
                  </span>
                </h2>
                <SegmentedControl value={layout} onChange={setLayout}
                  options={[{v:"equation", l:"Equation"}, {v:"card", l:"Cards"}, {v:"table", l:"Table"}]}/>
              </div>

              {layout === "equation" && (
                <div style={{ display:"flex", flexDirection:"column", gap: 28 }}>
                  {reactions.map((r, i) => <NotebookEntry key={i} index={i+1} rxn={r} dens={dens}/>)}
                </div>
              )}
              {layout === "card" && (
                <div style={{ display:"grid", gridTemplateColumns: reactions.length > 4 ? "repeat(2, 1fr)" : "1fr", gap: 16 }}>
                  {reactions.map((r, i) => <NotebookCard key={i} rxn={r} dens={dens}/>)}
                </div>
              )}
              {layout === "table" && (
                <NotebookTable reactions={reactions} dens={dens}/>
              )}

              <div style={{ marginTop: 32 }}>
                <SmallCapsLabel>Enrichment · Rxn-INSIGHT</SmallCapsLabel>
                <div style={{ marginTop: 10 }}>
                  <StructuredInsightPanel analyses={insights} dens={dens}/>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>

      <ActivityDrawer log={MOCK_AGENT_LOG} open={logOpen} onToggle={() => setLogOpen(o => !o)} visible={tweaks.showLog}/>
    </div>
  );
}

// ── Small typographic helpers ─────────────────────────────────────────────
function SmallCapsLabel({ children }) {
  return (
    <h3 style={{
      margin: 0, fontFamily: FONT_SANS, fontSize: 10.5, fontWeight: 700,
      textTransform: "uppercase", letterSpacing: 1.4, color: TOKENS.muted,
    }}>{children}</h3>
  );
}
function Dt({ children }) {
  return <dt style={{ color: TOKENS.muted, fontFamily: FONT_SANS }}>{children}</dt>;
}
function Dd({ children, mono }) {
  return <dd style={{ margin: 0, color: TOKENS.ink, fontFamily: mono ? FONT_MONO : FONT_SANS, fontVariantNumeric:"tabular-nums" }}>{children}</dd>;
}

// ── Notebook entry (equation form) ────────────────────────────────────────
function NotebookEntry({ index, rxn, dens }) {
  return (
    <article style={{ position: "relative", display:"flex", flexDirection:"column", gap: 12 }}>
      <header style={{ display:"flex", alignItems:"baseline", gap: 12, borderBottom: `1px solid ${TOKENS.rule}`, paddingBottom: 8 }}>
        <span style={{
          width: 26, height: 26, borderRadius: 999,
          border: `1.5px solid ${TOKENS.ink}`, background: TOKENS.paper,
          display: "grid", placeItems: "center",
          fontFamily: FONT_SERIF, fontStyle: "italic", fontWeight: 700, fontSize: 14, color: TOKENS.ink,
        }}>{index}</span>
        <h4 style={{
          margin: 0, fontFamily: FONT_SERIF, fontWeight: 600, fontStyle:"italic",
          fontSize: 18, color: TOKENS.ink, flex: 1, minWidth: 0,
        }}>{rxn.title}</h4>
        <code style={{
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted,
          letterSpacing: 0.5,
        }}>entry · {rxn.entry_id}</code>
        {(rxn.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note} big/>)}
      </header>

      {/* Equation row on paper plate */}
      <div style={{
        background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
        padding: 4, boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.5)",
      }}>
        <EquationRow rxn={rxn} dens={dens}/>
      </div>

      {/* Conditions paragraph */}
      <ConditionsParagraph rxn={rxn}/>

      {rxn.notes && (
        <p style={{
          margin: 0, fontFamily: FONT_SERIF, fontSize: 14, fontStyle: "italic",
          color: TOKENS.ink2, lineHeight: 1.55,
          paddingLeft: 12, borderLeft: `2px solid ${TOKENS.rule}`,
        }}><strong style={{ fontStyle:"normal", color: TOKENS.muted, marginRight: 6, fontSize: 11, letterSpacing: 1, textTransform:"uppercase" }}>Note</strong>{rxn.notes}</p>
      )}
    </article>
  );
}

function ConditionsParagraph({ rxn }) {
  const c = rxn.conditions || {};
  const parts = [];
  for (const r of (rxn.reagents || [])) {
    parts.push({ k: r.role, label: r.name, sub: r.loading });
  }
  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap: 10, alignItems:"center", fontFamily: FONT_SANS, fontSize: 13, color: TOKENS.ink2 }}>
      {parts.map((p, i) => (
        <span key={i} style={{ display:"inline-flex", alignItems:"center", gap: 5 }}>
          <span style={{ width: 6, height: 6, borderRadius: 999, background: TOKENS[p.k] || TOKENS.ink2 }}/>
          <span style={{ color: TOKENS.ink, fontWeight: 600 }}>{p.label}</span>
          {p.sub && <span style={{ color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 11.5 }}>({p.sub})</span>}
        </span>
      ))}
      {(c.solvent || c.temperature || c.time) && (
        <span style={{
          display:"inline-flex", alignItems:"center", gap: 6,
          paddingLeft: 12, marginLeft: 2, borderLeft: `1px solid ${TOKENS.rule}`,
          fontFamily: FONT_SERIF, fontStyle: "italic", fontSize: 14, color: TOKENS.ink,
        }}>
          {[c.solvent, c.temperature, c.time, c.atmosphere].filter(Boolean).join(", ")}
        </span>
      )}
    </div>
  );
}

// ── Notebook card view ────────────────────────────────────────────────────
function NotebookCard({ rxn, dens }) {
  return (
    <article style={{
      background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
      padding: 16, display:"flex", flexDirection:"column", gap: 10,
      boxShadow: "0 1px 0 rgba(15,26,36,0.04)",
    }}>
      <header style={{ display:"flex", alignItems:"baseline", gap: 8 }}>
        <h4 style={{
          margin: 0, fontFamily: FONT_SERIF, fontStyle:"italic", fontWeight: 600,
          fontSize: 16, color: TOKENS.ink, flex: 1, minWidth: 0,
        }}>{rxn.title}</h4>
        <code style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>{rxn.entry_id}</code>
      </header>

      <div style={{
        display:"grid", gridTemplateColumns:"1fr auto 1fr", alignItems:"center", gap: 8,
        background: TOKENS.paper, borderRadius: 6, padding: 8,
      }}>
        <div style={{ display:"flex", gap: 4, flexWrap:"wrap", justifyContent:"center" }}>
          {(rxn.reactants || []).map((m, j) => (
            <MolStructurePlaceholder key={j} smiles={m.smiles} width={60} height={50}/>
          ))}
        </div>
        <div style={{ minWidth: 40, opacity: 0.6 }}><Icon.arrow/></div>
        <div style={{ display:"flex", gap: 4, flexWrap:"wrap", justifyContent:"center" }}>
          {(rxn.products || []).map((m, j) => (
            <MolStructurePlaceholder key={j} smiles={m.smiles} width={60} height={50}/>
          ))}
        </div>
      </div>

      <ConditionRow conditions={rxn.conditions} reagents={rxn.reagents} compact/>

      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap: 8 }}>
        <div style={{ display:"flex", gap: 6, alignItems:"center" }}>
          {(rxn.products || []).map((p, j) => (
            <Fragment key={j}>
              <code style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.ink2 }}>{p.label || p.name}</code>
              <YieldBadge value={p.yield_pct} note={p.yield_note}/>
            </Fragment>
          ))}
        </div>
        <CopyButton text={(rxn.products[0] || {}).smiles}/>
      </div>
    </article>
  );
}

// ── Notebook table ────────────────────────────────────────────────────────
function NotebookTable({ reactions, dens }) {
  return (
    <div style={{ background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8, overflow: "hidden" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: FONT_SANS, fontSize: 13 }}>
        <thead>
          <tr style={{ background: TOKENS.paper }}>
            <th style={thStyle}>#</th>
            <th style={thStyle}>Entry</th>
            <th style={thStyle}>Title</th>
            <th style={thStyle}>Product</th>
            <th style={thStyleR}>Yield</th>
          </tr>
        </thead>
        <tbody>
          {reactions.map((r, i) => {
            const p = (r.products || [])[0] || {};
            return (
              <tr key={i} style={{ borderTop: `1px solid ${TOKENS.rule}` }}>
                <td style={tdStyle}>
                  <span style={{
                    width: 22, height: 22, borderRadius: 999, border: `1.5px solid ${TOKENS.ink}`,
                    display:"inline-grid", placeItems:"center",
                    fontFamily: FONT_SERIF, fontStyle:"italic", fontWeight: 700, fontSize: 12,
                  }}>{i+1}</span>
                </td>
                <td style={tdStyle}>
                  <code style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent, background: "#E8F1F8", padding:"1px 6px", borderRadius: 3 }}>{r.entry_id}</code>
                </td>
                <td style={{ ...tdStyle, fontFamily: FONT_SERIF, fontStyle:"italic", color: TOKENS.ink }}>{r.title}</td>
                <td style={tdStyle}>{p.name || "—"}</td>
                <td style={tdStyleR}><YieldBadge value={p.yield_pct} note={p.yield_note}/></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
const thStyle  = { padding:"10px 12px", textAlign:"left",  fontWeight: 600, fontSize: 11, textTransform:"uppercase", letterSpacing: 0.8, color: TOKENS.muted, borderBottom: `1px solid ${TOKENS.rule}` };
const thStyleR = { ...thStyle, textAlign:"right" };
const tdStyle  = { padding:"10px 12px", color: TOKENS.ink2, verticalAlign:"middle" };
const tdStyleR = { ...tdStyle, textAlign:"right" };

Object.assign(window, { NotebookApp });
