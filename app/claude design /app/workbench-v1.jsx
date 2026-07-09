// Direction A — Workbench
// Data-engineer-first: dense, table-dominant, source figure pinned left.

function WorkbenchApp({ tweaks }) {
  const data =
  tweaks.dataset === "small" ? MOCK_SMALL :
  tweaks.dataset === "large" ? MOCK_LARGE : MOCK_MEDIUM;
  const dens = useDensity(tweaks.density);
  const [modelId, setModelId] = useState(MOCK_MODELS[0].id);
  const [layout, setLayout] = useState(tweaks.layout);
  useEffect(() => setLayout(tweaks.layout), [tweaks.layout]);
  const [logOpen, setLogOpen] = useState(false);
  const [selectedRxn, setSelectedRxn] = useState(0);
  const [verifiedSet, setVerifiedSet] = useState(
    () => new Set(data.reactions.map((r, i) => r.verified ? i : null).filter((v) => v !== null))
  );
  useEffect(() => {
    setVerifiedSet(new Set(data.reactions.map((r, i) => r.verified ? i : null).filter((v) => v !== null)));
    setSelectedRxn(0);
  }, [tweaks.dataset]);

  const reactions = data.reactions;
  const insights = useMemo(() => {
    const out = {};
    for (const r of reactions) if (MOCK_INSIGHT[r.entry_id]) out[r.entry_id] = MOCK_INSIGHT[r.entry_id];
    return out;
  }, [reactions]);

  const totalProducts = reactions.reduce((a, r) => a + (r.products || []).length, 0);
  const yields = reactions.flatMap((r) => (r.products || []).map((p) => p.yield_pct)).filter((v) => v != null);
  const avgYield = yields.length ? Math.round(yields.reduce((a, b) => a + b, 0) / yields.length) : null;
  const verifiedCount = verifiedSet.size;

  return (
    <div style={{
      width: "100%", height: "100%",
      background: TOKENS.paper, color: TOKENS.ink,
      fontFamily: FONT_SANS, fontSize: 14, lineHeight: 1.45,
      display: "grid", gridTemplateRows: "auto 1fr auto",
      overflow: "hidden"
    }}>
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header style={{
        display: "flex", alignItems: "center", gap: 14,
        padding: "10px 18px",
        borderBottom: `1px solid ${TOKENS.rule}`,
        background: "white"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 26, height: 26, borderRadius: 6,
            background: TOKENS.ink, color: "white",
            display: "grid", placeItems: "center",
            fontFamily: FONT_SERIF, fontStyle: "italic", fontWeight: 700, fontSize: 16
          }}>R</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink, letterSpacing: 0.1 }}>Rxn-EXCTRACTION</span>
            <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>run · 13415d0010f8 · 2 min ago</span>
          </div>
        </div>

        <div style={{ flex: 1 }} />

        <div style={{
          display: "inline-flex", alignItems: "center", gap: 6,
          padding: "4px 10px", borderRadius: 999,
          background: TOKENS.pos2, border: `1px solid ${TOKENS.pos}33`,
          color: TOKENS.pos, fontFamily: FONT_SANS, fontSize: 11.5, fontWeight: 600
        }}>
          <span style={{ width: 7, height: 7, borderRadius: 999, background: TOKENS.pos }} />
          Extraction complete · {reactions.length} reactions
        </div>

        <ModelPickerCard models={MOCK_MODELS} value={modelId} onChange={setModelId} compact />

        <Button icon={<Icon.upload />}>New figure</Button>
        <Button primary icon={<Icon.spark />}>Re-run with INSIGHT</Button>
      </header>

      {/* ── Main grid ───────────────────────────────────────────── */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "320px 1fr",
        gap: 0, minHeight: 0
      }}>

        {/* Left rail: figure + meta */}
        <aside style={{
          borderRight: `1px solid ${TOKENS.rule}`,
          background: "white",
          padding: 16, display: "flex", flexDirection: "column", gap: 14,
          overflow: "auto"
        }}>
          <SectionLabel>Source figure</SectionLabel>
          <div style={{
            background: TOKENS.paper, border: `1px solid ${TOKENS.rule}`,
            borderRadius: 6, padding: 8, position: "relative"
          }}>
            <img src={data.source_image} alt="source figure"
            style={{ width: "100%", display: "block", borderRadius: 3 }} />
            <div style={{
              position: "absolute", top: 10, right: 10,
              background: "rgba(255,255,255,0.9)", padding: "2px 6px", borderRadius: 3,
              fontFamily: FONT_MONO, fontSize: 10, color: TOKENS.ink2, border: `1px solid ${TOKENS.rule}`
            }}>input.png · 1240×860</div>
          </div>
          <p style={{
            margin: 0, fontFamily: FONT_SERIF, fontStyle: "italic",
            fontSize: 13, color: TOKENS.ink2, lineHeight: 1.4
          }}>{data.figure_caption}</p>

          <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
            <SectionLabel>Run metadata</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10, marginBottom: 10 }}>
              <Stat label="Reactions" value={reactions.length} />
              <Stat label="Products" value={totalProducts} />
              <Stat label="Avg yield" value={avgYield != null ? `${avgYield}%` : "—"} accent={avgYield != null && avgYield >= 75 ? TOKENS.pos : null} />
              <Stat label="Verified" value={`${verifiedCount} / ${reactions.length}`} />
              <Stat label="Steps" value={MOCK_METADATA.steps} mono />
              <Stat label="Wall time" value={`${MOCK_METADATA.wall_time_s}s`} mono />
            </div>
            <div style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted, display: "flex", justifyContent: "space-between" }}>
              <span>tokens</span>
              <span>{MOCK_METADATA.input_tokens.toLocaleString()} in · {MOCK_METADATA.output_tokens.toLocaleString()} out</span>
            </div>
          </div>

          {data.extraction_notes &&
          <div style={{
            borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12
          }}>
              <SectionLabel>Extractor note</SectionLabel>
              <p style={{
              margin: 0, padding: "8px 10px",
              background: TOKENS.warn2 + "55", borderLeft: `2px solid ${TOKENS.warn}`,
              fontFamily: FONT_SERIF, fontSize: 13, color: TOKENS.ink2, fontStyle: "italic", lineHeight: 1.5
            }}>{data.extraction_notes}</p>
            </div>
          }

          <div style={{ borderTop: `1px solid ${TOKENS.rule}`, paddingTop: 12 }}>
            <SectionLabel>Export</SectionLabel>
            <div style={{ display: "grid", gap: 6 }}>
              <Button icon={<Icon.download />}>reactions.csv</Button>
              <Button icon={<Icon.download />}>extraction.json</Button>
              <Button icon={<Icon.download />}>insight.csv</Button>
            </div>
          </div>
        </aside>

        {/* Right column: results */}
        <main style={{ overflow: "auto", padding: "16px 22px 24px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 14 }}>
            <h2 style={{
              margin: 0, fontFamily: FONT_SANS, fontWeight: 700, fontSize: 18, color: TOKENS.ink
            }}>Extracted reactions</h2>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <SegmentedControl value={layout} onChange={setLayout}
              options={[{ v: "table", l: "Table" }, { v: "card", l: "Cards" }, { v: "equation", l: "Equation" }]} />
            </div>
          </div>

          {layout === "table" &&
          <ReactionTable
            reactions={reactions} dens={dens}
            selected={selectedRxn} onSelect={setSelectedRxn}
            verifiedSet={verifiedSet}
            onToggleVerified={(i) => {
              const s = new Set(verifiedSet);
              s.has(i) ? s.delete(i) : s.add(i);
              setVerifiedSet(s);
            }} />

          }
          {layout === "card" &&
          <ReactionCards reactions={reactions} dens={dens} />
          }
          {layout === "equation" &&
          <ReactionEquations reactions={reactions} dens={dens} />
          }

          <div style={{ marginTop: 22 }}>
            <SectionLabel right={<span style={{ fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted }}>via rxn-insight v0.2.1</span>}>
              Enrichment
            </SectionLabel>
            <StructuredInsightPanel analyses={insights} dens={dens} />
          </div>
        </main>
      </div>

      <ActivityDrawer log={MOCK_AGENT_LOG} open={logOpen} onToggle={() => setLogOpen((o) => !o)} visible={tweaks.showLog} />
    </div>);

}

// ── Segmented control ─────────────────────────────────────────────────────
function SegmentedControl({ value, onChange, options }) {
  return (
    <div style={{
      display: "inline-flex", padding: 2, background: TOKENS.paper,
      border: `1px solid ${TOKENS.rule}`, borderRadius: 6
    }}>
      {options.map((o) =>
      <button key={o.v} onClick={() => onChange(o.v)} style={{
        padding: "5px 12px", border: "none",
        background: value === o.v ? "white" : "transparent",
        color: value === o.v ? TOKENS.ink : TOKENS.muted,
        fontFamily: FONT_SANS, fontWeight: 600, fontSize: 12,
        borderRadius: 4, cursor: "pointer",
        boxShadow: value === o.v ? `0 1px 2px rgba(15,26,36,0.06), inset 0 0 0 1px ${TOKENS.rule}` : "none"
      }}>{o.l}</button>
      )}
    </div>);

}

// ── Table view (data-engineer focus) ──────────────────────────────────────
function ReactionTable({ reactions, dens, selected, onSelect, verifiedSet, onToggleVerified }) {
  return (
    <div style={{
      background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
      overflow: "hidden"
    }}>
      <table style={{
        width: "100%", borderCollapse: "collapse",
        fontFamily: FONT_SANS, fontSize: dens.font - 1
      }}>
        <thead>
          <tr style={{ background: TOKENS.paper, textAlign: "left" }}>
            <Th w="28px"></Th>
            <Th w="78px">Entry</Th>
            <Th>Reactants</Th>
            <Th>Reagents · conditions</Th>
            <Th>Products</Th>
            <Th w="76px" align="right">Yield</Th>
          </tr>
        </thead>
        <tbody>
          {reactions.map((r, i) => {
            const sel = selected === i;
            const verified = verifiedSet.has(i);
            const expanded = sel;
            return (
              <Fragment key={i}>
                <tr onClick={() => onSelect(sel ? -1 : i)} style={{
                  borderTop: `1px solid ${TOKENS.rule}`,
                  background: sel ? "#F4F2E8" : "white",
                  cursor: "pointer"
                }}>
                  <Td>
                    <button onClick={(e) => {e.stopPropagation();onToggleVerified(i);}} style={{
                      width: 18, height: 18, padding: 0,
                      background: verified ? TOKENS.pos : "white",
                      border: `1.5px solid ${verified ? TOKENS.pos : TOKENS.rule}`,
                      borderRadius: 4, cursor: "pointer", color: "white",
                      display: "grid", placeItems: "center"
                    }} title={verified ? "Verified" : "Mark verified"}>
                      {verified && <Icon.check />}
                    </button>
                  </Td>
                  <Td>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <code style={{
                        fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
                        background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
                        padding: "1px 6px", borderRadius: 3, alignSelf: "flex-start"
                      }}>{r.entry_id}</code>
                    </div>
                  </Td>
                  <Td>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
                      {(r.reactants || []).map((m, j) => <MolChip key={j} mol={m} />)}
                    </div>
                  </Td>
                  <Td>
                    <ConditionRow conditions={r.conditions} reagents={r.reagents} compact />
                  </Td>
                  <Td>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
                      {(r.products || []).map((m, j) => <MolChip key={j} mol={m} />)}
                    </div>
                  </Td>
                  <Td align="right">
                    <div style={{ display: "flex", justifyContent: "flex-end", gap: 4, flexWrap: "wrap" }}>
                      {(r.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note} />)}
                    </div>
                  </Td>
                </tr>
                {expanded &&
                <tr style={{ background: "#FAF7EB", borderBottom: `1px solid ${TOKENS.rule}` }}>
                    <td colSpan={6} style={{ padding: 14 }}>
                      <ReactionDetail rxn={r} dens={dens} />
                    </td>
                  </tr>
                }
              </Fragment>);

          })}
        </tbody>
      </table>
    </div>);

}

function Th({ children, w, align }) {
  return (
    <th style={{
      padding: "9px 12px", fontWeight: 600, fontSize: 11,
      textTransform: "uppercase", letterSpacing: 0.6, color: TOKENS.muted,
      borderBottom: `1px solid ${TOKENS.rule}`,
      width: w, textAlign: align || "left", whiteSpace: "nowrap"
    }}>{children}</th>);

}
function Td({ children, align }) {
  return <td style={{ padding: "10px 12px", verticalAlign: "middle", textAlign: align || "left", color: TOKENS.ink2 }}>{children}</td>;
}

// Small molecule chip — used in table cells.
function MolChip({ mol }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      background: "white", border: `1px solid ${TOKENS.rule}`,
      borderRadius: 4, padding: "3px 7px 3px 4px"
    }}>
      <div style={{ width: 26, height: 22, display: "grid", placeItems: "center", flex: "0 0 auto" }}>
        <MolStructurePlaceholder smiles={mol.smiles} width={26} height={22} />
      </div>
      <span style={{ fontFamily: FONT_SANS, fontSize: 12, color: TOKENS.ink, fontWeight: 500, whiteSpace: "nowrap" }}>
        {mol.label && <span style={{ color: TOKENS.accent, fontWeight: 700, marginRight: 4 }}>{mol.label}</span>}
        {mol.name || "—"}
      </span>
    </div>);

}

// Expanded detail row in table view.
function ReactionDetail({ rxn, dens }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
      <div style={{ fontFamily: FONT_SERIF, fontSize: 15, color: TOKENS.ink, fontStyle: "italic" }}>{rxn.title}</div>
      <EquationRow rxn={rxn} dens={dens} />
      {rxn.notes &&
      <p style={{ margin: 0, fontFamily: FONT_SERIF, fontSize: 13, color: TOKENS.ink2, fontStyle: "italic", lineHeight: 1.5 }}>
          Note: {rxn.notes}
        </p>
      }
    </div>);

}

// ── Cards view ────────────────────────────────────────────────────────────
function ReactionCards({ reactions, dens }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 14 }}>
      {reactions.map((r, i) => <ReactionCard key={i} rxn={r} dens={dens} />)}
    </div>);

}
function ReactionCard({ rxn, dens }) {
  return (
    <article style={{
      background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
      padding: dens.pad + 4, display: "flex", flexDirection: "column", gap: 10
    }}>
      <header style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <code style={{
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
          background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
          padding: "1px 6px", borderRadius: 3
        }}>{rxn.entry_id}</code>
        <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 14, color: TOKENS.ink, flex: 1, minWidth: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{rxn.title}</span>
        {(rxn.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note} />)}
      </header>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {(rxn.reactants || []).map((m, j) => <MolChip key={j} mol={m} />)}
      </div>
      <ConditionRow conditions={rxn.conditions} reagents={rxn.reagents} compact />
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {(rxn.products || []).map((m, j) => <MolChip key={j} mol={m} />)}
      </div>
      {rxn.notes &&
      <p style={{ margin: 0, fontFamily: FONT_SERIF, fontSize: 12.5, color: TOKENS.ink2, fontStyle: "italic" }}>{rxn.notes}</p>
      }
    </article>);

}

// ── Equation view ─────────────────────────────────────────────────────────
function ReactionEquations({ reactions, dens }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {reactions.map((r, i) =>
      <article key={i} style={{
        background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 8,
        overflow: "hidden"
      }}>
          <header style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: `${dens.pad}px ${dens.pad + 4}px`,
          borderBottom: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper
        }}>
            <code style={{
            fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.accent,
            background: "#E8F1F8", border: `1px solid ${TOKENS.accent}33`,
            padding: "1px 6px", borderRadius: 3
          }}>{r.entry_id}</code>
            <span style={{ fontFamily: FONT_SANS, fontWeight: 600, fontSize: 14, color: TOKENS.ink, flex: 1 }}>{r.title}</span>
            {(r.products || []).map((p, j) => <YieldBadge key={j} value={p.yield_pct} note={p.yield_note} big />)}
          </header>
          <EquationRow rxn={r} dens={dens} />
          {r.notes &&
        <p style={{ margin: 0, padding: `0 ${dens.pad + 4}px ${dens.pad}px`, fontFamily: FONT_SERIF, fontSize: 13, color: TOKENS.ink2, fontStyle: "italic" }}>{r.notes}</p>
        }
        </article>
      )}
    </div>);

}

// Shared equation row
function EquationRow({ rxn, dens }) {
  const tileW = 168;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: dens.pad + 2, overflowX: "auto"
    }}>
      {(rxn.reactants || []).map((m, i) =>
      <Fragment key={i}>
          {i > 0 && <PlusSep />}
          <MolTile mol={m} kind="reactant" dens={dens} compactWidth={tileW} />
        </Fragment>
      )}
      {(!rxn.reactants || rxn.reactants.length === 0) && <EmptyTile dens={dens} w={tileW} />}

      <ArrowBlock rxn={rxn} />

      {(rxn.products || []).map((m, i) =>
      <Fragment key={i}>
          {i > 0 && <PlusSep />}
          <MolTile mol={m} kind="product" dens={dens} compactWidth={tileW} />
        </Fragment>
      )}
      {(!rxn.products || rxn.products.length === 0) && <EmptyTile dens={dens} w={tileW} />}
    </div>);

}
function PlusSep() {
  return <span style={{ fontSize: 22, color: TOKENS.muted, fontWeight: 300, padding: "0 4px" }}>+</span>;
}
function EmptyTile({ dens, w }) {
  return (
    <div style={{ width: w, height: dens.tileH + 64, display: "grid", placeItems: "center", color: TOKENS.muted, fontFamily: FONT_MONO, fontSize: 11, border: `1px dashed ${TOKENS.rule}`, borderRadius: 6 }}>(none)</div>);

}
function ArrowBlock({ rxn }) {
  const top = (rxn.reagents || []).map((r) => `${r.label ? r.label + ": " : ""}${r.name}${r.loading ? ` (${r.loading})` : ""}`).join(", ");
  const c = rxn.conditions || {};
  const bottom = [c.solvent, c.temperature, c.time, c.atmosphere].filter(Boolean).join(" · ");
  return (
    <div style={{ flex: "0 0 auto", minWidth: 170, maxWidth: 240, display: "flex", flexDirection: "column", alignItems: "center", gap: 2, padding: "0 4px" }}>
      <div style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.ink, textAlign: "center", lineHeight: 1.3, wordBreak: "break-word" }}>{top}</div>
      <div style={{ width: "100%", height: 14 }}><Icon.arrow /></div>
      <div style={{ fontFamily: FONT_SANS, fontSize: 11, color: TOKENS.muted, textAlign: "center", lineHeight: 1.3, wordBreak: "break-word" }}>{bottom}</div>
    </div>);

}

Object.assign(window, { WorkbenchApp });