// PDF progress modal — three phases:
//
//   1. detecting   — VisualHeist runs; figures populate as they're found.
//                    Read-only.
//   2. review      — detection complete. User checks the figures to extract,
//                    then clicks "Extract N selected".
//   3. extracting  — submitted figures stream child_* events. Unsubmitted
//                    figures are greyed out ("skipped"). Click a selected
//                    figure to open its child run in the Workbench.

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  TOKENS, FONT_SANS, FONT_SERIF, FONT_MONO, Icon, Button,
} from './shared.jsx';
import {
  fileUrl, pdfEventsUrl, pdfFileUrl, fetchPdfJob, submitPdfFigures,
} from './api.js';

export function PdfProgressModal({
  jobId, pdfName, onPickRun, onClose,
  extractModel, extractProvider,
}) {
  const [phase, setPhase]       = useState('detecting');  // detecting|review|extracting|complete|error
  const [totalPages, setTotalPages] = useState(null);
  const [pagesDone, setPagesDone]   = useState(0);
  const [figures, setFigures]   = useState([]);           // [{figure_id, page, figure_index, bbox, status, child_run_id?, step?, n_reactions?, error?}]
  const [selected, setSelected] = useState(new Set());    // figure_ids the user has checked
  const [errMsg, setErrMsg]     = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitErr, setSubmitErr] = useState(null);
  const [summary, setSummary]   = useState(null);
  const [folderName, setFolderName] = useState(() => defaultFolderName(pdfName));
  const [filedAs, setFiledAs]   = useState(null);          // name actually stored after submit
  const esRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(pdfEventsUrl(jobId));
    esRef.current = es;
    es.onmessage = (msg) => {
      let ev; try { ev = JSON.parse(msg.data); } catch { return; }
      handleEvent(ev);
    };
    es.onerror = () => {
      es.close();
      esRef.current = null;
      fetchPdfJob(jobId).then(state => {
        if (state.status === 'complete' || state.status === 'complete_with_errors') {
          setPhase('complete');
        } else if (state.status === 'error') {
          setPhase('error');
        }
      }).catch(() => {});
    };
    return () => { try { es.close(); } catch {} };

    function handleEvent(ev) {
      switch (ev.event) {
        case 'start':
          setTotalPages(ev.total_pages || 0);
          break;
        case 'page_done':
          setPagesDone(p => Math.max(p, ev.page || 0));
          break;
        case 'figure_detected':
          setFigures(prev => prev.find(f => f.figure_id === ev.figure_id)
            ? prev
            : [...prev, {
                figure_id:    ev.figure_id,
                page:         ev.page,
                figure_index: ev.figure_index,
                bbox:         ev.bbox,
                path:         ev.path || `figures/page_${String(ev.page).padStart(3,'0')}_fig_${String(ev.figure_index).padStart(2,'0')}.png`,
                status:       'detected',
              }]);
          break;
        case 'detection_complete':
          setPhase(p => (p === 'detecting' ? 'review' : p));
          if (Array.isArray(ev.figures) && ev.figures.length) {
            setFigures(prev => mergeFigures(prev, ev.figures));
          }
          // Pre-select all detected figures by default.
          setSelected(prev => {
            if (prev.size > 0) return prev;
            const s = new Set();
            (ev.figures || []).forEach(f => s.add(f.figure_id));
            return s;
          });
          break;
        case 'folder_created':
          setFiledAs(ev.folder_name || null);
          break;
        case 'submit_started':
          setPhase('extracting');
          break;
        case 'figure_selected':
          patch(ev.figure_id, { status: 'queued', child_run_id: ev.child_run_id, step: 0 });
          break;
        case 'child_started':
          patchByChild(ev.child_run_id, { status: 'running' });
          break;
        case 'child_progress':
          patchByChild(ev.child_run_id, prev => ({ status: 'running', step: (prev.step || 0) + 1, last_tool: ev.tool }));
          break;
        case 'child_complete':
          patchByChild(ev.child_run_id, { status: 'done', n_reactions: ev.n_reactions });
          break;
        case 'child_error':
          patchByChild(ev.child_run_id, { status: 'error', error: ev.message });
          break;
        case 'complete':
          setSummary({ n_figures: ev.n_figures, n_done: ev.n_done, n_failed: ev.n_failed });
          setPhase('complete');
          break;
        case 'error':
          setErrMsg(ev.message || 'PDF processing failed');
          setPhase('error');
          break;
        case 'state':
          if (ev.state) hydrateFromState(ev.state);
          break;
      }
    }
    function patch(figure_id, p) {
      setFigures(prev => prev.map(f => f.figure_id === figure_id
        ? { ...f, ...(typeof p === 'function' ? p(f) : p) } : f));
    }
    function patchByChild(child_run_id, p) {
      setFigures(prev => prev.map(f => f.child_run_id === child_run_id
        ? { ...f, ...(typeof p === 'function' ? p(f) : p) } : f));
    }
    function hydrateFromState(s) {
      const figs = s.figures || [];
      const childByFig = Object.fromEntries((s.child_runs || []).map(cr => [cr.figure_id, cr]));
      setFigures(figs.map(f => {
        const cr = childByFig[f.figure_id];
        if (!cr) return { ...f, status: 'detected' };
        return {
          ...f,
          child_run_id: cr.child_run_id,
          status:       cr.status || 'queued',
          n_reactions:  cr.n_reactions,
          error:        cr.error,
        };
      }));
      setTotalPages(s.total_pages || null);
      if (s.status === 'awaiting_selection') setPhase('review');
      else if (s.status === 'extracting')     setPhase('extracting');
      else if (s.status === 'complete' || s.status === 'complete_with_errors') setPhase('complete');
      else if (s.status === 'error') setPhase('error');
    }
  }, [jobId]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // ---- selection helpers ----

  function toggle(figId) {
    setSelected(prev => {
      const s = new Set(prev);
      s.has(figId) ? s.delete(figId) : s.add(figId);
      return s;
    });
  }
  function selectAll()  { setSelected(new Set(figures.map(f => f.figure_id))); }
  function selectNone() { setSelected(new Set()); }

  async function submit() {
    if (phase !== 'review' || selected.size === 0) return;
    setSubmitting(true); setSubmitErr(null);
    try {
      await submitPdfFigures(jobId, [...selected], {
        model:      extractModel,
        provider:   extractProvider,
        folderName: folderName.trim() || undefined,
      });
      // The SSE stream will emit folder_created + submit_started + figure_selected;
      // we don't need to advance phase here.
    } catch (e) {
      setSubmitErr(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  }

  // ---- header text ----

  const phaseLabel = useMemo(() => {
    if (phase === 'detecting' && pagesDone === 0 && figures.length === 0)
      return "Loading VisualHeist model… (~30 s the first time)";
    if (phase === 'detecting')
      return `Detecting figures…  pages ${pagesDone}${totalPages ? `/${totalPages}` : ''}  ·  found ${figures.length}`;
    if (phase === 'review')
      return `Review detected figures — ${figures.length} found · ${selected.size} selected`;
    if (phase === 'extracting')
      return `Extracting ${figures.filter(f => f.child_run_id).length} figures…`;
    if (phase === 'complete') {
      const done   = summary?.n_done   ?? figures.filter(f => f.status === 'done').length;
      const failed = summary?.n_failed ?? figures.filter(f => f.status === 'error').length;
      return `Done — ${done} succeeded, ${failed} failed.`;
    }
    if (phase === 'error') return `Error: ${errMsg || "see logs"}`;
    return "";
  }, [phase, pagesDone, totalPages, figures, selected, summary, errMsg]);

  const isReview     = phase === 'review';
  const isExtracting = phase === 'extracting' || phase === 'complete';

  return (
    <div onClick={onClose} style={{
      position: "absolute", inset: 0, zIndex: 140,
      background: "rgba(15,26,36,0.42)",
      display: "grid", placeItems: "center", padding: 24,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: "min(1080px, 100%)", height: "min(760px, 92vh)",
        background: "white", border: `1px solid ${TOKENS.rule}`, borderRadius: 10,
        boxShadow: "0 16px 48px rgba(15,26,36,0.18)",
        display: "grid", gridTemplateRows: "auto auto 1fr auto", overflow: "hidden",
      }}>
        <header style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "14px 18px", borderBottom: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper,
        }}>
          <span style={{
            width: 26, height: 26, borderRadius: 4,
            background: TOKENS.accent, color: "white",
            display: "grid", placeItems: "center",
            fontFamily: FONT_SANS, fontWeight: 700, fontSize: 11, letterSpacing: 0.4,
          }}>PDF</span>
          <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
            <span style={{ fontFamily: FONT_SANS, fontWeight: 700, fontSize: 14, color: TOKENS.ink, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {pdfName || "Document"}
            </span>
            <span style={{ fontFamily: FONT_MONO, fontSize: 10.5, color: TOKENS.muted }}>
              job · {jobId}
            </span>
          </div>
          <div style={{ flex: 1 }}/>
          <button onClick={onClose} style={{
            width: 26, height: 26, padding: 0, background: "transparent",
            border: `1px solid ${TOKENS.rule}`, borderRadius: 4, cursor: "pointer",
            fontFamily: FONT_SANS, fontWeight: 700, color: TOKENS.muted, fontSize: 14,
          }}>×</button>
        </header>

        <div style={{
          display: "flex", alignItems: "center", gap: 12,
          padding: "10px 18px", borderBottom: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper,
        }}>
          <PhaseDot phase={phase}/>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontFamily: FONT_SANS, fontSize: 13, fontWeight: 600, color: phase === 'error' ? TOKENS.err : TOKENS.ink }}>
              {phaseLabel}
            </div>
          </div>
          <FolderField
            value={folderName}
            onChange={setFolderName}
            editable={phase === 'detecting' || phase === 'review'}
            filedAs={filedAs}
          />
          {isReview && (
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <button onClick={selectAll}  style={smallLink}>Select all</button>
              <button onClick={selectNone} style={smallLink}>Select none</button>
              <Button primary disabled={submitting || selected.size === 0}
                      onClick={submit}>
                {submitting
                  ? "Submitting…"
                  : `Extract ${selected.size} selected`}
              </Button>
            </div>
          )}
        </div>

        <div style={{ overflowY: "auto", padding: 14 }}>
          {submitErr && (
            <div style={{
              padding: 10, borderRadius: 6, marginBottom: 10,
              background: TOKENS.err2, border: `1px solid ${TOKENS.err}55`,
              color: TOKENS.err, fontFamily: FONT_SANS, fontSize: 12.5,
            }}>{submitErr}</div>
          )}

          {figures.length === 0 ? (
            <div style={{
              padding: 64, textAlign: "center",
              color: TOKENS.muted, fontFamily: FONT_SERIF, fontStyle: "italic",
            }}>
              {phase === 'error'
                ? "Aborted before any figures were detected."
                : phase === 'complete'
                  ? "No figures detected in this PDF."
                  : "Waiting for the first detected figure…"}
            </div>
          ) : (
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 12,
            }}>
              {figures.map(f => (
                <FigureCard
                  key={f.figure_id}
                  fig={f}
                  jobId={jobId}
                  isReview={isReview}
                  isExtracting={isExtracting}
                  selected={selected.has(f.figure_id)}
                  onToggle={() => toggle(f.figure_id)}
                  onOpenChild={() => f.child_run_id && onPickRun(f.child_run_id)}
                />
              ))}
            </div>
          )}
        </div>

        <footer style={{
          padding: "10px 18px", borderTop: `1px solid ${TOKENS.rule}`,
          background: TOKENS.paper,
          fontFamily: FONT_MONO, fontSize: 11, color: TOKENS.muted,
          display: "flex", justifyContent: "space-between",
        }}>
          <span>
            {isReview     ? "Tick the figures you want to extract, then submit." :
             isExtracting ? "Closing keeps extractions running in the background." :
                            "Closing keeps detection running in the background."}
          </span>
          <span>esc to close</span>
        </footer>
      </div>
    </div>
  );
}

function mergeFigures(prev, incoming) {
  const byId = new Map(prev.map(f => [f.figure_id, f]));
  for (const f of incoming) {
    const existing = byId.get(f.figure_id);
    byId.set(f.figure_id, { ...(existing || {}), ...f, status: existing?.status || 'detected' });
  }
  return [...byId.values()].sort((a, b) =>
    (a.page - b.page) || (a.figure_index - b.figure_index));
}

function FigureCard({ fig, jobId, isReview, isExtracting, selected, onToggle, onOpenChild }) {
  // Thumbnail source: prefer the child run's input.png once available, else
  // the raw crop from the PDF job's figures/ dir.
  const thumb = fig.child_run_id
    ? fileUrl(fig.child_run_id, 'input.png')
    : pdfFileUrl(jobId, fig.path || `figures/page_${String(fig.page).padStart(3,'0')}_fig_${String(fig.figure_index).padStart(2,'0')}.png`);
  const isSkipped = isExtracting && !fig.child_run_id;
  const isClickable = isExtracting && fig.child_run_id && fig.status !== 'queued';

  function handleClick() {
    if (isReview) onToggle();
    else if (isClickable) onOpenChild();
  }

  const statusColor = {
    detected: TOKENS.muted,
    queued:   TOKENS.muted,
    running:  TOKENS.warn,
    done:     TOKENS.pos,
    error:    TOKENS.err,
  }[fig.status] || TOKENS.muted;

  return (
    <div onClick={handleClick} style={{
      textAlign: "left", padding: 0,
      background: "white",
      border: `1px solid ${selected && isReview ? TOKENS.accent : TOKENS.rule}`,
      borderRadius: 8,
      boxShadow: selected && isReview ? `0 0 0 2px ${TOKENS.accent}33` : "none",
      cursor: (isReview || isClickable) ? "pointer" : "default",
      display: "flex", flexDirection: "column", overflow: "hidden",
      opacity: isSkipped ? 0.4 : 1,
      transition: "border-color .1s, box-shadow .1s",
    }}>
      <div style={{
        position: "relative",
        height: 130, background: TOKENS.paper,
        display: "grid", placeItems: "center", overflow: "hidden",
        borderBottom: `1px solid ${TOKENS.rule2}`,
      }}>
        <img src={thumb} alt="" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" }}/>
        {isReview && (
          <div style={{
            position: "absolute", top: 6, left: 6,
            width: 22, height: 22, borderRadius: 4,
            background: selected ? TOKENS.accent : "rgba(255,255,255,0.92)",
            border: `1.5px solid ${selected ? TOKENS.accent : TOKENS.rule}`,
            display: "grid", placeItems: "center",
            color: "white",
          }}>
            {selected && <Icon.check/>}
          </div>
        )}
      </div>
      <div style={{ padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 8, height: 8, borderRadius: 999, background: statusColor,
            animation: fig.status === 'running' ? "rxn-pulse 1.2s ease-in-out infinite" : "none",
          }}/>
          <span style={{
            fontFamily: FONT_SANS, fontWeight: 700, fontSize: 11.5, color: TOKENS.ink2,
            textTransform: "uppercase", letterSpacing: 0.4,
          }}>{isSkipped ? "skipped" : fig.status}</span>
          <span style={{ flex: 1 }}/>
          <code style={{
            fontFamily: FONT_MONO, fontSize: 10, color: TOKENS.muted,
          }}>p.{fig.page}·{fig.figure_index}</code>
        </div>
        <div style={{ fontFamily: FONT_SANS, fontSize: 12, color: TOKENS.ink2, minHeight: 16 }}>
          {fig.status === 'done' && (
            <span><strong style={{ color: TOKENS.pos, fontFamily: FONT_MONO }}>{fig.n_reactions ?? 0}</strong> {fig.n_reactions === 1 ? "reaction" : "reactions"}</span>
          )}
          {fig.status === 'running' && (
            <span>step {fig.step || 0}{fig.last_tool ? ` · ${fig.last_tool}` : ""}</span>
          )}
          {fig.status === 'queued' && <span>queued for agent…</span>}
          {fig.status === 'error' && (
            <span style={{ color: TOKENS.err, fontFamily: FONT_MONO, fontSize: 11 }}>{(fig.error || "failed").slice(0, 80)}</span>
          )}
        </div>
      </div>
    </div>
  );
}

function PhaseDot({ phase }) {
  const color = {
    detecting:  TOKENS.warn,
    review:     TOKENS.accent,
    extracting: TOKENS.warn,
    complete:   TOKENS.pos,
    error:      TOKENS.err,
  }[phase] || TOKENS.muted;
  const pulsing = phase === 'detecting' || phase === 'extracting';
  return (
    <span style={{
      width: 12, height: 12, borderRadius: 999, background: color,
      animation: pulsing ? "rxn-pulse 1.2s ease-in-out infinite" : "none",
    }}/>
  );
}

function FolderField({ value, onChange, editable, filedAs }) {
  // Locked view (after submit): show a green chip with the actual folder name.
  if (!editable) {
    return (
      <div title={filedAs ? `Filed in folder: ${filedAs}` : "No folder"} style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        padding: "5px 10px",
        background: filedAs ? TOKENS.pos2 : TOKENS.paper,
        border: `1px solid ${filedAs ? TOKENS.pos + '55' : TOKENS.rule}`,
        color:  filedAs ? TOKENS.pos : TOKENS.muted,
        borderRadius: 6,
        fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
        maxWidth: 240, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
      }}>
        <span style={{ fontSize: 14 }}>📁</span>
        {filedAs || value || "(no folder)"}
      </div>
    );
  }
  // Editable: text input with the folder emoji prefix.
  return (
    <label title="Folder name — created when you submit" style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "4px 6px 4px 10px",
      background: "white",
      border: `1px solid ${TOKENS.rule}`, borderRadius: 6,
    }}>
      <span style={{ fontSize: 14 }}>📁</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="folder name"
        spellCheck={false}
        style={{
          border: "none", outline: "none", background: "transparent",
          fontFamily: FONT_SANS, fontSize: 12.5, fontWeight: 600,
          color: TOKENS.ink, width: 200, padding: "3px 0",
        }}
      />
    </label>
  );
}

export function defaultFolderName(pdfName) {
  if (!pdfName) return "PDF run";
  const stem = pdfName.replace(/\.pdf$/i, "");
  // Tidy: collapse underscores/dashes that look like filename noise.
  return stem.replace(/[_]+/g, " ").trim() || pdfName;
}

const smallLink = {
  background: "transparent", border: "none",
  color: TOKENS.accent, fontFamily: FONT_SANS, fontSize: 12, fontWeight: 600,
  cursor: "pointer", padding: "2px 4px",
};
