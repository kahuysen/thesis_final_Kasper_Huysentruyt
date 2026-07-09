// SSE-driven hook: manages one extraction run's lifecycle.
//
// phase:
//   idle       — no run yet
//   uploading  — POST /api/runs in flight
//   streaming  — SSE open, agent emitting events
//   done       — final `done` event received
//   error      — backend or transport failure
//
// Events from /api/runs/{id}/events:
//   step_start → {event, n, tool, args_preview}
//   step_done  → {event, n, tool, ok, summary}
//   complete   → {event, extraction, metadata}
//   done       → {event, extraction, csv, n_reactions, metadata}
//   error      → {event, message}

import { useCallback, useEffect, useRef, useState } from 'react';
import { createRun, eventsUrl, fetchExtraction } from './api.js';

export function useRun() {
  const [phase, setPhase] = useState('idle');
  const [runId, setRunId] = useState(null);
  const [agentLog, setAgentLog] = useState([]);
  const [extraction, setExtraction] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [inputName, setInputName] = useState(null);

  const esRef = useRef(null);

  const closeStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  useEffect(() => () => closeStream(), [closeStream]);

  const start = useCallback(async (file, { model, provider } = {}) => {
    closeStream();
    setAgentLog([]);
    setExtraction(null);
    setMetadata(null);
    setErrorMessage(null);
    setPhase('uploading');

    let runResp;
    try {
      runResp = await createRun(file);
    } catch (err) {
      setErrorMessage(String(err.message || err));
      setPhase('error');
      return null;
    }

    const id = runResp.run_id;
    setRunId(id);
    setInputName(runResp.input || null);
    setPhase('streaming');

    const es = new EventSource(eventsUrl(id, { model, provider }));
    esRef.current = es;

    es.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch { return; }
      handleEvent(ev);
    };
    es.onerror = () => {
      // EventSource fires `error` on normal close too, so only flag if we
      // haven't already moved to done/error.
      setPhase(p => (p === 'streaming' ? 'error' : p));
      setErrorMessage(prev => prev || 'Stream closed unexpectedly.');
      closeStream();
    };

    return id;

    function handleEvent(ev) {
      const kind = ev.event;
      if (kind === 'step_start') {
        setAgentLog(log => [...log, {
          n: ev.n,
          tool: ev.tool,
          args_preview: ev.args_preview || '',
          summary: ev.args_preview || '',
          ok: null,
          pending: true,
        }]);
      } else if (kind === 'step_done') {
        setAgentLog(log => log.map(row =>
          row.n === ev.n ? { ...row, ok: !!ev.ok, summary: ev.summary || row.summary, pending: false } : row
        ));
      } else if (kind === 'complete') {
        setExtraction(ev.extraction);
        if (ev.metadata) setMetadata(ev.metadata);
      } else if (kind === 'done') {
        if (ev.extraction && typeof ev.extraction === 'object') setExtraction(ev.extraction);
        if (ev.metadata) setMetadata(ev.metadata);
        setPhase('done');
        closeStream();
      } else if (kind === 'error') {
        setErrorMessage(ev.message || 'Unknown error');
        setPhase('error');
        closeStream();
      }
    }
  }, [closeStream]);

  const reset = useCallback(() => {
    closeStream();
    setPhase('idle');
    setRunId(null);
    setAgentLog([]);
    setExtraction(null);
    setMetadata(null);
    setErrorMessage(null);
    setInputName(null);
  }, [closeStream]);

  // Re-open a previously completed run by id. Skips the SSE stream and
  // loads the persisted extraction.json directly. Agent log is unavailable
  // (we don't persist it), so it stays empty.
  const openExisting = useCallback(async (runId, { inputName } = {}) => {
    closeStream();
    setPhase('uploading');
    setAgentLog([]);
    setExtraction(null);
    setMetadata(null);
    setErrorMessage(null);

    try {
      const ex = await fetchExtraction(runId);
      setRunId(runId);
      setInputName(inputName || null);
      setExtraction(ex);
      setMetadata(null);
      setPhase('done');
      return runId;
    } catch (err) {
      setErrorMessage(`Couldn't open run ${runId}: ${err.message || err}`);
      setPhase('error');
      return null;
    }
  }, [closeStream]);

  return { phase, runId, inputName, agentLog, extraction, metadata, errorMessage, start, reset, openExisting };
}
