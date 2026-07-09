// Top-level wiring: pulls health, owns the run hook, transforms backend
// shapes into props the Workbench expects, and triggers Rxn-INSIGHT on demand.

import { useEffect, useMemo, useState } from 'react';
import { WorkbenchApp } from './workbench.jsx';
import { useRun } from './useRun.js';
import { adaptInsight, adaptModels, createPdfJob, fetchHealth, fetchInsight, fetchInsightArtifact, fileUrl } from './api.js';
import { PdfProgressModal } from './PdfProgressModal.jsx';

export function App() {
  const [health, setHealth] = useState(null);
  const [healthErr, setHealthErr] = useState(null);
  const [modelId, setModelId] = useState(null);
  const [insight, setInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);

  const run = useRun();

  async function refreshHealth() {
    try {
      const h = await fetchHealth();
      setHealth(h);
      if (h?.model && !modelId) setModelId(h.model);
      setHealthErr(null);
    } catch (err) {
      setHealthErr(String(err.message || err));
    }
  }

  // Load health once on mount; populate the default model.
  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then(h => {
        if (cancelled) return;
        setHealth(h);
        if (h?.model) setModelId(h.model);
      })
      .catch(err => !cancelled && setHealthErr(String(err.message || err)));
    return () => { cancelled = true; };
  }, []);

  // Reset insight whenever a new run starts.
  useEffect(() => { setInsight(null); }, [run.runId]);

  const models = useMemo(
    () => adaptModels(health?.models, health?.provider),
    [health]
  );
  const selectedModel = models.find(m => m.id === modelId) || models[0];

  const [pdfJob, setPdfJob] = useState(null);    // { jobId, pdfName } | null

  async function handleUpload(file) {
    if (!file) return;
    const isPdf = file.type === 'application/pdf' || (file.name || '').toLowerCase().endsWith('.pdf');
    if (isPdf) {
      if (!health?.pdf_available) {
        alert("PDF processing is not configured on this server (.venv-chemeagle missing).");
        return;
      }
      try {
        const job = await createPdfJob(file, { modelSize: 'large' });
        setPdfJob({ jobId: job.job_id, pdfName: job.pdf_name || file.name });
      } catch (e) {
        alert(`PDF upload failed: ${e.message || e}`);
      }
      return;
    }
    run.start(file, {
      model: selectedModel?.id,
      provider: selectedModel?.provider,
    });
  }

  async function handlePickPdfChild(childRunId) {
    setInsight(null);
    await run.openExisting(childRunId, { inputName: 'input.png' });
    setPdfJob(null);   // close the modal — the workbench now shows the child run
  }

  // Open a previously completed run from the "My runs" picker. After the
  // extraction loads, try to also rehydrate persisted Rxn-INSIGHT output.
  async function handleOpenExistingRun(item) {
    const id = await run.openExisting(item.run_id, { inputName: item.input_name });
    if (!id) return;
    if (item.has_insight) {
      const data = await fetchInsightArtifact(id);
      if (Array.isArray(data)) {
        setInsight(adaptInsight(data));
      }
    } else {
      setInsight(null);
    }
  }

  async function handleRunInsight() {
    if (!run.runId) return;
    setInsightLoading(true);
    try {
      const result = await fetchInsight(run.runId);
      if (result.ok) {
        setInsight(adaptInsight(result.analyses));
      } else {
        alert(`Rxn-INSIGHT failed: ${result.error || 'unknown error'}`);
      }
    } finally {
      setInsightLoading(false);
    }
  }

  const sourceImageUrl = run.runId && run.inputName ? fileUrl(run.runId, run.inputName) : null;

  const downloads = run.runId && run.phase === 'done'
    ? {
        csv:    fileUrl(run.runId, 'reactions.csv'),
        json:   fileUrl(run.runId, 'extraction.json'),
        source: run.inputName ? fileUrl(run.runId, run.inputName) : undefined,
        insight: insight ? fileUrl(run.runId, 'insight.csv') : undefined,
      }
    : null;

  return (
    <>
    {pdfJob && (
      <PdfProgressModal
        jobId={pdfJob.jobId}
        pdfName={pdfJob.pdfName}
        extractModel={selectedModel?.id}
        extractProvider={selectedModel?.provider}
        onPickRun={handlePickPdfChild}
        onClose={() => setPdfJob(null)}
      />
    )}
    <WorkbenchApp
      extraction={run.extraction}
      agentLog={run.agentLog}
      insight={insight}
      metadata={run.metadata}
      models={models}
      modelId={selectedModel?.id || ''}
      onModelChange={setModelId}
      sourceImageUrl={sourceImageUrl}
      runId={run.runId}
      onUploadFile={handleUpload}
      onOpenExistingRun={handleOpenExistingRun}
      onSettingsSaved={refreshHealth}
      onRunInsight={handleRunInsight}
      rxnInsightAvailable={!!health?.rxn_insight_available && !insightLoading}
      downloads={downloads}
      phase={run.phase}
      errorMessage={run.errorMessage || healthErr}
      backendInfo={health?.backend || ''}
    />
    </>
  );
}
