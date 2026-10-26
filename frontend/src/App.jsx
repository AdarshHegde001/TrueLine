import { useEffect, useMemo, useState } from 'react';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';
const emptyKnowledge = { file: null, category: 'rulebook', brandId: 'brand_lumina', title: '', ruleData: '' };

function Icon({ children }) { return <span className="icon" aria-hidden="true">{children}</span>; }

function StatusPill({ status }) {
  const kind = status === 'ready' || status === 'complete' ? 'good'
    : status === 'error' ? 'bad' : 'pending';
  return <span className={`status-pill ${kind}`}><i />{status}</span>;
}

function App() {
  const [activeTab, setActiveTab] = useState('knowledge');
  const [health, setHealth] = useState(null);
  const [knowledge, setKnowledge] = useState(emptyKnowledge);
  const [reviewFile, setReviewFile] = useState(null);
  const [reviewBrand, setReviewBrand] = useState('brand_lumina');
  const [channel, setChannel] = useState('advertisement');
  const [uploads, setUploads] = useState([]);
  const [reviewResult, setReviewResult] = useState(null);
  const [knowledgeBusy, setKnowledgeBusy] = useState(false);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [toast, setToast] = useState('');

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const response = await fetch(`${API}/api/health`);
        const data = await response.json();
        if (alive) setHealth(data);
      } catch {
        if (alive) setHealth({ status: 'offline' });
      }
    };
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  const counts = useMemo(() => ({
    rulebooks: uploads.filter((item) => item.category === 'rulebook').length,
    resources: uploads.filter((item) => item.category === 'resource').length,
  }), [uploads]);

  function notify(message) {
    setToast(message);
    window.setTimeout(() => setToast(''), 4200);
  }

  async function submitKnowledge(event) {
    event.preventDefault();
    if (!knowledge.file) return notify('Choose a rulebook or resource file first.');
    setKnowledgeBusy(true);
    setToast('');
    try {
      const body = new FormData();
      body.append('file', knowledge.file);
      body.append('brandId', knowledge.brandId);
      body.append('title', knowledge.title || knowledge.file.name.replace(/\.[^.]+$/, ''));
      body.append('category', knowledge.category);
      body.append('ruleData', knowledge.ruleData);
      body.append('approved', 'true');
      const response = await fetch(`${API}/api/knowledge`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not ingest this source.');
      setUploads((items) => [{ ...data.source, category: knowledge.category, addedAt: new Date() }, ...items]);
      setKnowledge({ ...emptyKnowledge, brandId: knowledge.brandId });
      notify('Source parsed, stored, chunked, and indexed.');
    } catch (error) {
      notify(error.message);
    } finally {
      setKnowledgeBusy(false);
    }
  }

  async function submitReview(event) {
    event.preventDefault();
    if (!reviewFile) return notify('Choose a document to verify first.');
    setReviewBusy(true);
    setReviewResult(null);
    setToast('');
    try {
      const body = new FormData();
      body.append('file', reviewFile);
      body.append('brandId', reviewBrand);
      body.append('channel', channel);
      const response = await fetch(`${API}/api/review`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Review preparation failed.');
      setReviewResult(data);
    } catch (error) {
      notify(error.message);
    } finally {
      setReviewBusy(false);
    }
  }

  const deterministicFindings = reviewResult?.preparedReview?.deterministic?.findings || [];
  const aiStatus = reviewResult?.aiStatus;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup"><div className="brand-mark">T<span>.</span></div><span>trueline</span></div>
        <div className="workspace-label">WORKSPACE</div>
        <div className="workspace-card"><div className="workspace-avatar">L</div><div><b>Lumina Studio</b><small>Brand governance</small></div><span className="chevron">⌄</span></div>
        <div className="nav-label">REVIEW WORKSPACE</div>
        <button className={`nav-item ${activeTab === 'knowledge' ? 'selected' : ''}`} onClick={() => setActiveTab('knowledge')}>
          <Icon>▤</Icon> Knowledge library <span className="nav-count">{uploads.length}</span>
        </button>
        <button className={`nav-item ${activeTab === 'review' ? 'selected' : ''}`} onClick={() => setActiveTab('review')}>
          <Icon>◉</Icon> Document review
        </button>
        <div className="sidebar-bottom">
          <div className="system-card"><div className="system-head"><span className="pulse-dot" /> SYSTEM STATUS</div>
            <div className="system-row"><span>API + RAG worker</span><StatusPill status={health?.status || 'connecting'} /></div>
            <div className="system-row"><span>Gemini</span><StatusPill status={health?.geminiConfigured ? 'ready' : 'not configured'} /></div>
            {health?.embedder && <small className="embedder-note">Embedder · {health.embedder}</small>}
          </div>
          <div className="profile"><div className="profile-avatar">AH</div><div><b>Workspace admin</b><small>Local prototype</small></div><span className="more">···</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> {activeTab === 'knowledge' ? 'Knowledge library' : 'Document review'}</div><div className="top-actions"><span className="environment-label"><i /> LOCAL ENVIRONMENT</span><button className="help-button" title="Help">?</button></div></header>

        {activeTab === 'knowledge' ? (
          <section className="page-content">
            <div className="eyebrow"><span className="eyebrow-line" /> BRAND KNOWLEDGE</div>
            <div className="page-heading-row"><div><h1>Knowledge library</h1><p className="subtitle">Give Trueline the rules and references it needs to review your content.</p></div><button className="text-button" onClick={() => setActiveTab('review')}><span>Open document review</span><b>↗</b></button></div>

            <div className="stat-grid">
              <div className="stat-card"><div className="stat-icon violet">▤</div><div><span>RULEBOOKS</span><b>{counts.rulebooks}</b></div><small>approved sources</small></div>
              <div className="stat-card"><div className="stat-icon blue">◈</div><div><span>RESOURCES</span><b>{counts.resources}</b></div><small>approved sources</small></div>
              <div className="stat-card"><div className="stat-icon green">⌕</div><div><span>RETRIEVAL INDEX</span><b>{uploads.length ? 'Ready' : 'Waiting'}</b></div><small>Postgres + pgvector</small></div>
            </div>

            <div className="section-title-row"><div><h2>Add a knowledge source</h2><p>Upload a rulebook or a supporting resource for this brand.</p></div><span className="step-label"><b>01</b> INGEST</span></div>
            <div className="content-grid">
              <form className="panel upload-panel" onSubmit={submitKnowledge}>
                <div className="form-grid">
                  <label className="field"><span>Source type</span><select value={knowledge.category} onChange={(e) => setKnowledge({ ...knowledge, category: e.target.value, ruleData: e.target.value === 'resource' ? '' : knowledge.ruleData })}><option value="rulebook">Rulebook</option><option value="resource">Resource / approved facts</option></select></label>
                  <label className="field"><span>Brand ID</span><input required value={knowledge.brandId} onChange={(e) => setKnowledge({ ...knowledge, brandId: e.target.value })} placeholder="brand_lumina" /></label>
                </div>
                <label className="field"><span>Source title <em>Optional</em></span><input value={knowledge.title} onChange={(e) => setKnowledge({ ...knowledge, title: e.target.value })} placeholder="e.g. Visual identity guidelines" /></label>
                <label className="dropzone" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); const file = e.dataTransfer.files?.[0]; if (file) setKnowledge({ ...knowledge, file }); }}>
                  <input type="file" accept=".txt,.md,.pdf,.docx" onChange={(e) => setKnowledge({ ...knowledge, file: e.target.files?.[0] || null })} />
                  <span className="drop-icon">↑</span><b>{knowledge.file ? knowledge.file.name : 'Choose a file or drag it here'}</b><small>PDF, DOCX, TXT or Markdown · up to 15 MB</small>
                </label>

                {knowledge.category === 'rulebook' && <label className="field rule-data-field"><span>Structured deterministic rules <em>Optional</em></span><textarea rows="5" value={knowledge.ruleData} onChange={(e) => setKnowledge({ ...knowledge, ruleData: e.target.value })} placeholder={'Paste confirmed rules as a JSON array. These are applied by deterministic checks.\n\n[{"rule_id":"rule_1","kind":"terminology","mode":"replace_with_preferred","term":"Tru Line","replacement":"Trueline"}]'} /></label>}
                <div className="form-foot"><div className="approval-note"><span className="checkmark">✓</span><span><b>Approved source</b><small>This upload will be indexed for the selected brand.</small></span></div><button className="primary-button" type="submit" disabled={knowledgeBusy || !health || health.status !== 'ready'}>{knowledgeBusy ? <><span className="spinner" /> Indexing…</> : <>Add to knowledge base <b>→</b></>}</button></div>
                <p className="fine-print">Rulebook prose is searchable, but deterministic checks only run on rules supplied in the structured field above.</p>
              </form>
              <aside className="info-panel"><div className="info-icon">✳</div><h3>What happens after upload?</h3><ol><li><b>Text is extracted</b><span>PDF, DOCX and text files are parsed with source metadata.</span></li><li><b>Source is versioned</b><span>The original text and approved rule data are saved in PostgreSQL.</span></li><li><b>Search index is prepared</b><span>Text is chunked, embedded and made available to hybrid retrieval.</span></li></ol><div className="info-note"><b>Rulebook tip</b><p>For exact word and required-phrase checks, include reviewed rules in the structured rules field. Natural-language rules remain available as retrieval evidence.</p></div></aside>
            </div>

            <div className="section-title-row library-title"><div><h2>Recently added</h2><p>Your sources for this session.</p></div><span className="muted-label">{uploads.length} SOURCE{uploads.length === 1 ? '' : 'S'}</span></div>
            <div className="panel table-panel">{uploads.length ? <div className="source-list">{uploads.map((item) => <div className="source-row" key={item.version_id}><div className={`source-file-icon ${item.category}`}>{item.category === 'rulebook' ? '§' : '◈'}</div><div className="source-name"><b>{item.title}</b><small>{item.category === 'rulebook' ? 'Rulebook' : 'Resource'} · v{item.version} · {item.chunk_count} chunks</small></div><span className="rule-count">{item.rule_count ? `${item.rule_count} rules` : 'No structured rules'}</span><StatusPill status={item.status} /></div>)}</div> : <div className="empty-state"><div className="empty-icon">▤</div><b>No sources added yet</b><p>Upload a rulebook or supporting resource to prepare this brand’s search index.</p></div>}</div>
          </section>
        ) : (
          <section className="page-content review-page">
            <div className="eyebrow"><span className="eyebrow-line" /> CONTENT ASSURANCE</div>
            <div className="page-heading-row"><div><h1>Review a document</h1><p className="subtitle">Run deterministic checks, retrieve relevant evidence, then send the assembled context to Gemini.</p></div><span className="step-label"><b>02</b> REVIEW</span></div>
            <div className="review-layout">
              <form className="panel review-form" onSubmit={submitReview}>
                <div className="panel-heading"><div className="panel-step">1</div><div><h3>Choose the document</h3><p>We’ll parse it and locate relevant approved sources.</p></div></div>
                <label className="dropzone review-drop" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); setReviewFile(e.dataTransfer.files?.[0] || null); }}><input type="file" accept=".txt,.md,.pdf,.docx" onChange={(e) => setReviewFile(e.target.files?.[0] || null)} /><span className="drop-icon">↑</span><b>{reviewFile ? reviewFile.name : 'Choose a document to verify'}</b><small>PDF, DOCX, TXT or Markdown · up to 15 MB</small></label>
                <div className="form-grid review-fields"><label className="field"><span>Brand ID</span><input required value={reviewBrand} onChange={(e) => setReviewBrand(e.target.value)} /></label><label className="field"><span>Channel <em>Optional</em></span><select value={channel} onChange={(e) => setChannel(e.target.value)}><option value="advertisement">Advertisement</option><option value="social">Social media</option><option value="website">Website</option><option value="email">Email</option><option value="">No channel</option></select></label></div>
                <div className="review-pipeline"><div className="pipeline-title">REVIEW PIPELINE</div><div className="pipeline-step"><span>✓</span><div><b>Parse document</b><small>Extract text and passage locations</small></div></div><div className="pipeline-step"><span>2</span><div><b>Deterministic checks + retrieval</b><small>Apply structured rules and search approved sources</small></div></div><div className="pipeline-step"><span>3</span><div><b>Gemini context review</b><small>Review the assembled evidence-backed context</small></div></div></div>
                <button className="primary-button review-submit" disabled={reviewBusy || !health || health.status !== 'ready'} type="submit">{reviewBusy ? <><span className="spinner" /> Preparing review…</> : <>Run review <b>→</b></>}</button>
              </form>
              <aside className="review-side"><div className="review-side-card"><div className="side-card-icon">✧</div><div><b>Evidence-led review</b><p>Retrieved content is treated as evidence, not instructions. Findings keep links to the exact source version used.</p></div></div><div className="review-side-card quiet"><div className="side-card-icon">⚡</div><div><b>Prototype mode</b><p>AI output is a smoke test. It is not a final compliance decision or a substitute for human approval.</p></div></div></aside>
            </div>

            {reviewResult && <div className="results-section">
              <div className="results-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> REVIEW OUTPUT</div><h2>Review results</h2></div><StatusPill status={aiStatus || 'complete'} /></div>
              <div className="results-grid">
                <div className="panel result-panel"><div className="result-panel-heading"><div><span className="result-index">A</span><h3>Deterministic checks</h3></div><span className="result-count">{deterministicFindings.length} FINDING{deterministicFindings.length === 1 ? '' : 'S'}</span></div>
                  {deterministicFindings.length ? <div className="finding-list">{deterministicFindings.map((finding) => <div className="finding" key={finding.finding_id}><div className={`finding-tag ${finding.label}`}>{finding.label.replace('_', ' ')}</div><div><b>{finding.rule_id}</b><p>{finding.explanation}</p>{finding.quote && <blockquote>“{finding.quote}”</blockquote>}<small>Source: {finding.source?.title || finding.source?.source_id} · {finding.source?.version_id}</small></div></div>)}</div> : <div className="no-findings"><span>✓</span><div><b>No deterministic findings</b><p>{reviewResult.preparedReview.deterministic.checks_run?.length ? 'Configured checks passed for the submitted text.' : 'No structured rules were configured for this brand and channel.'}</p></div></div>}
                  <div className="result-meta">Checks run: {reviewResult.preparedReview.deterministic.checks_run?.join(', ') || 'none'} · Retrieval: {reviewResult.preparedReview.retrieval_status}</div>
                </div>
                <div className="panel result-panel ai-panel"><div className="result-panel-heading"><div><span className="result-index ai-index">✦</span><h3>Gemini review</h3></div><span className="result-count">{aiStatus?.toUpperCase()}</span></div>
                  {reviewResult.aiReview ? <div className="ai-response">{reviewResult.aiReview}</div> : <div className="ai-empty"><div className="ai-empty-icon">✦</div><b>{aiStatus === 'error' ? 'Gemini request failed' : 'Gemini is not configured'}</b><p>{reviewResult.aiError || 'Add your API key to backend/.env and restart the backend to enable the smoke test.'}</p></div>}
                </div>
              </div>
              <details className="context-details"><summary>Inspect the context sent for AI review</summary><pre>{JSON.stringify(reviewResult.preparedReview.ai_context, null, 2)}</pre></details>
            </div>}
          </section>
        )}
      </main>
      {toast && <div className="toast"><span>!</span>{toast}</div>}
    </div>
  );
}

export default App;
