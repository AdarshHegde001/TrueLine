import { useEffect, useMemo, useState } from 'react';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';
const emptyKnowledge = { file: null, documentType: 'brand_guidelines', title: '', ruleData: '' };
const documentTypeLabels = {
  brand_guidelines: 'Brand guidelines',
  tone_of_voice: 'Tone of voice',
  approved_claims: 'Approved claims',
  product_facts: 'Product facts',
  glossary: 'Glossary',
  asset_rules: 'Asset rules',
  legal_compliance: 'Legal & compliance',
  other_reference: 'Other reference',
};

function StatusPill({ status }) {
  const kind = status === 'ready' || status === 'complete' ? 'good'
    : status === 'error' ? 'bad' : 'pending';
  return <span className={`status-pill ${kind}`}><i />{status}</span>;
}

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [health, setHealth] = useState(null);
  const [knowledge, setKnowledge] = useState(emptyKnowledge);
  const [reviewFile, setReviewFile] = useState(null);
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
    guidance: uploads.filter((item) => ['brand_guidelines', 'tone_of_voice', 'legal_compliance', 'glossary', 'asset_rules'].includes(item.document_type)).length,
    approved: uploads.filter((item) => ['approved_claims', 'product_facts'].includes(item.document_type)).length,
  }), [uploads]);

  function notify(message) {
    setToast(message);
    window.setTimeout(() => setToast(''), 4200);
  }

  async function submitKnowledge(event) {
    event.preventDefault();
    if (!knowledge.file) return notify('Choose a source document first.');
    setKnowledgeBusy(true);
    setToast('');
    try {
      const body = new FormData();
      body.append('file', knowledge.file);
      body.append('title', knowledge.title || knowledge.file.name.replace(/\.[^.]+$/, ''));
      body.append('documentType', knowledge.documentType);
      body.append('ruleData', knowledge.ruleData);
      body.append('approved', 'true');
      const response = await fetch(`${API}/api/knowledge`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not ingest this source.');
      setUploads((items) => [{ ...data.source, addedAt: new Date() }, ...items]);
      setKnowledge(emptyKnowledge);
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
      <header className="site-header">
        <button className="logo-button" onClick={() => setActiveTab('home')} aria-label="TrueLine home">
          <img src="/TrueLine_logo.jpg" alt="TrueLine" />
        </button>
        <nav className="main-nav" aria-label="Main navigation">
          <button className={`nav-link ${activeTab === 'home' ? 'active' : ''}`} onClick={() => setActiveTab('home')}>Home</button>
          <button className={`nav-link ${activeTab === 'knowledge' ? 'active' : ''}`} onClick={() => setActiveTab('knowledge')}>Knowledge base</button>
          <button className={`nav-link ${activeTab === 'review' ? 'active' : ''}`} onClick={() => setActiveTab('review')}>Verify a document</button>
        </nav>
      </header>

      <main className="main-content">
        {activeTab === 'home' && (
          <>
            <section className="hero-section">
              <div className="hero-copy">
                <div className="eyebrow"><span className="eyebrow-line" /> PRE-PUBLICATION BRAND GOVERNANCE</div>
                <h1>Make every draft<br />feel <em>like your brand.</em></h1>
                <p className="hero-description">TrueLine checks content against approved guidelines, facts, and brand resources. Every finding is grounded in evidence, so your team can review with clarity and keep people in control.</p>
                <div className="hero-actions">
                  <button className="primary-button" onClick={() => setActiveTab('knowledge')}>Add brand sources <b>→</b></button>
                  <button className="text-button" onClick={() => setActiveTab('review')}><span>Verify content</span><b>↗</b></button>
                </div>
                <div className="hero-principle"><span>✓</span><p><b>Cite or stay silent.</b> Unsupported claims are marked unverified, never assumed false.</p></div>
              </div>
              <div className="hero-visual">
                <div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" />
                <div className="review-preview">
                  <div className="preview-top"><span className="preview-file-icon">▤</span><span><b>Campaign draft</b><small>Evidence-led review</small></span><span className="preview-menu">•••</span></div>
                  <div className="preview-content"><span className="preview-kicker">REVIEWING A CLAIM</span><p>“Our new formula is <mark>clinically proven</mark> to work.”</p>
                    <div className="preview-finding"><span className="finding-icon">!</span><span><b>Needs supporting evidence</b><small>Check against approved claims.</small></span><span className="finding-arrow">↗</span></div>
                    <div className="preview-source"><span className="source-check">✓</span><span><small>RETRIEVED SOURCE</small><b>Claims policy · Section 3.2</b></span><span className="source-link">↗</span></div>
                  </div>
                  <div className="preview-footer"><span><i /> Source linked</span><small>Human review stays in control</small></div>
                </div>
                <div className="floating-note"><span>↗</span><span><b>Evidence first</b><small>at every step</small></span></div>
              </div>
            </section>

            <section className="principles-section page-content">
              <div className="section-intro"><div className="eyebrow"><span className="eyebrow-line" /> THE TRUE LINE APPROACH</div><h2>Review content with confidence.</h2><p>Catch inconsistencies early, see why something was flagged, and make the final decision as a person.</p></div>
              <div className="principle-grid">
                <article className="principle-card"><span className="principle-number">01</span><div className="principle-icon">⌕</div><h3>Rules before guesswork</h3><p>Deterministic checks handle approved terminology and required wording consistently.</p></article>
                <article className="principle-card"><span className="principle-number">02</span><div className="principle-icon">◈</div><h3>Evidence with each finding</h3><p>Retrieved passages link back to approved source documents and their versions.</p></article>
                <article className="principle-card"><span className="principle-number">03</span><div className="principle-icon">◎</div><h3>People stay in control</h3><p>AI can support a review, but people remain responsible for the outcome.</p></article>
              </div>
            </section>

            <section className="workflow-section">
              <div className="page-content">
                <div className="section-intro workflow-intro"><div className="eyebrow"><span className="eyebrow-line" /> HOW IT WORKS</div><h2>From approved sources to a reviewed draft.</h2><p>The platform is being built in stages. The core source and retrieval path works today; the specialist agent workflow is planned next.</p></div>
                <div className="workflow-steps">
                  <article><span className="workflow-index">01</span><span className="workflow-icon">↥</span><h3>Add knowledge</h3><p>Upload guidelines, tone rules, claims, facts, glossaries, and asset rules with a document type.</p><span className="stage-tag available">Available now</span></article>
                  <article><span className="workflow-index">02</span><span className="workflow-icon">▤</span><h3>Prepare the index</h3><p>Text is parsed, chunked, tagged, and embedded for hybrid search with PostgreSQL and pgvector.</p><span className="stage-tag available">Available now</span></article>
                  <article><span className="workflow-index">03</span><span className="workflow-icon">⌕</span><h3>Check a draft</h3><p>Deterministic rules run and relevant approved evidence is retrieved for the submitted document.</p><span className="stage-tag available">Available now</span></article>
                  <article><span className="workflow-index">04</span><span className="workflow-icon">✧</span><h3>Ask specialist agents</h3><p>Planned tone and claim agents will assess focused evidence and return source-linked findings.</p><span className="stage-tag planned">Planned</span></article>
                </div>
              </div>
            </section>

            <section className="prototype-section page-content">
              <div className="prototype-card"><div className="prototype-copy"><div className="eyebrow"><span className="eyebrow-line" /> CURRENT PROTOTYPE</div><h2>A clear picture of what works today.</h2><p>Document parsing, deterministic checks, hybrid retrieval, and context assembly are in place. For now, the review page makes one Gemini smoke-test call with that prepared context. Multi-agent LangGraph orchestration, critique, suggestions, and human approval tools are future work.</p><button className="text-button" onClick={() => setActiveTab('review')}><span>See the current review flow</span><b>→</b></button></div><div className="prototype-checklist"><div><span className="check-state done">✓</span><span><b>Core review preparation</b><small>Parsing · rules · retrieval · context</small></span></div><div><span className="check-state demo">✦</span><span><b>Gemini smoke test</b><small>Single request with prepared context</small></span></div><div><span className="check-state next">→</span><span><b>Planned next</b><small>Specialists · critique · human decisions</small></span></div></div></div>
            </section>

            <section className="closing-cta page-content"><div><div className="eyebrow"><span className="eyebrow-line" /> START WITH YOUR SOURCES</div><h2>Give every review a trusted reference.</h2></div><button className="primary-button" onClick={() => setActiveTab('knowledge')}>Open knowledge base <b>→</b></button></section>
          </>
        )}

        {activeTab === 'knowledge' ? (
          <section className="page-content">
            <div className="eyebrow"><span className="eyebrow-line" /> KNOWLEDGE BASE</div>
            <div className="page-heading-row"><div><h1>Bring your brand rules together.</h1><p className="subtitle">Upload approved sources and label what each document contains so the right evidence can be found later.</p></div><button className="text-button" onClick={() => setActiveTab('review')}><span>Verify a document</span><b>↗</b></button></div>

            <div className="stat-grid">
              <div className="stat-card"><div className="stat-icon violet">▤</div><div><span>GUIDANCE</span><b>{counts.guidance}</b></div><small>tone, brand, compliance</small></div>
              <div className="stat-card"><div className="stat-icon blue">◈</div><div><span>CLAIMS & FACTS</span><b>{counts.approved}</b></div><small>approved sources</small></div>
              <div className="stat-card"><div className="stat-icon green">⌕</div><div><span>RETRIEVAL INDEX</span><b>{uploads.length ? 'Ready' : 'Waiting'}</b></div><small>Postgres + pgvector</small></div>
            </div>

            <div className="section-title-row"><div><h2>Add a source</h2><p>Text is extracted, versioned, chunked, and indexed for retrieval.</p></div><span className="step-label"><b>01</b> INGEST</span></div>
            <div className="content-grid">
              <form className="panel upload-panel" onSubmit={submitKnowledge}>
                <div className="form-grid">
                  <label className="field"><span>Document type</span><select value={knowledge.documentType} onChange={(e) => setKnowledge({ ...knowledge, documentType: e.target.value })}>{Object.entries(documentTypeLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
                </div>
                <label className="field"><span>Source title <em>Optional</em></span><input value={knowledge.title} onChange={(e) => setKnowledge({ ...knowledge, title: e.target.value })} placeholder="e.g. Visual identity guidelines" /></label>
                <label className="dropzone" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); const file = e.dataTransfer.files?.[0]; if (file) setKnowledge({ ...knowledge, file }); }}>
                  <input type="file" accept=".txt,.md,.pdf,.docx" onChange={(e) => setKnowledge({ ...knowledge, file: e.target.files?.[0] || null })} />
                  <span className="drop-icon">↑</span><b>{knowledge.file ? knowledge.file.name : 'Choose a file or drag it here'}</b><small>PDF, DOCX, TXT or Markdown · up to 15 MB</small>
                </label>

                <label className="field rule-data-field"><span>Reviewed deterministic rules <em>Optional</em></span><textarea rows="5" value={knowledge.ruleData} onChange={(e) => setKnowledge({ ...knowledge, ruleData: e.target.value })} placeholder={'Add only rules that have been reviewed and confirmed. They will run as exact deterministic checks.\n\n[{"rule_id":"rule_1","kind":"terminology","mode":"replace_with_preferred","term":"Tru Line","replacement":"Trueline"}]'} /></label>
                <div className="form-foot"><div className="approval-note"><span className="checkmark">✓</span><span><b>Approved source</b><small>This document will be indexed for review.</small></span></div><button className="primary-button" type="submit" disabled={knowledgeBusy || !health || health.status !== 'ready'}>{knowledgeBusy ? <><span className="spinner" /> Indexing…</> : <>Add to knowledge base <b>→</b></>}</button></div>
                <p className="fine-print">The selected document type is included in the embedding input and stored as a retrieval filter. Prose is not converted into deterministic rules automatically.</p>
              </form>
              <aside className="info-panel"><div className="info-icon">✳</div><h3>What happens after upload?</h3><ol><li><b>Text is extracted</b><span>PDF, DOCX and text files are parsed with source metadata.</span></li><li><b>Source is versioned</b><span>The original text, document type, and approved rule data are saved in PostgreSQL.</span></li><li><b>Search index is prepared</b><span>Typed chunks are embedded and made available to hybrid retrieval.</span></li></ol><div className="info-note"><b>Document type tip</b><p>Choose the closest purpose. Future tone, claims, and fact agents can filter retrieval to that type.</p></div></aside>
            </div>

            <div className="section-title-row library-title"><div><h2>Recently added</h2><p>Uploads made during this session.</p></div><span className="muted-label">{uploads.length} SOURCE{uploads.length === 1 ? '' : 'S'}</span></div>
            <div className="panel table-panel">{uploads.length ? <div className="source-list">{uploads.map((item) => <div className="source-row" key={item.version_id}><div className="source-file-icon knowledge">▤</div><div className="source-name"><b>{item.title}</b><small>{documentTypeLabels[item.document_type] || 'Other reference'} · v{item.version} · {item.chunk_count} chunks</small></div><span className="rule-count">{item.rule_count ? `${item.rule_count} rules` : 'No structured rules'}</span><StatusPill status={item.status} /></div>)}</div> : <div className="empty-state"><div className="empty-icon">▤</div><b>No sources added yet</b><p>Upload approved organization guidance, facts, or claims to prepare retrieval.</p></div>}</div>
          </section>
        ) : activeTab === 'review' ? (
          <section className="page-content review-page">
            <div className="eyebrow"><span className="eyebrow-line" /> DOCUMENT REVIEW</div>
            <div className="page-heading-row"><div><h1>Check a draft before it goes out.</h1><p className="subtitle">Run the current deterministic checks and retrieval flow. Gemini is a smoke test for now, not the future agentic reviewer.</p></div><span className="step-label"><b>01</b> GEMINI SMOKE TEST</span></div>
            <div className="review-layout">
              <form className="panel review-form" onSubmit={submitReview}>
                <div className="panel-heading"><div className="panel-step">1</div><div><h3>Choose the document</h3><p>We’ll parse it and locate relevant approved sources.</p></div></div>
                <label className="dropzone review-drop" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); setReviewFile(e.dataTransfer.files?.[0] || null); }}><input type="file" accept=".txt,.md,.pdf,.docx" onChange={(e) => setReviewFile(e.target.files?.[0] || null)} /><span className="drop-icon">↑</span><b>{reviewFile ? reviewFile.name : 'Choose a document to verify'}</b><small>PDF, DOCX, TXT or Markdown · up to 15 MB</small></label>
                <div className="form-grid review-fields single-field"><label className="field"><span>Channel <em>Optional</em></span><select value={channel} onChange={(e) => setChannel(e.target.value)}><option value="advertisement">Advertisement</option><option value="social">Social media</option><option value="website">Website</option><option value="email">Email</option><option value="">No channel</option></select></label></div>
                <div className="review-pipeline"><div className="pipeline-title">CURRENT REVIEW PATH</div><div className="pipeline-step"><span>1</span><div><b>Parse the document</b><small>Extract text and passage locations</small></div></div><div className="pipeline-step"><span>2</span><div><b>Run rules + retrieve evidence</b><small>Apply structured rules and search approved sources</small></div></div><div className="pipeline-step"><span>3</span><div><b>Send context to Gemini</b><small>One basic smoke-test request</small></div></div></div>
                <button className="primary-button review-submit" disabled={reviewBusy || !health || health.status !== 'ready'} type="submit">{reviewBusy ? <><span className="spinner" /> Preparing review…</> : <>Run review <b>→</b></>}</button>
              </form>
              <aside className="review-side"><div className="review-side-card"><div className="side-card-icon">✧</div><div><b>Evidence-led review</b><p>Retrieved passages are treated as evidence, not instructions, and stay linked to their source version.</p></div></div><div className="review-side-card quiet"><div className="side-card-icon">⚡</div><div><b>Prototype mode</b><p>This single Gemini call is a smoke test, not a final compliance decision or the planned multi-agent workflow.</p></div></div></aside>
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
        ) : null}
      </main>
      <footer className="site-footer"><span>TrueLine</span><span>Evidence-led brand governance · Human decisions stay in control</span></footer>
      {toast && <div className="toast"><span>!</span>{toast}</div>}
    </div>
  );
}

export default App;
