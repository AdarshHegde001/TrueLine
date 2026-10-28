import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? 'http://localhost:3001' : '');
const defaultDocumentType = 'brand_guidelines';
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

const documentTypeDescriptions = {
  brand_guidelines: 'Visual identity, colour palettes, logo usage, typography standards.',
  tone_of_voice: 'Writing style, vocabulary preferences, personality guidelines.',
  approved_claims: 'Pre-approved marketing statements and benefit claims.',
  product_facts: 'Verified product specifications, certifications, and data.',
  glossary: 'Approved terminology and preferred phrasing.',
  asset_rules: 'Usage rules for imagery, icons, and creative assets.',
  legal_compliance: 'Regulatory requirements, disclaimers, and legal copy rules.',
  other_reference: 'Any additional approved reference material.',
};

function StatusPill({ status }) {
  const normalized = String(status || '').toUpperCase();
  const kind = ['READY', 'COMPLETE', 'PASSED'].includes(normalized) ? 'good'
    : ['ERROR', 'CHECK_FAILED', 'VIOLATION'].includes(normalized) ? 'bad'
      : ['UNSUPPORTED', 'UNVERIFIED'].includes(normalized) ? 'pending' : 'pending';
  const label = normalized === 'COMPLETE' ? 'COMPLETED' : String(status || 'pending').replaceAll('_', ' ');
  return <span className={`status-pill ${kind}`}><i />{label}</span>;
}

function MarkdownText({ children, className }) {
  if (!children) return null;
  // Raw HTML remains inert; do not add rehype-raw or another unsafe renderer.
  return <div className={className}><ReactMarkdown remarkPlugins={[remarkGfm]}>{String(children)}</ReactMarkdown></div>;
}

/* ─── Guided onboarding steps component ─────────────────────────────────── */
function GuidedSteps({ session, uploads, knowledgeBatchComplete, onStartSession, onGoKnowledge, onGoReview, sessionBusy, sessionLoading }) {
  const step = !session ? 1 : uploads.length === 0 ? 2 : !knowledgeBatchComplete ? 3 : 4;
  const steps = [
    { id: 1, label: 'Start a session', desc: 'Create a private workspace. Uploaded knowledge is cleared when you end the session.', icon: '◎', done: !!session },
    { id: 2, label: 'Add source documents', desc: 'Upload brand guidelines, tone rules, approved claims, and other reference files.', icon: '↥', done: uploads.length > 0 },
    { id: 3, label: 'Index your knowledge', desc: 'Submit the batch to embed and index files so they can be retrieved during review.', icon: '▤', done: knowledgeBatchComplete },
    { id: 4, label: 'Verify a document', desc: 'Upload a draft and run deterministic + AI-powered checks against your sources.', icon: '✧', done: false },
  ];

  return (
    <div className="guided-steps-bar">
      <div className="guided-steps-label">
        <span className="guided-steps-eyebrow">QUICK START</span>
        <span className="guided-steps-subtitle">You're on step {step} of 4</span>
      </div>
      <div className="guided-steps-track">
        {steps.map((s, i) => {
          const isActive = s.id === step;
          const isDone = s.done;
          return (
            <div key={s.id} className={`guided-step ${isDone ? 'done' : isActive ? 'active' : 'idle'}`}>
              <div className="guided-step-icon">{isDone ? '✓' : s.icon}</div>
              <div className="guided-step-body">
                <b>{s.label}</b>
                <small>{s.desc}</small>
              </div>
              {isActive && !isDone && (
                <button
                  className="guided-step-cta"
                  disabled={sessionBusy || sessionLoading}
                  onClick={
                    s.id === 1 ? onStartSession
                    : s.id === 2 || s.id === 3 ? onGoKnowledge
                    : onGoReview
                  }
                >
                  {s.id === 1 ? (sessionLoading ? 'Restoring…' : sessionBusy ? 'Starting…' : 'Start') : 'Open →'}
                </button>
              )}
              {i < steps.length - 1 && <div className="guided-step-connector" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Document type helper tooltip ───────────────────────────────────────── */
function DocTypeTooltip({ type }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="doctype-tooltip-wrap">
      <button
        type="button"
        className="doctype-help-btn"
        aria-label="Document type info"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >?</button>
      {open && type && (
        <span className="doctype-tooltip">{documentTypeDescriptions[type] || 'Categorise this file so agents can filter retrieval correctly.'}</span>
      )}
    </span>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [health, setHealth] = useState(null);
  const [knowledgeFiles, setKnowledgeFiles] = useState([]);
  const [logos, setLogos] = useState([]);
  const [logoFile, setLogoFile] = useState(null);
  const [logoTitle, setLogoTitle] = useState('');
  const [logoBusy, setLogoBusy] = useState(false);
  const logoInputRef = useRef(null);
  const [reviewFile, setReviewFile] = useState(null);
  const [channel, setChannel] = useState('advertisement');
  const [uploads, setUploads] = useState([]);
  const [session, setSession] = useState(null);
  const [sessionLoading, setSessionLoading] = useState(true);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [reviewResult, setReviewResult] = useState(null);
  const [knowledgeBusy, setKnowledgeBusy] = useState(false);
  const [knowledgeBatchComplete, setKnowledgeBatchComplete] = useState(false);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [correctionBusy, setCorrectionBusy] = useState(false);
  const [correctedDocx, setCorrectedDocx] = useState(null);
  const [toast, setToast] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [reviewDragOver, setReviewDragOver] = useState(false);

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

  useEffect(() => {
    const savedSessionId = localStorage.getItem('trueline.sessionId');
    if (!savedSessionId) {
      setSessionLoading(false);
      return undefined;
    }
    let alive = true;
    const resume = async () => {
      try {
        const response = await fetch(`${API}/api/session/resume`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ sessionId: savedSessionId }),
        });
        const data = await response.json();
        if (!response.ok) {
          if (response.status === 409) localStorage.removeItem('trueline.sessionId');
          throw new Error(data.error || 'Could not resume the session.');
        }
        if (alive) {
          setSession(data.session);
          try {
            const sourceResponse = await fetch(`${API}/api/knowledge?sessionId=${encodeURIComponent(savedSessionId)}`);
            const sourceData = await sourceResponse.json();
            if (sourceResponse.ok) setUploads(sourceData.sources || []);
          } catch { /* Keep the valid session; source listing can be retried after reconnect. */ }
          try {
            const logoResponse = await fetch(`${API}/api/logos?sessionId=${encodeURIComponent(savedSessionId)}`);
            const logoData = await logoResponse.json();
            if (alive && logoResponse.ok) setLogos(logoData.logos || []);
          } catch { /* Logo references can be refreshed after reconnect. */ }
        }
      } catch {
        if (!localStorage.getItem('trueline.sessionId') && alive) setSession(null);
      } finally {
        if (alive) setSessionLoading(false);
      }
    };
    resume();
    return () => { alive = false; };
  }, []);

  const counts = useMemo(() => ({
    guidance: uploads.filter((item) => ['brand_guidelines', 'tone_of_voice', 'legal_compliance', 'glossary', 'asset_rules'].includes(item.document_type)).length,
    approved: uploads.filter((item) => ['approved_claims', 'product_facts'].includes(item.document_type)).length,
  }), [uploads]);

  function notify(message) {
    setToast(message);
    window.setTimeout(() => setToast(''), 4200);
  }

  function goToReview() {
    if (!session) return notify('Start a session before opening document review.');
    if (knowledgeBusy) return notify('Wait for the current embedding batch to finish first.');
    if (logoBusy) return notify('Wait for the logo reference upload to finish first.');
    if (!uploads.length) return notify('Add and index approved sources before starting a review.');
    if (knowledgeFiles.some((item) => item.status !== 'ready')) return notify('Finish indexing every source file before moving to review.');
    setActiveTab('review');
  }

  function goToKnowledge() {
    if (!session) return notify('Start a session before adding source files.');
    setActiveTab('knowledge');
  }

  async function startSession() {
    setSessionBusy(true);
    setToast('');
    try {
      const response = await fetch(`${API}/api/session/start`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not start a session.');
      localStorage.setItem('trueline.sessionId', data.session.sessionId);
      setSession(data.session);
      setKnowledgeFiles([]);
      setUploads([]);
      setLogos([]);
      setLogoFile(null);
      setLogoTitle('');
      setReviewFile(null);
      setReviewResult(null);
      setCorrectedDocx(null);
      setKnowledgeBatchComplete(false);
      setActiveTab('knowledge');
      notify('Session started. Add your source files; they will be cleared when you end the session.');
    } catch (error) {
      notify(error.message);
    } finally {
      setSessionBusy(false);
    }
  }

  async function endSession() {
    if (!session) return;
    if (knowledgeBusy || reviewBusy || correctionBusy || logoBusy) return notify('Wait for the current operation to finish before ending the session.');
    const approved = window.confirm('End this session and permanently delete all knowledge sources and embeddings from the database?');
    if (!approved) return;
    setSessionBusy(true);
    setToast('');
    try {
      const response = await fetch(`${API}/api/session/stop`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ sessionId: session.sessionId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not end the session.');
      localStorage.removeItem('trueline.sessionId');
      setSession(null);
      setKnowledgeFiles([]);
      setUploads([]);
      setLogos([]);
      setLogoFile(null);
      setLogoTitle('');
      setReviewFile(null);
      setReviewResult(null);
      setCorrectedDocx(null);
      setKnowledgeBatchComplete(false);
      setActiveTab('home');
      notify('Session ended. All uploaded knowledge and embeddings were cleared.');
    } catch (error) {
      notify(error.message);
    } finally {
      setSessionBusy(false);
    }
  }

  async function submitKnowledge(event) {
    event.preventDefault();
    if (!session) return notify('Start a session before adding sources.');
    const pending = knowledgeFiles.filter((item) => item.status !== 'ready');
    if (!pending.length) return notify('Add one or more source documents first.');
    setKnowledgeBusy(true);
    setKnowledgeBatchComplete(false);
    setToast('');
    let failed = 0;
    for (const item of pending) {
      setKnowledgeFiles((items) => items.map((current) => current.id === item.id ? { ...current, status: 'indexing', error: '' } : current));
      try {
        const body = new FormData();
        body.append('file', item.file);
        body.append('sessionId', session.sessionId);
        body.append('title', item.title.trim() || item.file.name.replace(/\.[^.]+$/, ''));
        body.append('documentType', item.documentType);
        body.append('approved', 'true');
        const response = await fetch(`${API}/api/knowledge`, { method: 'POST', body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not ingest this source.');
        setKnowledgeFiles((items) => items.map((current) => current.id === item.id ? { ...current, status: 'ready', source: data.source, error: '' } : current));
        setUploads((items) => [{ ...data.source, addedAt: new Date() }, ...items]);
      } catch (error) {
        failed += 1;
        setKnowledgeFiles((items) => items.map((current) => current.id === item.id ? { ...current, status: 'error', error: error.message } : current));
      }
    }
    setKnowledgeBusy(false);
    setKnowledgeBatchComplete(failed === 0);
    notify(failed ? `${failed} file${failed === 1 ? '' : 's'} failed to index. Fix or remove them and retry.` : 'All source files are embedded and indexed. You can continue to review.');
  }

  async function submitLogo(event) {
    event.preventDefault();
    if (!session) return notify('Start a session before adding logo references.');
    if (!logoFile) return notify('Choose a logo image first.');
    setLogoBusy(true);
    try {
      const body = new FormData();
      body.append('file', logoFile);
      body.append('title', logoTitle.trim());
      body.append('sessionId', session.sessionId);
      const response = await fetch(`${API}/api/logos`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not save this logo.');
      setLogos((items) => [data.logo, ...items.filter((item) => item.logo_id !== data.logo.logo_id)]);
      setLogoFile(null);
      setLogoTitle('');
      if (logoInputRef.current) logoInputRef.current.value = '';
      notify('Approved logo reference saved for this session.');
    } catch (error) {
      notify(error.message);
    } finally {
      setLogoBusy(false);
    }
  }

  async function removeLogo(logoId) {
    if (!session) return;
    try {
      const response = await fetch(`${API}/api/logos/${encodeURIComponent(logoId)}?sessionId=${encodeURIComponent(session.sessionId)}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not remove this logo.');
      setLogos((items) => items.filter((item) => item.logo_id !== logoId));
      notify('Logo reference removed.');
    } catch (error) {
      notify(error.message);
    }
  }

  function addKnowledgeFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    setKnowledgeBatchComplete(false);
    setKnowledgeFiles((items) => [...items, ...incoming.map((file) => ({
      id: crypto.randomUUID(), file, title: file.name.replace(/\.[^.]+$/, ''),
      documentType: defaultDocumentType, status: 'queued', error: '',
    }))]);
  }

  async function submitReview(event) {
    event.preventDefault();
    if (!session) return notify('Start a session before reviewing a document.');
    if (!reviewFile) return notify('Choose a document to verify first.');
    setReviewBusy(true);
    setReviewResult(null);
    setCorrectedDocx(null);
    setToast('');
    try {
      const body = new FormData();
      body.append('file', reviewFile);
      body.append('sessionId', session.sessionId);
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

  async function generateCorrection() {
    if (!session || !reviewFile || !summaryResult?.summary) return notify('Complete a review before generating a corrected draft.');
    setCorrectionBusy(true);
    setCorrectedDocx(null);
    try {
      const findings = summaryResult.findings || [];
      const extractedRules = agentResults.extraction?.rules || [];
      const body = new FormData();
      body.append('file', reviewFile);
      body.append('sessionId', session.sessionId);
      body.append('reviewData', JSON.stringify({ summary: summaryResult.summary, extracted_rules: extractedRules, findings }));
      const response = await fetch(`${API}/api/review/correct`, { method: 'POST', body });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Could not generate a corrected draft.');
      }
      const blob = await response.blob();
      const stem = reviewFile.name.replace(/\.[^.]+$/, '') || 'review';
      setCorrectedDocx({
        blob, filename: `${stem}-corrected.docx`,
        editsApplied: Number(response.headers.get('X-Trueline-Edits-Applied') || 0),
        unresolvedCount: Number(response.headers.get('X-Trueline-Unresolved-Count') || 0),
        formatNote: decodeURIComponent(response.headers.get('X-Trueline-Format-Note') || ''),
      });
    } catch (error) {
      notify(error.message);
    } finally {
      setCorrectionBusy(false);
    }
  }

  function downloadCorrection() {
    if (!correctedDocx) return;
    const url = URL.createObjectURL(correctedDocx.blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = correctedDocx.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const deterministicFindings = reviewResult?.preparedReview?.deterministic?.findings || [];
  const agentResults = reviewResult?.agentResults || {};
  const summaryResult = agentResults.summary;
  const extractionResult = agentResults.extraction;
  const logoAgentResult = agentResults.logo;
  const specialistEvidence = reviewResult?.agentEvidence || {};

  return (
    <div className="app-shell">
      <header className="site-header">
        <button className="logo-button" onClick={() => setActiveTab('home')} aria-label="TrueLine home">
          <img src="/TrueLine_logo.jpg" alt="TrueLine" />
        </button>
        <nav className="main-nav" aria-label="Main navigation">
          <button className={`nav-link ${activeTab === 'home' ? 'active' : ''}`} onClick={() => setActiveTab('home')}>Home</button>
          <button className={`nav-link ${activeTab === 'knowledge' ? 'active' : ''}`} disabled={!session || sessionLoading} onClick={goToKnowledge}>Knowledge base</button>
          <button className={`nav-link ${activeTab === 'review' ? 'active' : ''}`} disabled={!session || sessionLoading} onClick={goToReview}>Verify a document</button>
        </nav>
        <div className="session-control">
          {session && <span className="session-state"><i /> Session active</span>}
          {session ? <button className="session-button stop" disabled={sessionBusy || sessionLoading || knowledgeBusy || reviewBusy || correctionBusy || logoBusy} onClick={endSession}>{sessionBusy ? 'Please wait…' : 'End session'}</button>
            : <button className="session-button" disabled={sessionBusy || sessionLoading} onClick={startSession}>{sessionLoading ? 'Restoring…' : sessionBusy ? 'Starting…' : 'Start session'}</button>}
        </div>
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
                  <button className="primary-button" onClick={goToKnowledge}>Add brand sources <b>→</b></button>
                  <button className="text-button" onClick={goToReview}><span>Verify content</span><b>↗</b></button>
                </div>
                <div className="hero-principle"><span>✓</span><p><b>Cite or stay silent.</b> Unsupported claims are marked unverified, never assumed false.</p></div>
              </div>
              <div className="hero-visual">
                <div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" />
                <div className="review-preview">
                  <div className="preview-top"><span className="preview-file-icon">▤</span><span><b>Campaign draft</b><small>Evidence-led review</small></span><span className="preview-menu">•••</span></div>
                  <div className="preview-content"><span className="preview-kicker">REVIEWING A CLAIM</span><p>"Our new formula is <mark>clinically proven</mark> to work."</p>
                    <div className="preview-finding"><span className="finding-icon">!</span><span><b>Needs supporting evidence</b><small>Check against approved claims.</small></span><span className="finding-arrow">↗</span></div>
                    <div className="preview-source"><span className="source-check">✓</span><span><small>RETRIEVED SOURCE</small><b>Claims policy · Section 3.2</b></span><span className="source-link">↗</span></div>
                  </div>
                  <div className="preview-footer"><span><i /> Source linked</span><small>Human review stays in control</small></div>
                </div>
                <div className="floating-note"><span>↗</span><span><b>Evidence first</b><small>at every step</small></span></div>
              </div>
            </section>

            {/* ── Guided steps bar ─────────────────────────────────── */}
            <div className="page-content">
              <GuidedSteps
                session={session}
                uploads={uploads}
                knowledgeBatchComplete={knowledgeBatchComplete}
                onStartSession={startSession}
                onGoKnowledge={goToKnowledge}
                onGoReview={goToReview}
                sessionBusy={sessionBusy}
                sessionLoading={sessionLoading}
              />
            </div>

            <section className={`session-banner page-content ${session ? 'active' : ''}`}>
              <div className="session-banner-icon">{session ? '✓' : '↗'}</div>
              <div><b>{session ? 'Your review session is active' : 'Start a private review session'}</b><p>{session ? 'Add approved source files, wait for indexing, then review a document. Ending this session deletes its indexed knowledge.' : 'Starting creates a fresh knowledge index. Ending clears uploaded sources and embeddings from the database.'}</p></div>
              {session ? <button className="primary-button" onClick={goToKnowledge}>Open knowledge base <b>→</b></button> : <button className="primary-button" disabled={sessionBusy || sessionLoading} onClick={startSession}>{sessionLoading ? 'Restoring…' : sessionBusy ? 'Starting…' : <>Start session <b>→</b></>}</button>}
            </section>

            <section className="principles-section page-content">
              <div className="section-intro"><div className="eyebrow"><span className="eyebrow-line" /> THE TRUE LINE APPROACH</div><h2>Review content with confidence.</h2><p>Catch inconsistencies early, see why something was flagged, and make the final decision as a person.</p></div>
              <div className="principle-grid">
                <article className="principle-card">
                  <span className="principle-number">01</span>
                  <div className="principle-icon">⌕</div>
                  <h3>Rules before guesswork</h3>
                  <p>Deterministic checks handle approved terminology and required wording consistently.</p>
                  <div className="principle-card-footer"><span className="principle-tag">Deterministic</span></div>
                </article>
                <article className="principle-card">
                  <span className="principle-number">02</span>
                  <div className="principle-icon">◈</div>
                  <h3>Evidence with each finding</h3>
                  <p>Retrieved passages link back to approved source documents and their versions.</p>
                  <div className="principle-card-footer"><span className="principle-tag">RAG-grounded</span></div>
                </article>
                <article className="principle-card">
                  <span className="principle-number">03</span>
                  <div className="principle-icon">◎</div>
                  <h3>People stay in control</h3>
                  <p>AI can support a review, but people remain responsible for the outcome.</p>
                  <div className="principle-card-footer"><span className="principle-tag">Human-in-the-loop</span></div>
                </article>
              </div>
            </section>

            <section className="workflow-section">
              <div className="page-content">
                <div className="section-intro workflow-intro"><div className="eyebrow"><span className="eyebrow-line" /> HOW IT WORKS</div><h2>From approved sources to a reviewed draft.</h2><p>Prepare your approved knowledge once, then run a deterministic and agent-assisted review whenever you have a draft.</p></div>
                <div className="workflow-steps">
                  <article><span className="workflow-index">01</span><span className="workflow-icon">↥</span><h3>Add knowledge</h3><p>Upload guidelines, tone rules, claims, facts, glossaries, and asset rules with a document type.</p><span className="stage-tag available">Available now</span></article>
                  <article><span className="workflow-index">02</span><span className="workflow-icon">▤</span><h3>Prepare the index</h3><p>Text is parsed, chunked, tagged, and embedded for hybrid search with PostgreSQL and pgvector.</p><span className="stage-tag available">Available now</span></article>
                  <article><span className="workflow-index">03</span><span className="workflow-icon">⌕</span><h3>Check a draft</h3><p>Deterministic rules run and relevant approved evidence is retrieved for the submitted document.</p><span className="stage-tag available">Available now</span></article>
                  <article><span className="workflow-index">04</span><span className="workflow-icon">✧</span><h3>Extract, review, summarize</h3><p>An extraction agent validates explicit rules and citations first; focused tone and claims agents review against them, while the final agent summarizes from the rules and draft.</p><span className="stage-tag available">Available now</span></article>
                </div>
              </div>
            </section>

            {/* ── What TrueLine checks ─── */}
            <section className="checks-section page-content">
              <div className="section-intro">
                <div className="eyebrow"><span className="eyebrow-line" /> WHAT GETS REVIEWED</div>
                <h2>Four layers of review in every run.</h2>
                <p>TrueLine combines rule-based precision with AI-assisted analysis to give you comprehensive coverage.</p>
              </div>
              <div className="checks-grid">
                <div className="check-card check-card--a">
                  <div className="check-card-badge">A</div>
                  <div className="check-card-icon">⌕</div>
                  <h3>Deterministic checks</h3>
                  <p>Structured rule engine validates required disclaimers, forbidden phrases, and required terminology against known patterns — no guesswork.</p>
                  <ul className="check-list">
                    <li>Forbidden word detection</li>
                    <li>Required disclaimer presence</li>
                    <li>Approved terminology matching</li>
                  </ul>
                </div>
                <div className="check-card check-card--b">
                  <div className="check-card-badge">B</div>
                  <div className="check-card-icon">◈</div>
                  <h3>Tone agent</h3>
                  <p>Retrieves approved tone-of-voice guidelines and evaluates whether the draft's voice, style, and personality match your brand standards.</p>
                  <ul className="check-list">
                    <li>Voice & personality alignment</li>
                    <li>Style consistency</li>
                    <li>Evidence-cited findings</li>
                  </ul>
                </div>
                <div className="check-card check-card--c">
                  <div className="check-card-badge">C</div>
                  <div className="check-card-icon">✦</div>
                  <h3>Claims agent</h3>
                  <p>Cross-references every claim in the draft against approved facts and verified statements, flagging anything that lacks supporting evidence.</p>
                  <ul className="check-list">
                    <li>Claim verification</li>
                    <li>Approved facts cross-check</li>
                    <li>Source citations per finding</li>
                  </ul>
                </div>
                <div className="check-card check-card--d">
                  <div className="check-card-badge">D</div>
                  <div className="check-card-icon">◉</div>
                  <h3>Logo review</h3>
                  <p>Compares logo candidates extracted from uploaded images against your approved logo references using visual similarity analysis.</p>
                  <ul className="check-list">
                    <li>Logo candidate extraction</li>
                    <li>Visual similarity comparison</li>
                    <li>Per-page observations</li>
                  </ul>
                </div>
              </div>
            </section>

            <section className="prototype-section page-content">
              <div className="prototype-card"><div className="prototype-copy"><div className="eyebrow"><span className="eyebrow-line" /> CURRENT WORKFLOW</div><h2>Every review traces back to its sources.</h2><p>Each review combines deterministic checks with focused tone and claims agents. A final LangGraph step summarizes the findings, and the interface displays the supporting retrieved citations.</p><button className="text-button" onClick={goToReview}><span>Review a document</span><b>→</b></button></div><div className="prototype-checklist"><div><span className="check-state done">✓</span><span><b>Source library</b><small>Typed · embedded · searchable</small></span></div><div><span className="check-state demo">✦</span><span><b>Specialist agents</b><small>Tone · claims · summary</small></span></div><div><span className="check-state next">→</span><span><b>Source citations</b><small>Evidence stays visible for review</small></span></div></div></div>
            </section>

            <section className="closing-cta page-content"><div><div className="eyebrow"><span className="eyebrow-line" /> START WITH YOUR SOURCES</div><h2>Give every review a trusted reference.</h2></div><button className="primary-button" onClick={goToKnowledge}>Open knowledge base <b>→</b></button></section>
          </>
        )}

        {activeTab === 'knowledge' ? (
          <section className="page-content animate-fadein">
            <div className="eyebrow"><span className="eyebrow-line" /> KNOWLEDGE BASE</div>
            <div className="page-heading-row"><div><h1>Bring your brand rules together.</h1><p className="subtitle">Upload approved sources and label what each document contains so the right evidence can be found later.</p></div><button className="text-button" onClick={goToReview}><span>Verify a document</span><b>↗</b></button></div>

            {/* ── Knowledge guide banner ── */}
            <div className="knowledge-guide-banner">
              <div className="kguide-step">
                <div className="kguide-num">1</div>
                <div><b>Choose files</b><small>PDF, DOCX, TXT or Markdown up to 15 MB</small></div>
              </div>
              <div className="kguide-arrow">→</div>
              <div className="kguide-step">
                <div className="kguide-num">2</div>
                <div><b>Assign document type</b><small>Guides agent retrieval filtering</small></div>
              </div>
              <div className="kguide-arrow">→</div>
              <div className="kguide-step">
                <div className="kguide-num">3</div>
                <div><b>Submit &amp; index</b><small>Chunks embedded into pgvector</small></div>
              </div>
              <div className="kguide-arrow">→</div>
              <div className="kguide-step">
                <div className="kguide-num">4</div>
                <div><b>Ready for review</b><small>Move to Verify a document</small></div>
              </div>
            </div>

            <div className="stat-grid">
              <div className="stat-card"><div className="stat-icon violet">▤</div><div><span>GUIDANCE</span><b>{counts.guidance}</b></div><small>tone, brand, compliance</small></div>
              <div className="stat-card"><div className="stat-icon blue">◈</div><div><span>CLAIMS &amp; FACTS</span><b>{counts.approved}</b></div><small>approved sources</small></div>
              <div className="stat-card"><div className="stat-icon green">⌕</div><div><span>RETRIEVAL INDEX</span><b>{uploads.length ? 'Ready' : 'Waiting'}</b></div><small>Postgres + pgvector</small></div>
            </div>

            <div className="section-title-row">
              <div><h2>Add source files</h2><p>Assign a document type to every file. Submit the batch and wait for indexing before moving to review.</p></div>
              <span className="step-label"><b>01</b> INGEST</span>
            </div>
            <div className="content-grid">
              <form className="panel upload-panel" onSubmit={submitKnowledge}>
                <label
                  className={`dropzone${dragOver ? ' dropzone--active' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); addKnowledgeFiles(e.dataTransfer.files); }}
                >
                  <input type="file" multiple accept=".txt,.md,.pdf,.docx" onChange={(e) => { addKnowledgeFiles(e.target.files); e.target.value = ''; }} />
                  <span className="drop-icon">↑</span><b>Choose multiple files or drag them here</b><small>PDF, DOCX, TXT or Markdown · up to 15 MB per file</small>
                </label>
                {knowledgeFiles.length > 0 && <div className="batch-list" aria-live="polite">
                  <div className="batch-list-heading"><b>{knowledgeFiles.length} file{knowledgeFiles.length === 1 ? '' : 's'} in this batch</b><span>{knowledgeFiles.filter((item) => item.status === 'ready').length} indexed</span></div>
                  {knowledgeFiles.map((item) => <div className={`batch-file ${item.status}`} key={item.id}>
                    <div className="batch-file-icon">▤</div>
                    <div className="batch-file-fields"><label className="field"><span>Source title</span><input value={item.title} aria-label={`Title for ${item.file.name}`} disabled={knowledgeBusy} onChange={(e) => { setKnowledgeBatchComplete(false); setKnowledgeFiles((items) => items.map((current) => current.id === item.id ? { ...current, title: e.target.value, status: current.status === 'ready' ? 'queued' : current.status, source: null } : current)); }} /></label><small className="batch-filename">{item.file.name} · {(item.file.size / 1024 / 1024).toFixed(2)} MB</small></div>
                    <label className="field batch-type">
                      <span>Document type <DocTypeTooltip type={item.documentType} /></span>
                      <select value={item.documentType} disabled={knowledgeBusy} onChange={(e) => { setKnowledgeBatchComplete(false); setKnowledgeFiles((items) => items.map((current) => current.id === item.id ? { ...current, documentType: e.target.value, status: current.status === 'ready' ? 'queued' : current.status, source: null } : current)); }}>{Object.entries(documentTypeLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select>
                    </label>
                    <div className="batch-status"><StatusPill status={item.status === 'indexing' ? 'indexing' : item.status === 'queued' ? 'waiting' : item.status} />{item.error && <small className="batch-error">{item.error}</small>}{item.source?.parse_warnings?.map((warning, index) => <small className="parse-warning" key={`${item.id}-warning-${index}`}>{warning}</small>)}</div>
                    {!knowledgeBusy && <button type="button" className="remove-file" aria-label={`Remove ${item.file.name}`} onClick={() => { setKnowledgeFiles((items) => items.filter((current) => current.id !== item.id)); setKnowledgeBatchComplete(false); }}>×</button>}
                  </div>)}
                  {knowledgeBusy && <div className="batch-progress"><div><span>Embedding and indexing sources sequentially</span><b>{knowledgeFiles.filter((item) => item.status === 'ready').length} / {knowledgeFiles.length}</b></div><progress max={knowledgeFiles.length} value={knowledgeFiles.filter((item) => item.status === 'ready').length} /></div>}
                </div>}
                <div className="form-foot"><div className="approval-note"><span className="checkmark">✓</span><span><b>Approved knowledge</b><small>Each file keeps its own type in the index.</small></span></div><button className="primary-button" type="submit" disabled={knowledgeBusy || !knowledgeFiles.some((item) => item.status !== 'ready') || !health || health.status !== 'ready'}>{knowledgeBusy ? <><span className="spinner" /> Indexing batch…</> : <>Submit all files <b>→</b></>}</button></div>
                {knowledgeBatchComplete && <div className="batch-complete"><span>✓</span><div><b>Knowledge base is ready for review</b><small>Every file in this batch finished embedding and indexing.</small></div><button className="primary-button" type="button" onClick={goToReview}>Continue to review <b>→</b></button></div>}
                <p className="fine-print">The selected document type is included in embedding input and stored as a retrieval filter. Structured deterministic rules are managed through the ingestion contract.</p>
              </form>
              <aside className="info-panel">
                <div className="info-icon">✳</div>
                <h3>What happens after upload?</h3>
                <ol>
                  <li><b>Text is extracted</b><span>PDF, DOCX and text files are parsed with source metadata.</span></li>
                  <li><b>Source is versioned</b><span>The original text and its selected document type are saved.</span></li>
                  <li><b>Search index is prepared</b><span>Typed chunks are embedded and made available to hybrid retrieval.</span></li>
                </ol>
                <div className="info-note"><b>Document type tip</b><p>Choose the closest purpose. Tone, claims, and fact agents filter retrieval using these types.</p></div>
                {/* ── Document type quick reference ── */}
                <div className="doctype-guide">
                  <div className="doctype-guide-title">Document type guide</div>
                  {Object.entries(documentTypeLabels).map(([key, label]) => (
                    <div className="doctype-guide-item" key={key}>
                      <span className="doctype-guide-label">{label}</span>
                      <span className="doctype-guide-desc">{documentTypeDescriptions[key]}</span>
                    </div>
                  ))}
                </div>
              </aside>
            </div>

            <div className="section-title-row library-title"><div><h2>Approved logo references</h2><p>Store the approved logo versions to keep them with this session.</p></div><span className="muted-label">{logos.length} LOGO{logos.length === 1 ? '' : 'S'}</span></div>
            <div className="panel logo-library">
              <form className="logo-upload-form" onSubmit={submitLogo}>
                <label className="field logo-title-field"><span>Logo name <em>Optional</em></span><input value={logoTitle} maxLength={120} onChange={(event) => setLogoTitle(event.target.value)} placeholder={logoFile?.name?.replace(/\.[^.]+$/, '') || 'Primary logo'} disabled={logoBusy} /></label>
                <label className="logo-picker"><input ref={logoInputRef} type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setLogoFile(event.target.files?.[0] || null)} disabled={logoBusy} /><span className="drop-icon">↑</span><span><b>{logoFile?.name || 'Choose an approved logo'}</b><small>PNG, JPEG, or WebP · up to 5 MB</small></span></label>
                <button className="primary-button" type="submit" disabled={logoBusy || !logoFile}>{logoBusy ? <><span className="spinner" /> Saving…</> : <>Save logo <b>→</b></>}</button>
              </form>
              {logos.length > 0 ? <div className="logo-reference-list">{logos.map((logo) => <article className="logo-reference-item" key={logo.logo_id}>
                <img src={`${API}/api/logos/${encodeURIComponent(logo.logo_id)}/image?sessionId=${encodeURIComponent(session.sessionId)}`} alt={logo.title} />
                <div><b>{logo.title}</b><small>{logo.original_name} · {logo.width} × {logo.height}</small></div>
                <button className="remove-file" type="button" aria-label={`Remove ${logo.title}`} onClick={() => removeLogo(logo.logo_id)} disabled={logoBusy}>×</button>
              </article>)}</div> : <div className="logo-empty">Add one or more approved logo files. They stay private to this session.</div>}
            </div>

            <div className="section-title-row library-title"><div><h2>Recently added</h2><p>Uploads made during this session.</p></div><span className="muted-label">{uploads.length} SOURCE{uploads.length === 1 ? '' : 'S'}</span></div>
            <div className="panel table-panel">{uploads.length ? <div className="source-list">{uploads.map((item) => <div className="source-row" key={item.version_id}><div className="source-file-icon knowledge">▤</div><div className="source-name"><b>{item.title}</b><small>{documentTypeLabels[item.document_type] || 'Other reference'} · v{item.version} · {item.chunk_count} chunks</small></div><span className="rule-count">{item.rule_count ? `${item.rule_count} rules` : 'No structured rules'}</span><StatusPill status={item.status} /></div>)}</div> : <div className="empty-state"><div className="empty-icon">▤</div><b>No sources added yet</b><p>Upload approved organization guidance, facts, or claims to prepare retrieval.</p></div>}</div>
          </section>
        ) : activeTab === 'review' ? (
          <section className="page-content review-page animate-fadein">
            <div className="eyebrow"><span className="eyebrow-line" /> DOCUMENT REVIEW</div>
            <div className="page-heading-row"><div><h1>Check a draft before it goes out.</h1><p className="subtitle">Run deterministic checks and evidence-led tone and claims reviews against your indexed sources.</p></div><span className="step-label"><b>01</b> AGENTIC REVIEW</span></div>

            {/* ── Review guide banner ── */}
            <div className="review-guide-banner">
              <div className="rguide-item">
                <span className="rguide-icon">▤</span>
                <div><b>Upload your draft</b><small>PDF, DOCX, TXT, Markdown, or image</small></div>
              </div>
              <div className="rguide-item">
                <span className="rguide-icon">◎</span>
                <div><b>Choose the channel</b><small>Helps agents apply the right context</small></div>
              </div>
              <div className="rguide-item">
                <span className="rguide-icon">✧</span>
                <div><b>Run the review</b><small>Tone, claims, logo + summary agents</small></div>
              </div>
              <div className="rguide-item">
                <span className="rguide-icon">◈</span>
                <div><b>Read findings or request a DOCX</b><small>Optional corrected draft after review</small></div>
              </div>
            </div>

            <div className="review-layout">
              <form className="panel review-form" onSubmit={submitReview}>
                <div className="panel-heading"><div className="panel-step">1</div><div><h3>Choose the document</h3><p>We'll parse it and locate relevant approved sources.</p></div></div>
                <label
                  className={`dropzone review-drop${reviewDragOver ? ' dropzone--active' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setReviewDragOver(true); }}
                  onDragLeave={() => setReviewDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setReviewDragOver(false); setReviewFile(e.dataTransfer.files?.[0] || null); setReviewResult(null); setCorrectedDocx(null); }}
                >
                  <input type="file" accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg,.webp" onChange={(e) => { setReviewFile(e.target.files?.[0] || null); setReviewResult(null); setCorrectedDocx(null); }} />
                  <span className="drop-icon">↑</span><b>{reviewFile ? reviewFile.name : 'Choose a document or image to verify'}</b><small>PDF, DOCX, TXT, Markdown, PNG, JPEG, or WebP · up to 15 MB</small>
                </label>
                <p className="image-privacy-note">Text and tables are extracted locally. Only approved logo references and image candidates matched to them are sent to the logo agent; charts and unrelated images stay local.</p>
                <div className="form-grid review-fields single-field"><label className="field"><span>Channel <em>Optional</em></span><select value={channel} onChange={(e) => setChannel(e.target.value)}><option value="advertisement">Advertisement</option><option value="social">Social media</option><option value="website">Website</option><option value="email">Email</option><option value="">No channel</option></select></label></div>
                <div className="review-pipeline"><div className="pipeline-title">REVIEW WORKFLOW</div><div className="pipeline-step"><span>1</span><div><b>Parse and check</b><small>Extract text and run deterministic rules</small></div></div><div className="pipeline-step"><span>2</span><div><b>Review voice and claims</b><small>Specialists retrieve relevant typed sources</small></div></div><div className="pipeline-step"><span>3</span><div><b>Summarize with citations</b><small>Final agent gets evidence and prior findings</small></div></div></div>
                <button className="primary-button review-submit" disabled={reviewBusy || !reviewFile || !health || health.status !== 'ready'} type="submit">{reviewBusy ? <><span className="spinner" /> Running agents…</> : <>Run review <b>→</b></>}</button>
              </form>
              <aside className="review-side">
                <div className="review-side-card"><div className="side-card-icon">✧</div><div><b>Evidence-led review</b><p>Retrieved passages are treated as evidence, not instructions, and stay linked to their source version.</p></div></div>
                <div className="review-side-card quiet"><div className="side-card-icon">⚡</div><div><b>Human review stays in control</b><p>Agent findings and suggested interpretations support your review; they do not make a compliance decision.</p></div></div>
                {/* ── Channel guide card ── */}
                <div className="review-side-card channel-guide-card">
                  <div className="side-card-icon">◎</div>
                  <div>
                    <b>Channel context guide</b>
                    <div className="channel-guide-list">
                      <div><span className="channel-tag">Advertisement</span><small>Regulatory claims, visual prominence rules</small></div>
                      <div><span className="channel-tag">Social media</span><small>Tone, engagement language, hashtag use</small></div>
                      <div><span className="channel-tag">Website</span><small>SEO copy, long-form brand voice</small></div>
                      <div><span className="channel-tag">Email</span><small>Subject lines, personalisation, disclaimers</small></div>
                    </div>
                  </div>
                </div>
              </aside>
            </div>

            {reviewResult && <div className="results-section">
              <div className="results-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> REVIEW OUTPUT</div><h2>Review results</h2></div><StatusPill status={summaryResult?.status || 'complete'} /></div>
              {reviewResult.preparedReview?.document?.warnings?.length > 0 && <div className="parse-warning review-warning">{reviewResult.preparedReview.document.warnings.map((warning, index) => <p key={`review-warning-${index}`}>{warning}</p>)}</div>}
              {reviewResult.visualAssets?.length > 0 && <div className="visual-review-notice"><b>{reviewResult.visualAssets.length} image{reviewResult.visualAssets.length === 1 ? '' : 's'} extracted</b><span>{[...new Set(reviewResult.visualAssets.map((item) => item.page))].map((page) => `page ${page}`).join(' · ')} · {reviewResult.logoReview?.notes || 'Visual review pending'}</span></div>}
              <div className="panel result-panel ai-panel summary-panel"><div className="result-panel-heading"><div><span className="result-index ai-index">✦</span><h3>Final review summary</h3></div><span className="result-count">{summaryResult?.status === 'error' ? 'CHECK FAILED' : 'COMPLETED'}</span></div>
                {summaryResult?.summary ? <MarkdownText className="markdown-summary">{summaryResult.summary}</MarkdownText> : <div className="ai-empty"><div className="ai-empty-icon">✦</div><b>{summaryResult?.status === 'error' ? 'Summary unavailable' : 'No summary returned'}</b><p>The deterministic results and specialist agent outputs are shown below.</p></div>}
                {summaryResult?.key_findings?.length > 0 && <div className="summary-key-findings"><b>Key findings</b><ul>{summaryResult.key_findings.map((finding, index) => <li key={`${index}-${finding}`}>{finding}</li>)}</ul></div>}
                {summaryResult?.summary && summaryResult?.status !== 'error' && <div className="correction-action">
                  <div><b>Create a corrected draft?</b><p>The correction agent will use the original upload and the findings above to prepare a DOCX draft. Review it before use.</p></div>
                  {correctedDocx
                    ? <button className="primary-button" type="button" onClick={downloadCorrection}>Download corrected DOCX <b>↓</b></button>
                    : <button className="primary-button" type="button" onClick={generateCorrection} disabled={correctionBusy || reviewBusy}>{correctionBusy ? <><span className="spinner" /> Generating…</> : <>Generate corrected DOCX <b>→</b></>}</button>}
                </div>}
                {correctedDocx && <p className="correction-success" role="status">Draft ready: {correctedDocx.editsApplied} edit{correctedDocx.editsApplied === 1 ? '' : 's'} applied{correctedDocx.unresolvedCount ? ` · ${correctedDocx.unresolvedCount} item${correctedDocx.unresolvedCount === 1 ? '' : 's'} need manual review` : ''}. {correctedDocx.formatNote || 'Formatting and visuals may differ from the uploaded file.'} Please review the draft before use.</p>}
              </div>
              <div className="results-grid">
                <div className="panel result-panel"><div className="result-panel-heading"><div><span className="result-index">A</span><h3>Deterministic checks</h3></div><span className="result-count">{deterministicFindings.length} FINDING{deterministicFindings.length === 1 ? '' : 'S'}</span></div>
                  {deterministicFindings.length ? <div className="finding-list">{deterministicFindings.map((finding) => <div className="finding" key={finding.finding_id}><div className="finding-severity">Severity: {finding.severity || 'not assigned'}</div><div><b>{finding.rule_id}</b><p>{finding.explanation}</p>{finding.quote && <blockquote>Draft excerpt: "{finding.quote}"</blockquote>}{finding.source && <><blockquote>"{finding.source.excerpt}"</blockquote><small>Evidence: {finding.source?.title || finding.source?.source_id} · version {finding.source?.version_id}</small></>}</div></div>)}</div> : <div className="no-findings"><span>✓</span><div><b>No deterministic findings</b><p>{reviewResult.preparedReview.deterministic.checks_run?.length ? 'Configured checks passed for the submitted text.' : 'No structured rules were configured for this brand and channel.'}</p></div></div>}
                  <div className="result-meta">Checks run: {reviewResult.preparedReview.deterministic.checks_run?.join(', ') || 'none'} · Retrieval: {reviewResult.preparedReview.retrieval_status}</div>
                </div>
              </div>
              <div className="agent-results-grid">
                <details className="panel result-panel specialist-panel specialist-details" open>
                  <summary className="result-panel-heading"><span className="specialist-summary-title"><span className="result-index">1</span><span className="agent-accordion-title">Rule extraction</span></span><span className="result-count">{extractionResult?.rules?.length || 0} RULE{extractionResult?.rules?.length === 1 ? '' : 'S'}</span></summary>
                  {extractionResult?.rules?.length ? <div className="agent-finding-list">{extractionResult.rules.map((rule) => <article className="agent-finding" key={rule.rule_id}><b>{rule.kind?.replaceAll('_', ' ')}</b><p>{rule.statement}</p>{(rule.citations || []).map((citation, index) => <article className="citation-card" key={`${rule.rule_id}-${citation.source_id}-${index}`}><div className="citation-heading"><b>{citation.source_title || citation.source_id}</b><span>v{citation.version_number ?? citation.version_id}</span></div><small>{citation.pages?.length ? `page${citation.pages.length > 1 ? 's' : ''} ${citation.pages.join(', ')}` : citation.page ? `page ${citation.page}` : 'page unavailable'}</small><blockquote>“{citation.quote}”</blockquote></article>)}</article>)}</div> : <div className="agent-empty"><b>{extractionResult?.classification === 'CHECK_FAILED' ? 'Extraction could not complete' : 'No supported rules extracted'}</b><p>{extractionResult?.notes || 'Retrieved sources did not contain explicit, relevant rules or approved facts.'}</p></div>}
                </details>
                {['tone', 'claims'].map((agentName) => {
                  const agent = agentResults[agentName];
                  const evidence = specialistEvidence[agentName] || [];
                  const findings = agent?.findings || [];
                  return <details className="panel result-panel specialist-panel specialist-details" key={agentName}>
                    <summary className="result-panel-heading"><span className="specialist-summary-title"><span className="result-index">{agentName === 'tone' ? '2' : '3'}</span><span className="agent-accordion-title">{agentName === 'tone' ? 'Tone agent' : 'Claims agent'}</span></span><span className="result-count">{findings.length} FINDING{findings.length === 1 ? '' : 'S'}</span></summary>
                    {findings.length ? <div className="agent-finding-list">{findings.map((finding, index) => {
                      return (
                        <article className="agent-finding" key={`${agentName}-${index}`}>
                          {agentName === 'tone' && <span className="finding-field-label">Reason</span>}
                          <MarkdownText className="agent-finding-title">{finding.claim || finding.assessment || 'Review finding'}</MarkdownText>
                          {(finding.explanation || finding.assessment) && <><span className="finding-field-label">{agentName === 'tone' ? 'Evidence-based explanation' : 'Explanation'}</span><MarkdownText className="agent-finding-copy">{finding.explanation || finding.assessment}</MarkdownText></>}
                          <small className="finding-severity">Severity: {finding.severity || 'not assigned'}</small>
                          {finding.document_quote && <blockquote>"{finding.document_quote}"</blockquote>}
                          {(finding.citations || []).length > 0 ? <div className="citation-list"><b>Evidence citation{finding.citations.length === 1 ? '' : 's'}</b>{finding.citations.map((citation, citationIndex) => {
                            const source = evidence.flatMap((bundle) => bundle.sources || []).find((candidate) => candidate.source_id === citation.source_id && candidate.version_id === citation.version_id && candidate.chunk_id === citation.chunk_id);
                            const pages = citation.pages?.length ? citation.pages : citation.page ? [citation.page] : source?.location?.pages || (source?.location?.page ? [source.location.page] : []);
                            return <article className="citation-card" key={`${citation.source_id}-${citation.chunk_id}-${citationIndex}`}><div className="citation-heading"><b>{citation.source_title || source?.title || citation.source_id}</b><span>v{citation.version_number ?? source?.version_number ?? citation.version_id}</span></div><small>{source?.document_type?.replaceAll('_', ' ') || 'approved source'} · chunk {citation.chunk_id}{pages.length ? ` · page${pages.length > 1 ? 's' : ''} ${pages.join(', ')}` : ''}</small><blockquote>"{citation.quote}"</blockquote></article>;
                          })}{finding.citation_validation?.rejected_count > 0 && <small className="citation-validation-note">{finding.citation_validation.rejected_count} citation(s) were omitted because they did not resolve to relevant retrieved evidence.</small>}</div> : <p className="citation-missing">{agent?.retrieval_status === 'failed'
                            ? 'Evidence retrieval failed for this agent; the result may be incomplete.'
                            : finding.citation_validation?.status === 'rejected'
                              ? 'The agent returned citation(s), but they did not match a retrieved source and were removed.'
                              : evidence.some((bundle) => bundle.sources?.length)
                                ? 'Evidence was retrieved, but the agent did not cite a relevant source for this finding.'
                                : 'No relevant source was retrieved for this finding.'}</p>}
                        </article>
                      );
                    })}</div> : <div className="agent-empty"><b>{agent?.classification === 'CHECK_FAILED' || agent?.status === 'error' ? 'Agent could not complete' : 'No specialist findings'}</b><p>{agent?.notes || (agent?.status === 'insufficient_evidence' ? 'The index did not return enough relevant approved evidence.' : 'This agent did not flag a finding.')}</p></div>}
                  </details>;
                })}
                <div className="panel result-panel specialist-panel logo-review-panel">
                  <div className="result-panel-heading"><div><span className="result-index">4</span><h3>Logo review</h3></div><span className="result-count">{logoAgentResult?.findings?.length || 0} OBSERVATION{logoAgentResult?.findings?.length === 1 ? '' : 'S'}</span></div>
                  {logoAgentResult?.findings?.length ? <div className="agent-finding-list">{logoAgentResult.findings.map((finding, index) => {
                    const reference = reviewResult.logoReferences?.find((item) => item.logo_id === finding.reference_logo_id);
                    return <article className="agent-finding" key={`${finding.review_image_id}-${index}`}>
                      <b>Logo candidate on page {finding.page}</b><p>{finding.observation}</p><small>Severity: not assessed for visual comparison</small><small>Visual comparison: {finding.comparison?.replaceAll('_', ' ')}</small>
                      {reference ? <div className="citation-list"><b>Approved logo reference</b><article className="citation-card"><div className="citation-heading"><b>{reference.title}</b><span>Reference</span></div><small>{reference.original_name} · reference {reference.logo_id}</small></article></div> : <p className="citation-missing">No approved logo reference was available for this comparison.</p>}
                    </article>;
                  })}</div> : <div className="agent-empty"><b>{logoAgentResult?.status === 'error' ? 'Logo agent could not complete' : logoAgentResult?.status === 'insufficient_evidence' ? 'Approved logo needed' : 'No likely logo matches found'}</b><p>{logoAgentResult?.notes || reviewResult.logoReview?.notes || 'Upload an approved logo reference to enable logo comparison.'}</p></div>}
                  {reviewResult.logoReferences?.length > 0 && <div className="logo-review-references"><b>Approved references in this session</b>{reviewResult.logoReferences.map((logo) => <small key={logo.logo_id}>{logo.title} · {logo.original_name}</small>)}</div>}
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
