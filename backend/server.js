import crypto from 'node:crypto';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import cors from 'cors';
import dotenv from 'dotenv';
import express from 'express';
import multer from 'multer';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
dotenv.config({ path: path.join(__dirname, '.env') });
const app = express();
const port = Number(process.env.PORT || 3001);
const organizationBrandId = process.env.TRUELINE_BRAND_ID || 'brand_lumina';
const organizationName = process.env.TRUELINE_ORGANIZATION_NAME || 'Lumina Studio';
const maxFileBytes = 25 * 1024 * 1024;
const documentTypes = new Set([
  'brand_guidelines', 'tone_of_voice', 'approved_claims', 'product_facts',
  'glossary', 'asset_rules', 'legal_compliance', 'other_reference',
]);

app.use(cors({
  origin: process.env.FRONTEND_ORIGIN || 'http://localhost:5173',
  exposedHeaders: ['X-Trueline-Edits-Applied', 'X-Trueline-Unresolved-Count', 'X-Trueline-Format-Note'],
}));
app.use(express.json({ limit: '1mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: maxFileBytes, files: 1 },
});
const logoUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024, files: 1 },
});

class PythonWorker {
  #child;
  #pending = new Map();
  #readyResolve;
  #readyReject;
  ready = new Promise((resolve, reject) => {
    this.#readyResolve = resolve;
    this.#readyReject = reject;
  });

  constructor() {
    const isWindows = process.platform === 'win32';
    const localVenvPython = isWindows
      ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
      : path.join(projectRoot, '.venv', 'bin', 'python');
    const python = process.env.PYTHON_BIN || (existsSync(localVenvPython) ? localVenvPython : 'python');
    this.#child = spawn(python, ['-u', path.join(__dirname, 'python_worker.py')], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PYTHONPATH: [projectRoot, path.join(projectRoot, 'data_ingestion'), process.env.PYTHONPATH]
          .filter(Boolean).join(path.delimiter),
      },
      stdio: ['pipe', 'pipe', 'inherit'],
    });
    const lines = readline.createInterface({ input: this.#child.stdout });
    lines.on('line', (line) => this.#onLine(line));
    this.#child.on('error', (error) => this.#failAll(error));
    this.#child.on('exit', (code) => this.#failAll(new Error(`Python worker exited (${code})`)));
  }

  #onLine(line) {
    let message;
    try { message = JSON.parse(line); } catch { return; }
    if (message.type === 'ready') {
      this.#readyResolve({ embedder: message.embedder, agentModels: message.agentModels });
      return;
    }
    if (message.type === 'fatal') {
      this.#readyReject(new Error(message.error || 'Python worker failed to start'));
      this.#failAll(new Error(message.error || 'Python worker failed'));
      return;
    }
    const pending = this.#pending.get(message.id);
    if (!pending) return;
    this.#pending.delete(message.id);
    if (message.ok) pending.resolve(message.result);
    else pending.reject(Object.assign(new Error(message.error || 'Python operation failed'), {
      status: message.status || 500,
    }));
  }

  #failAll(error) {
    this.#readyReject(error);
    for (const pending of this.#pending.values()) pending.reject(error);
    this.#pending.clear();
  }

  async call(operation, payload) {
    await this.ready;
    const id = crypto.randomUUID();
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#child.stdin.write(`${JSON.stringify({ id, operation, ...payload })}\n`, (error) => {
        if (error) {
          this.#pending.delete(id);
          reject(error);
        }
      });
    });
  }
}

const worker = new PythonWorker();
const allowedExtensions = new Set(['.txt', '.md', '.pdf', '.docx']);
const logoTypes = new Map([
  ['.png', 'image/png'], ['.jpg', 'image/jpeg'], ['.jpeg', 'image/jpeg'], ['.webp', 'image/webp'],
]);

function filePayload(file, { allowImages = false } = {}) {
  if (!file) throw Object.assign(new Error('Choose a file to upload.'), { status: 400 });
  const ext = path.extname(file.originalname).toLowerCase();
  const allowed = allowImages ? new Set([...allowedExtensions, ...logoTypes.keys()]) : allowedExtensions;
  if (!allowed.has(ext)) {
    throw Object.assign(new Error(allowImages
      ? 'Supported files are TXT, MD, PDF, DOCX, PNG, JPEG, and WebP.'
      : 'Supported files are TXT, MD, PDF, and DOCX.'), { status: 400 });
  }
  return {
    fileBase64: file.buffer.toString('base64'),
    fileName: path.basename(file.originalname),
    contentType: file.mimetype,
  };
}

function logoPayload(file) {
  if (!file) throw Object.assign(new Error('Choose a logo image to upload.'), { status: 400 });
  const fileName = path.basename(file.originalname);
  const ext = path.extname(fileName).toLowerCase();
  const expectedType = logoTypes.get(ext);
  if (!expectedType || file.mimetype !== expectedType) {
    throw Object.assign(new Error('Logo images must be PNG, JPEG, or WebP.'), { status: 400 });
  }
  return {
    fileBase64: file.buffer.toString('base64'),
    fileName,
    contentType: expectedType,
  };
}

function requiredField(value, label, maxLength = 160) {
  const cleaned = String(value || '').trim();
  if (!cleaned) throw Object.assign(new Error(`${label} is required.`), { status: 400 });
  if (cleaned.length > maxLength) {
    throw Object.assign(new Error(`${label} is too long.`), { status: 400 });
  }
  return cleaned;
}

app.get('/api/health', async (_req, res) => {
  try {
    const workerInfo = await worker.ready;
    res.json({ status: 'ready', python: 'ready', embedder: workerInfo.embedder,
      organizationName, workflow: 'langgraph', agentModels: workerInfo.agentModels });
  } catch (error) {
    res.status(503).json({ status: 'starting_or_failed', error: error.message });
  }
});

app.post('/api/session/start', async (_req, res, next) => {
  try {
    const session = await worker.call('start_session', { organizationId: organizationBrandId });
    res.status(201).json({ session });
  } catch (error) { next(error); }
});

app.post('/api/session/resume', async (req, res, next) => {
  try {
    const sessionId = requiredField(req.body?.sessionId, 'Session');
    const session = await worker.call('resume_session', {
      sessionId,
      organizationId: organizationBrandId,
    });
    res.json({ session });
  } catch (error) { next(error); }
});

app.post('/api/session/stop', async (req, res, next) => {
  try {
    const sessionId = requiredField(req.body?.sessionId, 'Session');
    const result = await worker.call('end_session', {
      sessionId,
      organizationId: organizationBrandId,
    });
    res.json(result);
  } catch (error) { next(error); }
});

app.get('/api/knowledge', async (req, res, next) => {
  try {
    const sessionId = requiredField(req.query.sessionId, 'Session');
    const sources = await worker.call('list_knowledge', {
      sessionId,
      organizationId: organizationBrandId,
    });
    res.json({ sources });
  } catch (error) { next(error); }
});

app.get('/api/logos', async (req, res, next) => {
  try {
    const logos = await worker.call('list_logos', {
      sessionId: requiredField(req.query.sessionId, 'Session'),
      organizationId: organizationBrandId,
    });
    res.json({ logos });
  } catch (error) { next(error); }
});

app.post('/api/logos', logoUpload.single('file'), async (req, res, next) => {
  try {
    const payload = logoPayload(req.file);
    const logo = await worker.call('upload_logo', {
      ...payload,
      sessionId: requiredField(req.body.sessionId, 'Session'),
      organizationId: organizationBrandId,
      title: String(req.body.title || '').trim(),
    });
    res.status(201).json({ logo });
  } catch (error) { next(error); }
});

app.get('/api/logos/:logoId/image', async (req, res, next) => {
  try {
    const image = await worker.call('get_logo_image', {
      logoId: requiredField(req.params.logoId, 'Logo ID'),
      sessionId: requiredField(req.query.sessionId, 'Session'),
      organizationId: organizationBrandId,
    });
    res.set('Cache-Control', 'private, no-store');
    res.type(image.contentType).send(Buffer.from(image.fileBase64, 'base64'));
  } catch (error) { next(error); }
});

app.delete('/api/logos/:logoId', async (req, res, next) => {
  try {
    const result = await worker.call('delete_logo', {
      logoId: requiredField(req.params.logoId, 'Logo ID'),
      sessionId: requiredField(req.query.sessionId, 'Session'),
      organizationId: organizationBrandId,
    });
    res.json(result);
  } catch (error) { next(error); }
});

app.post('/api/knowledge', upload.single('file'), async (req, res, next) => {
  try {
    const documentType = req.body.documentType;
    if (!documentTypes.has(documentType)) {
      throw Object.assign(new Error('Choose a supported document type.'), { status: 400 });
    }
    let ruleData = [];
    if (req.body.ruleData?.trim()) {
      try { ruleData = JSON.parse(req.body.ruleData); }
      catch { throw Object.assign(new Error('Rules must be valid JSON.'), { status: 400 }); }
      if (!Array.isArray(ruleData)) {
        throw Object.assign(new Error('Rules must be a JSON array.'), { status: 400 });
      }
    }
    const payload = filePayload(req.file);
    const result = await worker.call('ingest', {
      ...payload,
      sessionId: requiredField(req.body.sessionId, 'Session'),
      organizationId: organizationBrandId,
      title: requiredField(req.body.title || req.file.originalname, 'Title'),
      documentType,
      ruleData,
      approved: req.body.approved !== 'false',
    });
    res.status(201).json({ source: result });
  } catch (error) { next(error); }
});

app.post('/api/review', upload.single('file'), async (req, res, next) => {
  try {
    const file = filePayload(req.file, { allowImages: true });
    const review = await worker.call('review', {
      ...file,
      sessionId: requiredField(req.body.sessionId, 'Session'),
      organizationId: organizationBrandId,
      channel: String(req.body.channel || '').trim(),
    });
    res.json(review);
  } catch (error) { next(error); }
});

app.post('/api/review/correct', upload.single('file'), async (req, res, next) => {
  try {
    const file = filePayload(req.file, { allowImages: true });
    let reviewData;
    try { reviewData = JSON.parse(req.body.reviewData || ''); }
    catch { throw Object.assign(new Error('Completed review findings are required.'), { status: 400 }); }
    const result = await worker.call('correct', {
      ...file,
      sessionId: requiredField(req.body.sessionId, 'Session'),
      organizationId: organizationBrandId,
      reviewData,
    });
    const stem = path.parse(file.fileName).name.replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 100) || 'review';
    const filename = `${stem}-corrected.docx`;
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('X-Trueline-Edits-Applied', String(result.editsApplied || 0));
    res.setHeader('X-Trueline-Unresolved-Count', String(result.unresolvedFindings?.length || 0));
    res.setHeader('X-Trueline-Format-Note', encodeURIComponent(result.formatNote || ''));
    res.send(Buffer.from(result.fileBase64, 'base64'));
  } catch (error) { next(error); }
});

app.use((error, _req, res, _next) => {
  const status = error.status || (error instanceof multer.MulterError ? 400 : 500);
  res.status(status).json({ error: error.message || 'Unexpected server error.' });
});

const server = app.listen(port, () => {
  console.log(`TrueLine API listening on http://localhost:${port}`);
  worker.ready.then(({ embedder }) => console.log(`Python RAG worker ready (${embedder}).`))
    .catch((error) => console.error(`Python RAG worker failed: ${error.message}`));
});

function shutdown() {
  server.close();
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
