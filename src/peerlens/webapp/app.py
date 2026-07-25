"""Flask app: PDF upload + arXiv URL + editorial dashboard.

Endpoints
---------
GET  /                     Dashboard (upload form + processed papers)
POST /upload               Handle PDF upload(s) or arXiv URL submission
GET  /status/<job_id>      Poll status of a background job
GET  /jobs                 List all recent jobs
GET  /report/<doi>         Render the markdown referee report for a paper
GET  /dashboard-static     Return the freshly-rebuilt dashboard HTML

Design
------
The web app never invokes agents directly. It spawns analyze_paper.py
as a subprocess for each upload. This keeps the web layer resilient
to Gemini failures and allows multiple uploads to be processed in
parallel (each in its own process).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for, redirect

# Make peerlens importable
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from peerlens.settings import get_path
from peerlens.ingestion.arxiv import (
    normalize_arxiv_id,
    fetch_metadata as arxiv_fetch_metadata,
    arxiv_id_to_doi_label,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB per upload


# ---------------------------------------------------------------------------
# Job registry (in-memory; adequate for prototype)
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    doi_label: str
    source: str                # "pdf" | "arxiv"
    input_ref: str             # filename or arxiv id
    status: str = "queued"     # queued | running | done | failed
    score: int | None = None
    verdict: str | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Background dispatch
# ---------------------------------------------------------------------------

def _run_pipeline(job: Job) -> None:
    """Run analyze_paper.py as a subprocess and update job status."""
    with _JOBS_LOCK:
        job.status = "running"

    analyze_script = _ROOT / "scripts" / "analyze_paper.py"
    if job.source == "arxiv":
        cmd = [sys.executable, str(analyze_script), "--arxiv", job.input_ref]
    else:
        cmd = [sys.executable, str(analyze_script), "--pdf", job.input_ref,
               "--doi", job.doi_label]

    try:
        proc = subprocess.run(
            cmd,
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=900,   # 15 minutes
        )
        if proc.returncode != 0:
            with _JOBS_LOCK:
                job.status = "failed"
                job.error = (proc.stderr[-800:] or "unknown error").strip()
                job.finished_at = time.time()
            return
    except subprocess.TimeoutExpired:
        with _JOBS_LOCK:
            job.status = "failed"
            job.error = "pipeline timed out after 15 minutes"
            job.finished_at = time.time()
        return
    except Exception as e:
        with _JOBS_LOCK:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.finished_at = time.time()
        return

    # Read triage result
    triage_path = get_path("processed_data") / "triage_scores" / f"{job.doi_label}.json"
    if triage_path.exists():
        try:
            t = json.loads(triage_path.read_text())
            with _JOBS_LOCK:
                job.status = "done"
                job.score = t.get("score")
                job.verdict = t.get("verdict")
                job.finished_at = time.time()
            return
        except Exception:
            pass

    with _JOBS_LOCK:
        job.status = "failed"
        job.error = "pipeline finished but no triage_scores JSON produced"
        job.finished_at = time.time()


def _dispatch(job: Job) -> None:
    thread = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PeerLens — Editorial Triage</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 40px auto; max-width: 1180px;
         color: #222; background: #fafafa; }
  h1 { margin-bottom: 6px; }
  .subtitle { color: #666; margin-bottom: 24px; }

  .upload-card { background: white; border: 1px solid #ddd; border-radius: 8px;
                 padding: 20px 24px; margin-bottom: 28px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  .upload-card h2 { margin-top: 0; font-size: 1.1em; color: #333; }
  .upload-row { display: flex; gap: 20px; margin-bottom: 14px; align-items: flex-start; }
  .upload-col { flex: 1; }
  .upload-col label { display: block; font-weight: 600; margin-bottom: 6px; color: #444; font-size: 0.9em; }
  input[type=file], input[type=text] { width: 100%; padding: 6px 8px; box-sizing: border-box;
                                        border: 1px solid #ccc; border-radius: 4px; font-size: 0.95em; }
  button { padding: 8px 18px; background: #005ea6; color: white; border: 0; border-radius: 4px;
           font-weight: 600; cursor: pointer; font-size: 0.95em; }
  button:hover { background: #004b85; }
  .hint { color: #999; font-size: 0.8em; margin-top: 2px; }

  #jobs { margin-top: 20px; }
  .job-row { padding: 6px 10px; background: white; margin-bottom: 4px; border-radius: 4px;
             font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.85em;
             display: grid; grid-template-columns: 100px 1fr 100px 90px 100px; gap: 12px; align-items: center; }
  .status-queued { color: #666; }
  .status-running { color: #005ea6; }
  .status-done { color: #0a7c00; font-weight: 600; }
  .status-failed { color: #a00000; font-weight: 600; }

  iframe { border: 1px solid #ddd; border-radius: 8px; width: 100%; height: 900px; margin-top: 20px; }
</style>
</head>
<body>
<h1>PeerLens — Editorial Triage</h1>
<p class="subtitle">Automated pre-screening for journal submissions.</p>

<div class="upload-card">
  <h2>Submit manuscripts</h2>
  <form id="upload-form" enctype="multipart/form-data" method="post" action="/upload">
    <div class="upload-row">
      <div class="upload-col">
        <label>Upload PDF file(s)</label>
        <input type="file" name="pdfs" accept=".pdf" multiple>
        <div class="hint">One or more PDF files. Filenames become identifiers.</div>
      </div>
      <div class="upload-col">
        <label>Or paste an arXiv ID / URL</label>
        <input type="text" name="arxiv" placeholder="e.g. 2401.12345 or https://arxiv.org/abs/2401.12345">
        <div class="hint">The PDF will be fetched automatically.</div>
      </div>
    </div>
    <button type="submit">Submit for triage</button>
  </form>
</div>

<div id="jobs">
  <h2>Recent jobs</h2>
  <div id="job-list"></div>
</div>

<h2 style="margin-top: 32px;">Dashboard</h2>
<iframe src="/dashboard-static"></iframe>

<script>
document.getElementById('upload-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const resp = await fetch('/upload', { method: 'POST', body: fd });
  const data = await resp.json();
  if (data.error) {
    alert("Upload error: " + data.error);
    return;
  }
  form.reset();
  refreshJobs();
});

async function refreshJobs() {
  try {
    const resp = await fetch('/jobs');
    const jobs = await resp.json();
    const container = document.getElementById('job-list');
    if (!jobs.length) {
      container.innerHTML = '<div style="color:#999">No jobs yet.</div>';
      return;
    }
    let html = '';
    for (const j of jobs) {
      const statusClass = 'status-' + j.status;
      const verdict = j.status === 'done'
        ? `<span class="${statusClass}">${j.verdict?.toUpperCase() || ''}</span>`
        : `<span class="${statusClass}">${j.status}</span>`;
      const score = j.score != null ? `${j.score}/100` : '';
      html += `<div class="job-row">
        <span>${new Date(j.created_at*1000).toLocaleTimeString()}</span>
        <span>${j.doi_label} <span style="color:#999">(${j.source}: ${j.input_ref})</span></span>
        <span>${verdict}</span>
        <span>${score}</span>
        <span>${j.error ? '<span style="color:#a00000" title="'+j.error.replace(/"/g,'&quot;')+'">error</span>' : ''}</span>
      </div>`;
    }
    container.innerHTML = html;
  } catch (e) {
    console.warn(e);
  }
}

refreshJobs();
setInterval(refreshJobs, 3000);

// Reload iframe periodically when there are running jobs
setInterval(async () => {
  const resp = await fetch('/jobs');
  const jobs = await resp.json();
  const anyRunning = jobs.some(j => j.status === 'running' || j.status === 'queued');
  const anyRecentlyDone = jobs.some(j => j.status === 'done'
                                       && j.finished_at
                                       && (Date.now()/1000 - j.finished_at) < 20);
  if (anyRecentlyDone) {
    document.querySelector('iframe').src = '/dashboard-static?t=' + Date.now();
  }
}, 5000);
</script>
</body>
</html>
"""


REPORT_TEMPLATE = r"""
<!doctype html>
<html><head><meta charset="utf-8"><title>{{doi}} — report</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; color: #222; }
  pre { background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }
  a { color: #005ea6; }
  .back { margin-bottom: 20px; }
</style>
</head><body>
<div class="back"><a href="/">← Back to dashboard</a></div>
<pre>{{body}}</pre>
</body></html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(INDEX_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF or arXiv submissions."""
    arxiv_field = (request.form.get("arxiv") or "").strip()
    files = request.files.getlist("pdfs")

    dispatched: list[dict] = []

    # Handle arXiv input
    if arxiv_field:
        try:
            aid = normalize_arxiv_id(arxiv_field)
            doi_label = arxiv_id_to_doi_label(aid)
            job = Job(job_id=_new_job_id(), doi_label=doi_label,
                      source="arxiv", input_ref=aid)
            with _JOBS_LOCK:
                JOBS[job.job_id] = job
            _dispatch(job)
            dispatched.append({"job_id": job.job_id, "doi_label": doi_label, "source": "arxiv"})
        except Exception as e:
            return jsonify({"error": f"invalid arXiv input: {e}"}), 400

    # Handle PDF uploads
    incoming_dir = get_path("raw_data") / "pdf"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if not f.filename:
            continue
        if not f.filename.lower().endswith(".pdf"):
            continue
        # Sanitize filename to make doi_label
        stem = Path(f.filename).stem
        doi_label = "".join(c if c.isalnum() or c in ".-_" else "_" for c in stem) or "manuscript"
        dst = incoming_dir / f"{doi_label}.pdf"
        f.save(str(dst))
        job = Job(job_id=_new_job_id(), doi_label=doi_label,
                  source="pdf", input_ref=f.filename)
        with _JOBS_LOCK:
            JOBS[job.job_id] = job
        _dispatch(job)
        dispatched.append({"job_id": job.job_id, "doi_label": doi_label, "source": "pdf"})

    if not dispatched:
        return jsonify({"error": "no files or arXiv id supplied"}), 400

    return jsonify({"dispatched": dispatched})


@app.route("/jobs")
def list_jobs():
    with _JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: -j.created_at)[:30]
        return jsonify([asdict(j) for j in jobs])


@app.route("/status/<job_id>")
def job_status(job_id):
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        return jsonify(asdict(job))


@app.route("/dashboard-static")
def dashboard_static():
    """Rebuild and serve the editorial dashboard HTML."""
    dash_script = _ROOT / "scripts" / "build_editorial_dashboard.py"
    try:
        subprocess.run([sys.executable, str(dash_script)],
                       cwd=_ROOT, check=True, timeout=20,
                       capture_output=True, text=True)
    except Exception as e:
        return f"<pre>Dashboard build failed: {e}</pre>", 500

    dashboard_path = get_path("processed_data") / "editorial_dashboard.html"
    if not dashboard_path.exists():
        return "<pre>Dashboard not yet built. Process at least one paper.</pre>"
    return dashboard_path.read_text()


@app.route("/report/<doi>")
def report(doi):
    md_path = get_path("processed_data") / "referee_sections" / f"{doi}_full.md"
    if not md_path.exists():
        return f"<pre>No report for {doi}</pre>", 404
    return render_template_string(REPORT_TEMPLATE, doi=doi, body=md_path.read_text())


# The dashboard HTML contains links like "referee_sections/xxx_full.md"
# so we need to serve that directory as static.
@app.route("/referee_sections/<path:filename>")
def serve_report_file(filename):
    directory = get_path("processed_data") / "referee_sections"
    return send_from_directory(str(directory), filename)


if __name__ == "__main__":
    print("PeerLens web UI starting on http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
