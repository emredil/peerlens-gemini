"""Password gate + in-memory rate limiting for the PeerLens demo web app.

Two decorators:
  @requires_auth  -- redirect to /login when session cookie missing.
  @rate_limited   -- 429 when the current session or IP exceeded quotas.

Configuration via environment variables:
  DEMO_PASSWORD              single shared password; set on fly secrets in prod.
  DEMO_SESSION_DAILY_LIMIT   analyses per session per 24h  (default 5)
  DEMO_IP_HOURLY_LIMIT       analyses per IP  per 1h       (default 3)
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from functools import wraps
from threading import Lock

from flask import request, session, redirect, url_for, jsonify, render_template_string


SESSION_DAILY_LIMIT = int(os.environ.get("DEMO_SESSION_DAILY_LIMIT", "5"))
IP_HOURLY_LIMIT = int(os.environ.get("DEMO_IP_HOURLY_LIMIT", "3"))
DAY_SECONDS = 24 * 3600
HOUR_SECONDS = 3600


_session_history: dict[str, deque] = defaultdict(deque)
_ip_history: dict[str, deque] = defaultdict(deque)
_lock = Lock()


def _prune(dq: deque, window: int) -> None:
    cutoff = time.time() - window
    while dq and dq[0] < cutoff:
        dq.popleft()


def _client_ip() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _session_key() -> str:
    return session.get("session_id", "no-session")


def requires_auth(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            if request.method == "GET":
                return redirect(url_for("login", next=request.path))
            return jsonify({"error": "authentication required"}), 401
        return f(*args, **kwargs)
    return wrapped


def rate_limited(f):
    """Increments session + IP counters BEFORE the handler runs.
    Handler must succeed for the count to stay (no rollback on failure —
    generous behaviour; a failed run still counts against quota).
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        skey = _session_key()
        ip = _client_ip()
        now = time.time()

        with _lock:
            sdq = _session_history[skey]
            idq = _ip_history[ip]
            _prune(sdq, DAY_SECONDS)
            _prune(idq, HOUR_SECONDS)

            if len(sdq) >= SESSION_DAILY_LIMIT:
                reset_in = int(DAY_SECONDS - (now - sdq[0]))
                return jsonify({
                    "error": "session daily quota exceeded",
                    "limit": SESSION_DAILY_LIMIT,
                    "window_seconds": DAY_SECONDS,
                    "reset_in_seconds": max(0, reset_in),
                    "message": f"You have used {SESSION_DAILY_LIMIT} analyses in the past 24 hours. Try again in {max(0, reset_in) // 3600} hour(s).",
                }), 429

            if len(idq) >= IP_HOURLY_LIMIT:
                reset_in = int(HOUR_SECONDS - (now - idq[0]))
                return jsonify({
                    "error": "ip hourly quota exceeded",
                    "limit": IP_HOURLY_LIMIT,
                    "window_seconds": HOUR_SECONDS,
                    "reset_in_seconds": max(0, reset_in),
                    "message": f"Your IP has submitted {IP_HOURLY_LIMIT} analyses in the past hour. Try again in {max(0, reset_in) // 60} minute(s).",
                }), 429

            sdq.append(now)
            idq.append(now)

        return f(*args, **kwargs)
    return wrapped


LOGIN_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>PeerLens Demo — Sign In</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #f6f7fa; margin: 0; padding: 4rem 1rem; }
  .box { max-width: 380px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
  h1 { margin-top: 0; font-size: 1.25rem; }
  p.tag { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
  input[type=password] { width: 100%; padding: 0.6rem 0.8rem; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; box-sizing: border-box; }
  button { margin-top: 1rem; width: 100%; padding: 0.7rem; background: #2c5aa0; color: white; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }
  button:hover { background: #1e4380; }
  .err { color: #b00020; margin-top: 0.75rem; font-size: 0.9rem; }
  .note { color: #888; font-size: 0.8rem; margin-top: 1.5rem; text-align: center; }
</style>
</head>
<body>
  <div class="box">
    <h1>PeerLens — Demo Access</h1>
    <p class="tag">This is a research prototype. Access is by shared password.</p>
    <form method="POST">
      <input type="hidden" name="next" value="{{ next_url }}">
      <input type="password" name="password" placeholder="Access password" autofocus required>
      <button type="submit">Sign in</button>
      {% if error %}<div class="err">{{ error }}</div>{% endif %}
    </form>
    <div class="note">Editorial preview for AI-assisted peer-review triage.</div>
  </div>
</body>
</html>"""


def register_auth_routes(app):
    """Attach /login and /logout endpoints to the given Flask app."""
    demo_password = os.environ.get("DEMO_PASSWORD", "")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        next_url = request.args.get("next") or request.form.get("next") or "/"
        if request.method == "POST":
            entered = request.form.get("password", "")
            if not demo_password:
                error = "Server misconfigured: DEMO_PASSWORD not set."
            elif entered == demo_password:
                session["authed"] = True
                import uuid as _uuid
                session["session_id"] = _uuid.uuid4().hex[:16]
                return redirect(next_url)
            else:
                error = "Incorrect password."
        return render_template_string(LOGIN_PAGE, next_url=next_url, error=error)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))
