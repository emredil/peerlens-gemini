"""Generic PDF-URL fetcher.

Downloads a PDF from any HTTP(S) URL that responds with a real PDF payload.
Works for ChemRxiv, bioRxiv, medRxiv, arXiv PDF links, personal websites,
publisher PDF endpoints — anything that returns application/pdf bytes.
"""
from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


def label_from_url(url: str) -> str:
    """Produce a stable, filesystem-safe stem for a URL.

    Prefers a recognisable last-segment slug when present; falls back to a
    short SHA-1 of the full URL otherwise.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.replace("www.", "").split(":")[0]
    host_slug = host.replace(".", "_")

    path = parsed.path.rstrip("/")
    tail = path.rsplit("/", 1)[-1] if "/" in path else path
    tail = tail.replace(".pdf", "").replace(".PDF", "")
    tail_clean = re.sub(r"[^A-Za-z0-9._-]+", "_", tail).strip("_")

    if tail_clean and len(tail_clean) <= 64:
        return f"{host_slug}_{tail_clean}"

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{host_slug}_{digest}"


def download_pdf(url: str, dest: Path, retries: int = 3,
                 timeout: int = 120) -> Path:
    """Download the PDF at `url` to `dest`. Validates %PDF magic before writing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (PeerLens/0.1 research prototype)",
                    "Accept": "application/pdf,*/*;q=0.5",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type", "")
            if len(data) < 4096:
                raise RuntimeError(
                    f"payload suspiciously small ({len(data)} bytes) — "
                    f"content-type={content_type}"
                )
            if not data.startswith(b"%PDF"):
                head = data[:80].decode("latin-1", errors="replace")
                raise RuntimeError(
                    f"payload lacks %PDF magic (content-type={content_type}, "
                    f"head={head!r}) — URL likely redirected to an HTML page"
                )
            dest.write_bytes(data)
            return dest
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"PDF download failed for {url}: {last_err!r}")
