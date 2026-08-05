"""Launch the PeerLens web UI.

Environment variables:
  PORT   listen port (default 5001 locally; Fly.io sets 8080 in the platform)
  HOST   bind address (default 127.0.0.1 locally; set to 0.0.0.0 for containers)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.webapp.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"PeerLens web UI: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
