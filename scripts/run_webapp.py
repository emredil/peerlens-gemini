"""Launch the PeerLens web UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.webapp.app import app

if __name__ == "__main__":
    print("PeerLens web UI: http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
