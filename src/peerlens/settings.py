"""Central settings loader. Reads config/settings.yaml and .env."""
from __future__ import annotations

from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def get_path(key: str) -> Path:
    rel = CONFIG["paths"][key]
    p = PROJECT_ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    return key
