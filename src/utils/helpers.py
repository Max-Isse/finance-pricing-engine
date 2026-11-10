"""Shared helpers: project paths, seeding, JSON files and the optional Redis metrics cache."""
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
CONFIG_DIR = ROOT / "config"


def ensure_dir(path):
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    """Seed Python, NumPy and (when installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def _redis_client():
    import redis

    host = os.environ.get("REDIS_HOST", "redis-cache")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, db=0, socket_timeout=2, socket_connect_timeout=2)


def publish_metrics(metrics):
    """Write metrics to Redis. Returns False (without raising) when Redis is unavailable."""
    try:
        client = _redis_client()
        for key, value in metrics.items():
            client.set(key, str(value))
        return True
    except Exception as error:  # Redis is optional: pipelines must run without it
        print(f"[REDIS] Metrics not published ({type(error).__name__}).")
        return False


def read_metrics(keys):
    """Read metrics from Redis. Returns an empty dict when Redis is unavailable."""
    try:
        client = _redis_client()
        values = {}
        for key in keys:
            raw = client.get(key)
            if raw is not None:
                values[key] = raw.decode() if isinstance(raw, bytes) else str(raw)
        return values
    except Exception:
        return {}
