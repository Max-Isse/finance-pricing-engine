"""Makes the repository root importable in tests and keeps tests away from Redis."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    """Point the metrics cache at an unreachable port so tests never touch a real Redis."""
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "1")
