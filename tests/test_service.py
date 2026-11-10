import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from pipelines.weekly_retrain import run_weekly_retrain  # noqa: E402
from src.service.app import create_app  # noqa: E402

PAYLOAD = {"features": [0.1, -0.2, 0.3, 0.0, 1.0], "treasury_10y_yield": 0.042}


def test_health_reports_no_model_before_training(tmp_path):
    response = TestClient(create_app(tmp_path)).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": None}


def test_price_returns_503_before_training(tmp_path):
    assert TestClient(create_app(tmp_path)).post("/price", json=PAYLOAD).status_code == 503


def test_price_returns_a_number_after_training(tmp_path):
    run_weekly_retrain(n_rows=1000, seed=42, epochs=50, artifact_dir=tmp_path)
    client = TestClient(create_app(tmp_path))
    body = client.post("/price", json=PAYLOAD).json()
    assert isinstance(body["price"], float)
    assert body["model"] == "champion.pt"
    assert client.get("/health").json()["model"] == "champion.pt"


def test_price_rejects_the_wrong_number_of_features(tmp_path):
    bad = {"features": [0.1, 0.2], "treasury_10y_yield": 0.042}
    assert TestClient(create_app(tmp_path)).post("/price", json=bad).status_code == 422


def test_metrics_is_empty_without_redis(tmp_path):
    assert TestClient(create_app(tmp_path)).get("/metrics").json() == {}
