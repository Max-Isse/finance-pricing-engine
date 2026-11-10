import pytest

pytest.importorskip("torch")

from pipelines.intraday_recalibration import run_intraday_recalibration  # noqa: E402
from pipelines.weekly_retrain import REPORT_FILE, run_weekly_retrain  # noqa: E402
from src.data.simulate import generate_pricing_data  # noqa: E402
from src.models.train import CHAMPION_FILE, INTRADAY_FILE  # noqa: E402
from src.utils.helpers import read_json  # noqa: E402


@pytest.fixture
def trained_dir(tmp_path):
    run_weekly_retrain(n_rows=2500, seed=42, epochs=200, artifact_dir=tmp_path)
    return tmp_path


def test_first_retrain_promotes_the_challenger(trained_dir):
    report = read_json(trained_dir / REPORT_FILE)
    assert report["promoted"] and report["champion_before"] is None
    assert (trained_dir / CHAMPION_FILE).exists()


def test_a_worse_challenger_is_not_promoted(trained_dir):
    before = (trained_dir / CHAMPION_FILE).read_bytes()
    report = run_weekly_retrain(n_rows=2500, seed=42, epochs=1, artifact_dir=trained_dir)
    assert not report["promoted"]
    assert report["challenger"]["rmse"] > report["champion_before"]["rmse"]
    assert (trained_dir / CHAMPION_FILE).read_bytes() == before


def test_no_recalibration_without_drift(trained_dir):
    baseline = generate_pricing_data(n_rows=2000, seed=42)
    recent = generate_pricing_data(n_rows=1000, seed=5)
    report = run_intraday_recalibration(baseline, recent, artifact_dir=trained_dir)
    assert not report["recalibrated"]
    assert not (trained_dir / INTRADAY_FILE).exists()


def test_recalibration_improves_the_model_after_a_regime_change(trained_dir):
    champion_before = (trained_dir / CHAMPION_FILE).read_bytes()
    report = run_intraday_recalibration(artifact_dir=trained_dir)
    assert report["recalibrated"] and report["accepted"]
    assert report["rmse_after"] < report["rmse_before"]
    assert (trained_dir / INTRADAY_FILE).exists()
    assert (trained_dir / CHAMPION_FILE).read_bytes() == champion_before


def test_recalibration_needs_a_champion(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_intraday_recalibration(artifact_dir=tmp_path)
