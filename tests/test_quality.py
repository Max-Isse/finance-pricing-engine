import numpy as np
import pandas as pd
import pytest

from pipelines.fallback_injector import FALLBACK_FILE, run_fallback
from pipelines.quality_gate import SUITE_PATH, run_gate
from src.data.simulate import FEATURE_COLUMNS, TARGET_COLUMN, YIELD_COLUMN, generate_macro_feed, generate_pricing_data
from src.quality.expectations import load_suite, validate


@pytest.fixture
def suite():
    return load_suite(SUITE_PATH)


def test_simulated_pricing_data_is_reproducible():
    first, second = generate_pricing_data(200, seed=3), generate_pricing_data(200, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == FEATURE_COLUMNS + [YIELD_COLUMN, TARGET_COLUMN]
    assert not first.isna().any().any()


def test_clean_feed_passes_the_suite(suite):
    report = validate(generate_macro_feed(seed=0), suite)
    assert report["success"]
    assert len(report["results"]) == len(suite["expectations"])


def test_null_yield_fails_the_suite(suite):
    feed = generate_macro_feed(seed=0)
    feed.loc[3, YIELD_COLUMN] = np.nan
    report = validate(feed, suite)
    assert not report["success"]
    failed = [r for r in report["results"] if not r["success"]]
    assert failed[0]["column"] == YIELD_COLUMN and failed[0]["unexpected_count"] == 1


def test_volatility_outside_range_fails_the_suite(suite):
    feed = generate_macro_feed(seed=0)
    feed.loc[0, "implied_volatility"] = 2.5
    feed.loc[1, "implied_volatility"] = 0.001
    report = validate(feed, suite)
    assert not report["success"]
    assert sum(r["unexpected_count"] for r in report["results"]) == 2


def test_missing_column_fails_the_suite(suite):
    report = validate(generate_macro_feed(seed=0).drop(columns=["implied_volatility"]), suite)
    assert not report["success"]


def test_unknown_expectation_type_raises():
    bad_suite = {"expectations": [{"expectation_type": "expect_magic", "kwargs": {"column": "x"}}]}
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"x": [1]}), bad_suite)


def test_gate_passes_on_a_fresh_feed(tmp_path):
    path = tmp_path / "feed.csv"
    generate_macro_feed(seed=0).to_csv(path, index=False)
    report = run_gate(path)
    assert report["success"], report


def test_gate_fails_on_a_stale_feed(tmp_path):
    path = tmp_path / "feed.csv"
    generate_macro_feed(seed=0, end="2026-01-05 16:00").to_csv(path, index=False)
    report = run_gate(path, now="2026-01-07 06:00+00:00")
    assert not report["success"]
    assert "stale" in report["reason"]


def test_gate_fails_when_the_feed_is_missing(tmp_path):
    assert not run_gate(tmp_path / "missing.csv")["success"]


def test_fallback_writes_a_feed_that_passes_the_suite(tmp_path, suite):
    path = run_fallback("Stale Upstream Data", output_dir=tmp_path / "data")
    assert path.name == FALLBACK_FILE
    feed = pd.read_csv(path)
    assert len(feed) == 1 and feed.loc[0, "fallback_reason"] == "Stale Upstream Data"
    assert validate(feed, suite)["success"]
    assert run_gate(path)["success"]
