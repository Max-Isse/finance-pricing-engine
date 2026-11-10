"""Simulated data. Nothing in this project uses real market data."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"feature_{i}" for i in range(5)]
YIELD_COLUMN = "treasury_10y_yield"
TARGET_COLUMN = "price"
REFERENCE_YIELD = 0.0425


def generate_pricing_data(
    n_rows=5000,
    seed=42,
    yield_mean=0.041,
    yield_std=0.002,
    rate_sensitivity=1.0,
    noise_std=0.5,
):
    """Simulate instrument features, the 10-year yield and a price.

    price = base(features) + sensitivity(features) * (yield - REFERENCE_YIELD) + noise

    The sensitivity is negative (prices fall when yields rise). `rate_sensitivity`
    scales it, which lets a test or a pipeline simulate a regime change.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, len(FEATURE_COLUMNS)))
    yields = np.clip(rng.normal(yield_mean, yield_std, size=n_rows), 0.001, None)

    base = 100 + 8 * x[:, 0] - 5 * x[:, 1] + 3 * x[:, 2] * x[:, 3] + 2 * np.tanh(x[:, 4])
    sensitivity = -(400 + 60 * x[:, 0] + 40 * x[:, 1] ** 2) * rate_sensitivity
    price = base + sensitivity * (yields - REFERENCE_YIELD) + rng.normal(0, noise_std, size=n_rows)

    frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    frame[YIELD_COLUMN] = yields
    frame[TARGET_COLUMN] = price
    return frame


def generate_macro_feed(n_rows=390, seed=None, yield_mean=0.041, end=None):
    """Simulate a minute-by-minute macro feed ending at `end` (default: now, UTC)."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(end=end, periods=n_rows, freq="min"),
            YIELD_COLUMN: np.clip(rng.normal(yield_mean, 0.002, size=n_rows), 0.001, None),
            "implied_volatility": np.clip(rng.lognormal(np.log(0.22), 0.15, size=n_rows), 0.03, 1.5),
            "bid_ask_spread": np.abs(rng.normal(0.0005, 0.0001, size=n_rows)),
        }
    )
