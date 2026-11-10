"""Writes a last-known-good macro snapshot when the upstream feed fails."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import DATA_DIR, ensure_dir, publish_metrics  # noqa: E402

FALLBACK_FILE = "fallback_hydrated_feed.csv"

# Last-known-good values used when no fresh data is available.
FALLBACK_SNAPSHOT = {
    "treasury_10y_yield": 0.0425,
    "implied_volatility": 0.22,
    "bid_ask_spread": 0.0005,
}


def run_fallback(reason, output_dir=DATA_DIR):
    """Write the fallback snapshot to `output_dir` and return the file path."""
    print(f"[FALLBACK LOG] Executing hard override pipeline due to: {reason}")
    frame = pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")], **{k: [v] for k, v in FALLBACK_SNAPSHOT.items()}})
    frame["fallback_reason"] = reason
    path = ensure_dir(output_dir) / FALLBACK_FILE
    frame.to_csv(path, index=False)
    publish_metrics({"pipeline_status": "FALLBACK", "fallback_reason": reason})
    print(f"[FALLBACK LOG] Wrote {path}. Ready for opening-bell processing.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", type=str, required=True)
    args = parser.parse_args()
    run_fallback(args.reason)
