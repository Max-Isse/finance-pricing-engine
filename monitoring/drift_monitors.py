"""Population stability index (PSI) drift monitor for the macro inputs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import publish_metrics  # noqa: E402

MODERATE_DRIFT = 0.10
SIGNIFICANT_DRIFT = 0.25


def calculate_psi(baseline, target, num_bins=10, eps=1e-4):
    """PSI between a baseline sample and a live sample.

    Bins are the baseline's quantiles, with the two outer bins left open-ended so
    live values beyond the baseline's range are still counted. Empty bins are
    floored at `eps` to keep the logarithm finite.
    """
    baseline = np.asarray(baseline, dtype=float)
    target = np.asarray(target, dtype=float)
    if baseline.size == 0 or target.size == 0:
        raise ValueError("PSI needs non-empty baseline and target samples.")

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, num_bins + 1)))
    if edges.size < 2:
        return 0.0
    inner_edges = edges[1:-1]

    def bin_shares(values):
        counts = np.bincount(np.searchsorted(inner_edges, values, side="right"), minlength=inner_edges.size + 1)
        return np.clip(counts / values.size, eps, None)

    baseline_share, target_share = bin_shares(baseline), bin_shares(target)
    return float(np.sum((target_share - baseline_share) * np.log(target_share / baseline_share)))


def classify_drift(psi):
    if psi < MODERATE_DRIFT:
        return "STABLE"
    if psi < SIGNIFICANT_DRIFT:
        return "MODERATE_DRIFT"
    return "SIGNIFICANT_DRIFT"


def run_drift_check(baseline, live, publish=True):
    """Compute PSI, classify it and (optionally) publish both to Redis."""
    score = calculate_psi(baseline, live)
    status = classify_drift(score)
    if publish:
        publish_metrics({
            "latest_drift_score": f"{score:.4f}",
            "pipeline_status": "HEALTHY" if status == "STABLE" else status,
        })
    return {"psi": score, "status": status}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    historical_yields = rng.normal(loc=0.041, scale=0.002, size=1000)
    live_yields_drifted = rng.normal(loc=0.049, scale=0.002, size=1000)

    result = run_drift_check(historical_yields, live_yields_drifted)
    print(f"DRIFT_CALCULATION_SCORE: {result['psi']:.4f} ({result['status']})")
