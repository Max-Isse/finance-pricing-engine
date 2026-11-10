"""Weekly retrain: fit a challenger and promote it only if it beats the current champion."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.simulate import generate_pricing_data  # noqa: E402
from src.models.train import CHAMPION_FILE, evaluate, load_checkpoint, save_checkpoint, train_model  # noqa: E402
from src.utils.helpers import ARTIFACT_DIR, ensure_dir, publish_metrics, write_json  # noqa: E402

REPORT_FILE = "last_retrain.json"


def run_weekly_retrain(data=None, n_rows=5000, seed=42, epochs=300, artifact_dir=ARTIFACT_DIR, holdout_fraction=0.2):
    """Train a challenger on `data` (simulated when None) and compare it with the champion.

    Both models are scored on the same holdout (the last `holdout_fraction` of rows).
    The challenger replaces the champion only when its RMSE is lower.
    """
    artifact_dir = ensure_dir(artifact_dir)
    frame = generate_pricing_data(n_rows=n_rows, seed=seed) if data is None else data
    split = int(len(frame) * (1 - holdout_fraction))
    train, holdout = frame.iloc[:split], frame.iloc[split:]

    challenger, scaler = train_model(train, epochs=epochs, seed=seed)
    challenger_metrics = evaluate(challenger, scaler, holdout)

    champion_path = artifact_dir / CHAMPION_FILE
    champion_metrics = None
    if champion_path.exists():
        champion, champion_scaler, _ = load_checkpoint(champion_path)
        champion_metrics = evaluate(champion, champion_scaler, holdout)

    promoted = champion_metrics is None or challenger_metrics["rmse"] < champion_metrics["rmse"]
    if promoted:
        save_checkpoint(champion_path, challenger, scaler, challenger_metrics)

    report = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "challenger": challenger_metrics,
        "champion_before": champion_metrics,
        "promoted": promoted,
    }
    write_json(artifact_dir / REPORT_FILE, report)
    live_rmse = challenger_metrics["rmse"] if promoted else champion_metrics["rmse"]
    publish_metrics({"model_rmse": f"{live_rmse:.4f}", "last_retrain": report["trained_at"]})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    result = run_weekly_retrain(n_rows=args.rows, seed=args.seed, epochs=args.epochs)
    print(f"[RETRAIN] Challenger RMSE {result['challenger']['rmse']:.3f} on {result['holdout_rows']} holdout rows.")
    if result["champion_before"]:
        print(f"[RETRAIN] Champion RMSE {result['champion_before']['rmse']:.3f}.")
    print("[RETRAIN] Challenger promoted." if result["promoted"] else "[RETRAIN] Champion kept.")
