"""Intraday recalibration: when the yield distribution drifts, refit only the macro adjustment layer."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.drift_monitors import SIGNIFICANT_DRIFT, calculate_psi  # noqa: E402
from src.data.simulate import YIELD_COLUMN, generate_pricing_data  # noqa: E402
from src.models.train import (  # noqa: E402
    CHAMPION_FILE,
    INTRADAY_FILE,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from src.utils.helpers import ARTIFACT_DIR, publish_metrics  # noqa: E402


def run_intraday_recalibration(
    baseline=None,
    recent=None,
    artifact_dir=ARTIFACT_DIR,
    psi_threshold=SIGNIFICANT_DRIFT,
    epochs=200,
    lr=0.02,
    check_fraction=0.3,
):
    """Recalibrate the champion's gamma layer on `recent` data if yields have drifted.

    The recalibrated model is fitted on the first part of `recent` and checked on the
    rest. It is saved (as intraday.pt, next to the untouched champion) only when it
    lowers the RMSE on that check sample.
    """
    artifact_dir = Path(artifact_dir)
    if baseline is None:
        baseline = generate_pricing_data(n_rows=5000, seed=42)
    if recent is None:
        recent = generate_pricing_data(n_rows=1500, seed=7, yield_mean=0.049, rate_sensitivity=2.0)

    psi = calculate_psi(baseline[YIELD_COLUMN], recent[YIELD_COLUMN])
    report = {"psi": psi, "threshold": psi_threshold, "recalibrated": False, "accepted": False}
    if psi < psi_threshold:
        return report

    champion_path = artifact_dir / CHAMPION_FILE
    if not champion_path.exists():
        raise FileNotFoundError(f"No champion at {champion_path}. Run pipelines/weekly_retrain.py first.")
    model, scaler, _ = load_checkpoint(champion_path)

    split = int(len(recent) * (1 - check_fraction))
    fit_sample, check_sample = recent.iloc[:split], recent.iloc[split:]

    before = evaluate(model, scaler, check_sample)
    train_model(fit_sample, model=model, scaler=scaler, params="gamma", epochs=epochs, lr=lr)
    after = evaluate(model, scaler, check_sample)

    report.update(recalibrated=True, rmse_before=before["rmse"], rmse_after=after["rmse"],
                  accepted=after["rmse"] < before["rmse"])
    if report["accepted"]:
        save_checkpoint(artifact_dir / INTRADAY_FILE, model, scaler, after)
        publish_metrics({"model_rmse": f"{after['rmse']:.4f}", "pipeline_status": "RECALIBRATED"})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    result = run_intraday_recalibration(epochs=args.epochs)
    print(f"[RECALIBRATION] Yield PSI {result['psi']:.3f} (threshold {result['threshold']}).")
    if not result["recalibrated"]:
        print("[RECALIBRATION] No significant drift; model left unchanged.")
    else:
        print(f"[RECALIBRATION] RMSE {result['rmse_before']:.3f} -> {result['rmse_after']:.3f}; "
              + ("saved intraday model." if result["accepted"] else "no improvement, model left unchanged."))
