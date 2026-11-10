# Finance Pricing Engine: MLOps around a macro-aware pricing model

A small pricing model with the production plumbing around it: a data-quality gate, drift monitoring, a fallback feed for when upstream data fails, champion/challenger retraining, intraday recalibration, and a containerised inference service with a Redis metrics cache.

The model is a PyTorch network that prices an instrument from five features and adjusts the price for the 10-year Treasury yield:

```
price = base_head(hidden) + gamma_layer(hidden) × scaled_yield
```

Keeping the yield adjustment in its own layer means it can be recalibrated on its own when the rate regime moves, without retraining the rest of the network.

> **All data is simulated.** Nothing here uses real market data, and the model is a demonstration of the engineering, not a production pricing model.

## What each part does

| Component | File | What it does |
|---|---|---|
| Simulated data | `src/data/simulate.py` | Generates instrument features, yields and prices, plus a minute-by-minute macro feed. A `rate_sensitivity` parameter simulates a regime change. |
| Pricing network | `src/models/pricing_arch.py` | Two hidden layers, a base-price head and a yield adjustment head. Returns one price per row. |
| Training | `src/models/train.py` | Scaling, training, evaluation (RMSE, MAE) and checkpoints. Can train the whole network or only the yield adjustment layer. |
| Data-quality gate | `pipelines/quality_gate.py`, `src/quality/expectations.py` | Runs the expectation suite in `config/expectations_macro.json` and a freshness check on the macro feed. Exits non-zero on failure. |
| Fallback feed | `pipelines/fallback_injector.py` | Writes a last-known-good macro snapshot when the gate fails. |
| Drift monitor | `monitoring/drift_monitors.py` | Population stability index (PSI) between a baseline and a live sample, classified as stable, moderate or significant. |
| Weekly retrain | `pipelines/weekly_retrain.py` | Trains a challenger and promotes it only if it beats the current champion on the same holdout. |
| Intraday recalibration | `pipelines/intraday_recalibration.py` | If yield PSI is above 0.25, refits only the yield adjustment layer on recent data and keeps the result only if it lowers the error on a check sample. |
| Inference service | `src/service/app.py` | FastAPI app: `POST /price`, `GET /health`, `GET /metrics`. |
| Scheduled gate | `.github/workflows/morning_quality_gate.yml` | Runs the gate at 06:00 UTC on weekdays against a simulated feed and triggers the fallback on failure. |
| Tests | `tests/`, `.github/workflows/tests.yml` | pytest suite, run on every push and pull request. |
| Root-cause notebook | `notebooks/root_cause/` | Gradients of the price with respect to each input on a drifted sample. |

## Run it

```bash
pip install -r requirements.txt

make test          # run the test suite
make gate          # data-quality gate on a simulated feed
make train         # weekly retrain: writes artifacts/champion.pt
make simulate-drift  # PSI on a simulated yield shift
make recalibrate   # intraday recalibration: writes artifacts/intraday.pt
make serve         # API on http://localhost:8000 (docs at /docs)
```

Price one instrument:

```bash
curl -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.3, 0.0, 1.0], "treasury_10y_yield": 0.042}'
```

With Docker (service plus Redis):

```bash
docker-compose up --build -d
```

Redis is optional everywhere. Without it the pipelines run normally and skip publishing metrics.

Requires Python 3.10+.

## Design notes

- **Data-quality suite.** `config/expectations_macro.json` keeps the Great Expectations JSON layout, but the gate runs it with a small pandas validator (`src/quality/expectations.py`) so the gate needs only pandas and NumPy. The validator raises an error on any expectation type it doesn't support, so a new check can't be skipped silently.
- **PSI.** Bins are the baseline's quantiles and the outer bins are open-ended, so live values beyond the baseline's range still count. Empty bins are floored at 0.0001. On the simulated example (yields shifted from 4.1% to 4.9%) the score is about 7.8, far above the 0.25 alert level.
- **Champion and intraday models.** Recalibration never overwrites the champion. It saves `intraday.pt` next to it, and the service prefers the intraday model when one exists.

## Not included

- A feature store. `config/feature_store.yaml` is a Feast-style configuration only and nothing reads it.
- A real upstream data feed. The gate and the pipelines use simulated data.
- Authentication, request logging and model versioning beyond the two checkpoint files.

## Layout

```
src/data/simulate.py             simulated pricing data and macro feed
src/models/pricing_arch.py       pricing network (PyTorch)
src/models/train.py              training, evaluation, checkpoints
src/quality/expectations.py      validator for the expectation suite
src/service/app.py               FastAPI inference service
src/utils/helpers.py             paths, seeding, optional Redis metrics
pipelines/quality_gate.py        morning data-quality gate
pipelines/fallback_injector.py   last-known-good snapshot
pipelines/weekly_retrain.py      champion/challenger retrain
pipelines/intraday_recalibration.py  yield-layer recalibration on drift
monitoring/drift_monitors.py     PSI drift score
config/                          expectation suite, feature store config
notebooks/root_cause/            gradient-based root-cause analysis
tests/                           pytest suite
```
