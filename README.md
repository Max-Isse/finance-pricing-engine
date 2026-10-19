# Finance Pricing Engine: MLOps scaffolding for a macro-aware pricing model

The production plumbing around a pricing model: data-quality gates, drift monitoring, a fallback feed when upstream data fails, and a containerised service with a Redis cache. The pricing model itself is a small PyTorch network whose output is adjusted by a macro input (the 10-year Treasury yield).

> **Status: work in progress.** The monitoring, fallback and container pieces run. Retraining, intraday recalibration, the feature store and the tests are placeholders. The table below shows exactly what works today. All data is simulated; nothing here uses real market data.

## What works today

| Component | File | Status |
|---|---|---|
| Drift monitor: population stability index (PSI) between a baseline and a live feature distribution, written to Redis | `monitoring/drift_monitors.py` | Runs. On its simulated example (yields shifted from 4.1% to 4.9%) it reports PSI ≈ 14.1, far above the usual 0.25 alert level. |
| Fallback feed: writes a last-known-good market snapshot (10y yield, implied volatility, spread) when upstream data fails | `pipelines/fallback_injector.py` | Runs |
| Data-quality expectations: 10y yield not null, implied volatility between 2% and 180% | `config/expectations_macro.json` | Defined (Great Expectations format), not yet executed by the gate |
| Morning quality gate: scheduled GitHub Actions job (06:00 UTC, weekdays) that triggers the fallback feed on failure | `.github/workflows/morning_quality_gate.yml` | Runs; currently checks only that the expectations file exists |
| Containerised service with Redis cache | `Dockerfile`, `docker-compose.yml` | Builds and starts the drift monitor next to Redis |
| Pricing network: two hidden layers plus a yield-scaled adjustment layer | `src/models/pricing_arch.py` | Forward pass runs; see known issues |
| Root-cause notebook: gradients of the price with respect to features and yield, to trace a drift incident | `notebooks/root_cause/root_cause_analysis_2026Q2.ipynb` | Code written; the saved output is an import error from an environment without PyTorch |

## Not built yet

- `pipelines/weekly_retrain.py` and `pipelines/intraday_recalibration.py` (empty)
- Feature store (`config/feature_store.yaml` is a Feast-style configuration only)
- Tests (`make test` expects `src/tests/`, which does not exist yet)
- `monitoring/dashboard_layouts.json` and `src/utils/helpers.py` (empty)

## Known issues

- `MacroCalibratedPricingNet` returns one value per hidden unit instead of one price per row: with `hidden_dim=16` the output shape is `(n, 16)`, not `(n, 1)`. The yield adjustment `(n, 1)` is broadcast onto the hidden layer instead of a single base price. A final linear layer to `output_dim` before the adjustment would fix it.
- The CI gate installs `great_expectations` but does not run the expectation suite yet.

## Run it

```bash
pip install -r requirements.txt
python monitoring/drift_monitors.py                        # PSI on simulated drift (Redis optional)
python pipelines/fallback_injector.py --reason "Stale Upstream Data"
docker-compose up --build -d                               # service plus Redis
```

Requires Python 3.10+, PyTorch 2.2, pandas, NumPy, SciPy, Great Expectations 0.18 and redis-py.

## Layout

```
src/models/pricing_arch.py     macro-calibrated pricing network (PyTorch)
monitoring/drift_monitors.py   PSI drift score, pushed to Redis
pipelines/fallback_injector.py last-known-good market snapshot when upstream data fails
pipelines/                     weekly retrain and intraday recalibration (planned)
config/                        data-quality expectations, feature store config
notebooks/root_cause/          gradient-based root-cause analysis
.github/workflows/             scheduled morning quality gate
Dockerfile, docker-compose.yml containerised service with Redis
```
