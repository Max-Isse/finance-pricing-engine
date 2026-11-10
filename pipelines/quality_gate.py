"""Morning data-quality gate: runs the expectation suite and a freshness check on the macro feed.

Exits with status 1 when the gate fails, so a scheduler can trigger the fallback feed.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.simulate import generate_macro_feed  # noqa: E402
from src.quality.expectations import load_suite, validate  # noqa: E402
from src.utils.helpers import CONFIG_DIR, DATA_DIR, ensure_dir, publish_metrics  # noqa: E402

FEED_FILE = "macro_feed.csv"
SUITE_PATH = CONFIG_DIR / "expectations_macro.json"


def run_gate(feed_path, suite_path=SUITE_PATH, max_age_hours=24.0, now=None):
    """Validate the feed at `feed_path`. Returns a report with a top-level "success" flag."""
    feed_path = Path(feed_path)
    if not feed_path.exists():
        return {"success": False, "reason": f"feed missing: {feed_path}", "results": []}

    frame = pd.read_csv(feed_path)
    if frame.empty:
        return {"success": False, "reason": "feed is empty", "results": []}

    report = validate(frame, load_suite(suite_path))

    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if "timestamp" not in frame.columns:
        report.update(success=False, reason="feed has no timestamp column")
        return report
    latest = pd.to_datetime(frame["timestamp"], utc=True).max()
    age_hours = (now - latest).total_seconds() / 3600
    report["age_hours"] = round(age_hours, 2)
    if age_hours > max_age_hours:
        report.update(success=False, reason=f"feed is stale ({age_hours:.1f}h old, limit {max_age_hours}h)")
    elif not report["success"]:
        report["reason"] = "expectation suite failed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", default=str(DATA_DIR / FEED_FILE))
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--simulate", action="store_true", help="write a fresh simulated feed before checking")
    args = parser.parse_args()

    if args.simulate:
        ensure_dir(Path(args.feed).parent)
        generate_macro_feed().to_csv(args.feed, index=False)
        print(f"[GATE] Wrote simulated feed to {args.feed}")

    outcome = run_gate(args.feed, max_age_hours=args.max_age_hours)
    for result in outcome["results"]:
        mark = "PASS" if result["success"] else "FAIL"
        print(f"[GATE] {mark} {result['expectation_type']}({result['column']}): "
              f"{result['unexpected_count']} {result['detail']}")
    if outcome["success"]:
        publish_metrics({"pipeline_status": "HEALTHY"})
        print("[GATE] Passed.")
        sys.exit(0)
    publish_metrics({"pipeline_status": "GATE_FAILED"})
    print(f"[GATE] Failed: {outcome.get('reason')}")
    sys.exit(1)
