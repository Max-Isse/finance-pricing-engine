"""Runs a Great Expectations-format suite with pandas.

The suite file (config/expectations_macro.json) keeps the Great Expectations JSON
layout, so it can be loaded by that library too. This validator covers the
expectation types the suite uses and fails loudly on any other type, so a new
expectation can never be skipped silently.
"""
import json
from pathlib import Path

import pandas as pd


def load_suite(path):
    return json.loads(Path(path).read_text())


def _not_null(frame, column, **_):
    unexpected = int(frame[column].isna().sum())
    return unexpected, "null values"


def _between(frame, column, min_value=None, max_value=None, **_):
    values = frame[column].dropna()
    outside = pd.Series(False, index=values.index)
    if min_value is not None:
        outside = outside | (values < min_value)
    if max_value is not None:
        outside = outside | (values > max_value)
    return int(outside.sum()), f"values outside [{min_value}, {max_value}]"


_CHECKS = {
    "expect_column_values_to_not_be_null": _not_null,
    "expect_column_values_to_be_between": _between,
}


def validate(frame, suite):
    """Run every expectation in `suite` against `frame`.

    Returns {"success": bool, "results": [...]}; each result has the expectation
    type, the column, whether it passed and how many rows broke it.
    """
    results = []
    for expectation in suite.get("expectations", []):
        kind = expectation["expectation_type"]
        kwargs = expectation.get("kwargs", {})
        column = kwargs.get("column")
        if kind not in _CHECKS:
            raise ValueError(f"Unsupported expectation type: {kind}")
        if column not in frame.columns:
            results.append({"expectation_type": kind, "column": column, "success": False,
                            "unexpected_count": len(frame), "detail": "column missing"})
            continue
        unexpected, detail = _CHECKS[kind](frame, **kwargs)
        results.append({"expectation_type": kind, "column": column, "success": unexpected == 0,
                        "unexpected_count": unexpected, "detail": detail})
    return {"success": all(result["success"] for result in results), "results": results}
