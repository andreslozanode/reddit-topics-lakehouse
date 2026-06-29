"""Lightweight, dependency-free data-quality framework."""

from reddit_lakehouse.quality.checks import (  # noqa: F401
    CheckResult,
    QualityReport,
    Rule,
    Severity,
    accepted_values,
    freshness,
    in_range,
    min_row_count,
    not_null,
    run_checks,
    unique,
)
