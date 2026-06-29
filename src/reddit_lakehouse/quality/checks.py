"""Declarative data-quality rules evaluated against a Spark DataFrame.

A rule is a named predicate that returns the number of *violating* rows plus a
measured metric. Critical failures can abort a pipeline; warnings are logged and
surfaced in the report (and can be written to a quarantine/audit table).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"


@dataclass(frozen=True)
class Rule:
    name: str
    severity: Severity
    evaluate: Callable[[DataFrame], CheckResult]


@dataclass
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    violating_rows: int
    metric: float
    detail: str = ""


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.CRITICAL]

    def as_rows(self) -> list[dict]:
        return [r.__dict__ | {"severity": r.severity.value} for r in self.results]

    def summary(self) -> str:
        ok = sum(r.passed for r in self.results)
        return f"{ok}/{len(self.results)} checks passed"


# --------------------------------------------------------------------------- #
# Rule builders
# --------------------------------------------------------------------------- #
def not_null(columns: Sequence[str], severity: Severity = Severity.CRITICAL) -> Rule:
    cols = list(columns)

    def _eval(df: DataFrame) -> CheckResult:
        cond = None
        for c in cols:
            this = F.col(c).isNull()
            cond = this if cond is None else (cond | this)
        bad = df.filter(cond).count() if cond is not None else 0
        return CheckResult(
            name=f"not_null({', '.join(cols)})",
            severity=severity,
            passed=bad == 0,
            violating_rows=bad,
            metric=float(bad),
        )

    return Rule(name=f"not_null({','.join(cols)})", severity=severity, evaluate=_eval)


def unique(columns: Sequence[str], severity: Severity = Severity.CRITICAL) -> Rule:
    cols = list(columns)

    def _eval(df: DataFrame) -> CheckResult:
        total = df.count()
        distinct = df.select(*cols).distinct().count()
        dupes = total - distinct
        return CheckResult(
            name=f"unique({', '.join(cols)})",
            severity=severity,
            passed=dupes == 0,
            violating_rows=dupes,
            metric=float(dupes),
        )

    return Rule(name=f"unique({','.join(cols)})", severity=severity, evaluate=_eval)


def in_range(column: str, minimum: float, maximum: float, severity: Severity = Severity.WARNING) -> Rule:
    def _eval(df: DataFrame) -> CheckResult:
        bad = df.filter(
            F.col(column).isNotNull() & ((F.col(column) < minimum) | (F.col(column) > maximum))
        ).count()
        return CheckResult(
            name=f"in_range({column}, {minimum}, {maximum})",
            severity=severity,
            passed=bad == 0,
            violating_rows=bad,
            metric=float(bad),
        )

    return Rule(name=f"in_range({column})", severity=severity, evaluate=_eval)


def accepted_values(column: str, values: Sequence, severity: Severity = Severity.CRITICAL) -> Rule:
    allowed = list(values)

    def _eval(df: DataFrame) -> CheckResult:
        bad = df.filter(F.col(column).isNotNull() & ~F.col(column).isin(allowed)).count()
        return CheckResult(
            name=f"accepted_values({column})",
            severity=severity,
            passed=bad == 0,
            violating_rows=bad,
            metric=float(bad),
        )

    return Rule(name=f"accepted_values({column})", severity=severity, evaluate=_eval)


def min_row_count(minimum: int, severity: Severity = Severity.CRITICAL) -> Rule:
    def _eval(df: DataFrame) -> CheckResult:
        total = df.count()
        return CheckResult(
            name=f"min_row_count({minimum})",
            severity=severity,
            passed=total >= minimum,
            violating_rows=max(0, minimum - total),
            metric=float(total),
        )

    return Rule(name=f"min_row_count({minimum})", severity=severity, evaluate=_eval)


def max_null_ratio(column: str, threshold: float, severity: Severity = Severity.WARNING) -> Rule:
    def _eval(df: DataFrame) -> CheckResult:
        total = df.count()
        nulls = df.filter(F.col(column).isNull()).count() if total else 0
        ratio = (nulls / total) if total else 0.0
        return CheckResult(
            name=f"max_null_ratio({column}, {threshold})",
            severity=severity,
            passed=ratio <= threshold,
            violating_rows=nulls,
            metric=ratio,
            detail=f"null_ratio={ratio:.4f}",
        )

    return Rule(name=f"max_null_ratio({column})", severity=severity, evaluate=_eval)


def freshness(timestamp_col: str, max_age_days: int, severity: Severity = Severity.WARNING) -> Rule:
    def _eval(df: DataFrame) -> CheckResult:
        row = df.select(F.max(timestamp_col).alias("mx")).collect()
        latest = row[0]["mx"] if row else None
        if latest is None:
            return CheckResult(
                name=f"freshness({timestamp_col})",
                severity=severity,
                passed=False,
                violating_rows=0,
                metric=float("inf"),
                detail="no timestamp",
            )
        age_days = df.select(F.datediff(F.current_date(), F.lit(latest).cast("date")).alias("age")).collect()[
            0
        ]["age"]
        age_days = age_days if age_days is not None else 0
        return CheckResult(
            name=f"freshness({timestamp_col})",
            severity=severity,
            passed=age_days <= max_age_days,
            violating_rows=0,
            metric=float(age_days),
            detail=f"age_days={age_days}",
        )

    return Rule(name=f"freshness({timestamp_col})", severity=severity, evaluate=_eval)


def run_checks(df: DataFrame, rules: Sequence[Rule], *, cache: bool = True) -> QualityReport:
    """Evaluate all rules and return a consolidated report."""
    if cache:
        df = df.cache()
        df.count()  # materialize once for repeated scans
    report = QualityReport()
    for rule in rules:
        report.results.append(rule.evaluate(df))
    if cache:
        df.unpersist()
    return report
