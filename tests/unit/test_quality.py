from reddit_lakehouse.quality import (
    Severity,
    accepted_values,
    in_range,
    min_row_count,
    not_null,
    run_checks,
    unique,
)
from reddit_lakehouse.transform.silver import to_events


def test_quality_passes_on_clean_events(raw_posts, raw_comments):
    events = to_events(raw_posts, raw_comments, topic="sample")
    rules = [
        not_null(["event_id", "event_type"]),
        unique(["event_id"]),
        accepted_values("event_type", ["post", "comment"]),
        min_row_count(1),
    ]
    report = run_checks(events, rules)
    assert report.passed
    assert not report.critical_failures


def test_quality_detects_violation(spark):
    df = spark.createDataFrame([(1, None), (2, "x"), (2, "y")], ["id", "val"])
    report = run_checks(
        df,
        [not_null(["val"]), unique(["id"]), in_range("id", 0, 1, severity=Severity.WARNING)],
    )
    names = {r.name: r for r in report.results}
    assert not names["not_null(val)"].passed
    assert not names["unique(id)"].passed
    assert not report.passed
