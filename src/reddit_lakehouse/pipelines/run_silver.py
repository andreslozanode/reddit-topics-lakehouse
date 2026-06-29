"""Silver job: bronze raw -> cleaned, unified events with DQ gating."""

from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame

from reddit_lakehouse.common.io import read_table, table_exists
from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.common.spark import get_spark
from reddit_lakehouse.common.tables import optimize_table, write_delta
from reddit_lakehouse.pipelines._common import resolve_config
from reddit_lakehouse.quality import (
    accepted_values,
    in_range,
    not_null,
    run_checks,
    unique,
)
from reddit_lakehouse.transform.silver import to_events

logger = get_logger(__name__)


def main() -> None:
    cfg = resolve_config()
    spark = get_spark("silver")

    posts_fqn = cfg.bronze("posts_raw")
    comments_fqn = cfg.bronze("comments_raw")

    per_topic: list[DataFrame] = []
    for source in cfg.sources:
        posts = None
        comments = None
        if table_exists(spark, posts_fqn):
            posts = read_table(spark, posts_fqn).filter(f"_topic = '{source.topic}'")
            posts = posts if posts.take(1) else None
        if table_exists(spark, comments_fqn):
            comments = read_table(spark, comments_fqn).filter(f"_topic = '{source.topic}'")
            comments = comments if comments.take(1) else None
        if posts is None and comments is None:
            continue
        per_topic.append(to_events(posts, comments, topic=source.topic))

    if not per_topic:
        raise RuntimeError("No bronze data found to build silver events")

    events = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), per_topic)

    rules = [
        not_null(["event_id", "event_type", "topic", "created_ts"]),
        unique(["event_id"]),
        accepted_values("event_type", ["post", "comment"]),
        in_range("score", cfg.dq.min_score, cfg.dq.max_score),
    ]
    report = run_checks(events, rules)
    logger.info("Silver DQ: %s", report.summary())
    if report.critical_failures:
        raise RuntimeError(f"Silver DQ failed: {[r.name for r in report.critical_failures]}")

    write_delta(
        events,
        cfg.silver("events"),
        mode="merge",
        merge_keys=["event_id"],
        cluster_by=["topic", "subreddit", "created_date"],
        comment="Cleaned, unified Reddit events (posts + comments).",
    )
    optimize_table(spark, cfg.silver("events"), zorder_by=["subreddit", "created_date"])
    logger.info("Silver events written: %s rows", events.count())


if __name__ == "__main__":
    main()
