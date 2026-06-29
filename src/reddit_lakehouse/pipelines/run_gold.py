"""Gold job: build BI-ready aggregate tables from silver events."""

from __future__ import annotations

from reddit_lakehouse.common.io import read_table
from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.common.spark import get_spark
from reddit_lakehouse.common.tables import optimize_table, write_delta
from reddit_lakehouse.pipelines._common import resolve_config
from reddit_lakehouse.transform import gold

logger = get_logger(__name__)


def main() -> None:
    cfg = resolve_config()
    spark = get_spark("gold")
    events = read_table(spark, cfg.silver("events"))

    tables = {
        "subreddit_daily": gold.subreddit_daily(events),
        "subreddit_summary": gold.subreddit_summary(events),
        "hourly_activity": gold.hourly_activity(events),
    }
    for name, df in tables.items():
        write_delta(df, cfg.gold(name), mode="overwrite", comment=f"Gold aggregate: {name}")
        logger.info("Gold %s written", name)

    optimize_table(spark, cfg.gold("subreddit_daily"), zorder_by=["subreddit", "created_date"])


if __name__ == "__main__":
    main()
