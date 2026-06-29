"""Idempotent Unity Catalog provisioning + governance.

Creates the catalog, the bronze/silver/gold schemas and a raw landing volume,
applies grants, and attaches descriptive comments/tags. Safe to re-run.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.config import PipelineConfig

logger = get_logger(__name__)

TABLE_COMMENTS = {
    "silver.events": "Cleaned, unified Reddit events (posts + comments). One row per event.",
    "gold.subreddit_daily": "Daily activity & engagement per subreddit. Trend fact.",
    "gold.subreddit_summary": "Lifetime overview per subreddit. Leaderboard / dim.",
    "gold.hourly_activity": "Activity heatmap by day-of-week and hour.",
    "gold.topic_terms": "Top terms per LDA topic for dashboard labeling.",
    "gold.topic_trends": "Dominant topic distribution over time.",
    "gold.sentiment_daily": "Daily sentiment mix per subreddit.",
    "gold.engagement_predictions": "Per-event high-engagement probability.",
}


def provision(spark: SparkSession, cfg: PipelineConfig, *, reader_principal: str | None = None) -> None:
    """Create catalog, schemas and volume; grant least-privilege access."""
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
    for schema in (cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{schema}")
        logger.info("Ensured schema %s.%s", cfg.catalog, schema)

    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.bronze_schema}.{cfg.volume}")

    if reader_principal:
        # BI / dashboard service principal: read-only on gold.
        spark.sql(f"GRANT USE CATALOG ON CATALOG {cfg.catalog} TO `{reader_principal}`")
        spark.sql(f"GRANT USE SCHEMA ON SCHEMA {cfg.catalog}.{cfg.gold_schema} TO `{reader_principal}`")
        spark.sql(f"GRANT SELECT ON SCHEMA {cfg.catalog}.{cfg.gold_schema} TO `{reader_principal}`")
        logger.info("Granted read-only gold access to %s", reader_principal)


def apply_table_metadata(spark: SparkSession, cfg: PipelineConfig) -> None:
    """Attach descriptive comments to known tables (idempotent)."""
    schema_map = {
        "bronze": cfg.bronze_schema,
        "silver": cfg.silver_schema,
        "gold": cfg.gold_schema,
    }
    for key, comment in TABLE_COMMENTS.items():
        logical_schema, table = key.split(".", 1)
        fqn = f"{cfg.catalog}.{schema_map[logical_schema]}.{table}"
        if spark.catalog.tableExists(fqn):
            safe = comment.replace("'", "")
            spark.sql(f"COMMENT ON TABLE {fqn} IS '{safe}'")
