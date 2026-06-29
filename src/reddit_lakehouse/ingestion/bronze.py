"""Bronze layer: raw parquet -> Delta with audit/lineage metadata.

Bronze is append-only and schema-tolerant: it preserves the source as-is and
only adds ingestion metadata (batch id, ingest timestamp, source topic). All
cleaning happens downstream in silver.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _read_glob(spark: SparkSession, base_dir: str, glob: str) -> DataFrame | None:
    path = f"{base_dir}/{glob}"
    try:
        df = spark.read.parquet(path)
        return df if len(df.columns) else None
    except Exception:
        return None


def add_ingestion_metadata(df: DataFrame, *, topic: str, batch_id: str) -> DataFrame:
    """Attach lineage columns used for auditing and incremental processing."""
    return (
        df.withColumn("_topic", F.lit(topic))
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_ingest_date", F.current_date())
    )


def build_bronze_frames(
    spark: SparkSession,
    *,
    base_dir: str,
    topic: str,
    batch_id: str,
    posts_glob: str = "*posts*.parquet",
    comments_glob: str = "*comments*.parquet",
) -> dict[str, DataFrame]:
    """Read landed parquet for one topic and return bronze posts/comments frames.

    Missing kinds are simply omitted (some themed datasets ship posts only).
    """
    out: dict[str, DataFrame] = {}
    posts = _read_glob(spark, base_dir, posts_glob)
    comments = _read_glob(spark, base_dir, comments_glob)
    if posts is not None:
        out["posts"] = add_ingestion_metadata(posts, topic=topic, batch_id=batch_id)
    if comments is not None:
        out["comments"] = add_ingestion_metadata(comments, topic=topic, batch_id=batch_id)
    return out
