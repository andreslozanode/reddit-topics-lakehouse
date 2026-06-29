"""Delta write + maintenance helpers with production-grade optimizations.

Capabilities (applied when running on a Delta-enabled cluster):
  * idempotent upserts via ``MERGE`` on business keys
  * write-time optimization (optimizeWrite / autoCompact set in session conf)
  * ``OPTIMIZE ... ZORDER BY`` for data skipping on high-cardinality filters
  * liquid clustering via ``CLUSTER BY`` for tables where it is preferred
  * Delta table properties (deletion vectors, column mapping, log retention)
  * ``VACUUM`` for storage reclamation

Local runs without Delta fall back to parquet so unit tests stay lightweight.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pyspark.sql import DataFrame, SparkSession

from reddit_lakehouse.common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TABLE_PROPERTIES = {
    "delta.enableDeletionVectors": "true",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "delta.logRetentionDuration": "interval 30 days",
    "delta.deletedFileRetentionDuration": "interval 7 days",
}


def _is_delta(spark: SparkSession) -> bool:
    try:
        extensions = spark.conf.get("spark.sql.extensions", "") or ""
        return "Delta" in extensions
    except Exception:
        return False


def set_table_properties(spark: SparkSession, fqn: str, props: dict[str, str] | None = None) -> None:
    props = {**DEFAULT_TABLE_PROPERTIES, **(props or {})}
    pairs = ", ".join(f"'{k}' = '{v}'" for k, v in props.items())
    spark.sql(f"ALTER TABLE {fqn} SET TBLPROPERTIES ({pairs})")


def write_delta(
    df: DataFrame,
    fqn: str,
    *,
    mode: str = "overwrite",
    partition_by: Sequence[str] | None = None,
    cluster_by: Sequence[str] | None = None,
    merge_keys: Sequence[str] | None = None,
    comment: str | None = None,
) -> None:
    """Write ``df`` to ``fqn``.

    * ``merge_keys`` set -> idempotent MERGE upsert (creates the table on first run).
    * ``cluster_by`` -> liquid clustering (preferred over Z-ORDER on new tables).
    * otherwise a partitioned overwrite/append.
    """
    spark = df.sparkSession

    if not _is_delta(spark):
        # Local fallback: parquet via saveAsTable so downstream reads work.
        writer = df.write.mode("overwrite" if mode == "merge" else mode).format("parquet")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(fqn)
        return

    if merge_keys:
        _merge_upsert(df, fqn, merge_keys, partition_by, cluster_by)
    else:
        writer = df.write.mode(mode).format("delta").option("mergeSchema", "true")
        if cluster_by:
            writer = writer.clusterBy(*cluster_by)  # type: ignore[attr-defined]
        elif partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(fqn)

    set_table_properties(spark, fqn)
    if comment:
        spark.sql(f"COMMENT ON TABLE {fqn} IS '{comment.replace(chr(39), '')}'")


def _merge_upsert(
    df: DataFrame,
    fqn: str,
    merge_keys: Sequence[str],
    partition_by: Sequence[str] | None,
    cluster_by: Sequence[str] | None,
) -> None:
    spark = df.sparkSession
    if not spark.catalog.tableExists(fqn):
        writer = df.write.mode("overwrite").format("delta")
        if cluster_by:
            writer = writer.clusterBy(*cluster_by)  # type: ignore[attr-defined]
        elif partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(fqn)
        return

    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, fqn)
    cond = " AND ".join(f"t.{k} = s.{k}" for k in merge_keys)
    (target.alias("t").merge(df.alias("s"), cond).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())


def optimize_table(
    spark: SparkSession,
    fqn: str,
    *,
    zorder_by: Iterable[str] | None = None,
) -> None:
    """Run ``OPTIMIZE`` (+ optional ``ZORDER BY``) for compaction + data skipping."""
    if not _is_delta(spark):
        logger.info("Skipping OPTIMIZE for %s (non-Delta session)", fqn)
        return
    stmt = f"OPTIMIZE {fqn}"
    if zorder_by:
        stmt += f" ZORDER BY ({', '.join(zorder_by)})"
    spark.sql(stmt)
    logger.info("Optimized %s", fqn)


def vacuum_table(spark: SparkSession, fqn: str, retention_hours: int = 168) -> None:
    if not _is_delta(spark):
        return
    spark.sql(f"VACUUM {fqn} RETAIN {retention_hours} HOURS")
    logger.info("Vacuumed %s (retain %sh)", fqn, retention_hours)
