"""SparkSession factory + performance tuning.

On Databricks the active session is reused. Locally (tests / dev) a session is
created with Delta Lake if ``delta-spark`` is importable, otherwise plain Spark
is returned so unit tests can run on DataFrames without a Delta install.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from reddit_lakehouse.common.logging import get_logger

logger = get_logger(__name__)


def _delta_available() -> bool:
    try:
        import delta  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def apply_performance_conf(spark: SparkSession) -> SparkSession:
    """Apply lakehouse-wide performance defaults.

    These complement (do not replace) cluster-level Photon + AQE settings and
    are safe to set from application code.
    """
    conf = {
        # Adaptive Query Execution: dynamic partition coalescing + skew join handling
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        # Reasonable default for medium datasets; AQE will coalesce further
        "spark.sql.shuffle.partitions": "200",
        # Broadcast joins for dimension-sized tables (<= 50 MB)
        "spark.sql.autoBroadcastJoinThreshold": str(50 * 1024 * 1024),
        # Delta: write-time optimization (compaction + data skipping)
        "spark.databricks.delta.optimizeWrite.enabled": "true",
        "spark.databricks.delta.autoCompact.enabled": "true",
        "spark.databricks.delta.merge.repartitionBeforeWrite.enabled": "true",
        # Arrow for fast Python <-> JVM transfer
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    }
    for key, value in conf.items():
        try:
            spark.conf.set(key, value)
        except Exception:  # pragma: no cover - some confs are static locally
            logger.debug("Could not set %s locally", key)
    return spark


def get_spark(app_name: str = "reddit-topics-lakehouse") -> SparkSession:
    """Return the active session, or build a local Delta-enabled one."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return apply_performance_conf(active)

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
    )

    if _delta_available():
        from delta import configure_spark_with_delta_pip

        builder = builder.config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        ).config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    else:
        logger.warning("delta-spark not available — using plain Spark session")
        spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return apply_performance_conf(spark)
