"""Thin read helpers shared by ingestion and transform stages."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    """Read parquet from a path or glob (a directory, file or wildcard)."""
    return spark.read.parquet(path)


def read_table(spark: SparkSession, fqn: str) -> DataFrame:
    """Read a managed/external table by fully-qualified name."""
    return spark.read.table(fqn)


def table_exists(spark: SparkSession, fqn: str) -> bool:
    try:
        return spark.catalog.tableExists(fqn)
    except Exception:
        return False
