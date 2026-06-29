"""Provision Unity Catalog objects (catalog / schemas / volume / grants)."""

from __future__ import annotations

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.common.spark import get_spark
from reddit_lakehouse.governance import unity_catalog
from reddit_lakehouse.pipelines._common import resolve_config

logger = get_logger(__name__)


def main() -> None:
    cfg = resolve_config()
    spark = get_spark("setup")
    unity_catalog.provision(spark, cfg)
    logger.info("Unity Catalog provisioned for env=%s", cfg.env)


if __name__ == "__main__":
    main()
