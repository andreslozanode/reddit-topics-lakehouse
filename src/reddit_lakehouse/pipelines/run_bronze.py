"""Bronze job: download public datasets and persist raw Delta tables."""

from __future__ import annotations

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.common.spark import get_spark
from reddit_lakehouse.common.tables import write_delta
from reddit_lakehouse.ingestion import bronze, download
from reddit_lakehouse.pipelines._common import batch_id, resolve_config
from reddit_lakehouse.quality import min_row_count, not_null, run_checks

logger = get_logger(__name__)


def main() -> None:
    cfg = resolve_config()
    spark = get_spark("bronze")
    bid = batch_id()

    # Land parquet from Hugging Face into the raw landing zone (volume).
    local_dirs = download.download_all(cfg.sources, cfg.raw_landing_path)

    for source in cfg.sources:
        base_dir = local_dirs[source.topic]
        frames = bronze.build_bronze_frames(
            spark,
            base_dir=base_dir,
            topic=source.topic,
            batch_id=bid,
            posts_glob=source.posts_glob,
            comments_glob=source.comments_glob,
        )
        for kind, df in frames.items():
            report = run_checks(df, [min_row_count(cfg.dq.min_bronze_rows), not_null(["id"])])
            if report.critical_failures:
                raise RuntimeError(f"Bronze DQ failed for {source.topic}/{kind}: {report.summary()}")
            write_delta(
                df,
                cfg.bronze(f"{kind}_raw"),
                mode="append",
                partition_by=["_topic", "_ingest_date"],
                comment=f"Raw Reddit {kind} (SocialGrep schema). Append-only.",
            )
            logger.info("Bronze %s.%s written for topic=%s", kind, "raw", source.topic)


if __name__ == "__main__":
    main()
