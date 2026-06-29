"""Batch inference job: score the latest silver events with registered models."""

from __future__ import annotations

from reddit_lakehouse.common.io import read_table
from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.common.spark import get_spark
from reddit_lakehouse.common.tables import write_delta
from reddit_lakehouse.ml.batch_inference import load_spark_model, run_all_inference
from reddit_lakehouse.pipelines._common import resolve_config

logger = get_logger(__name__)


def main() -> None:
    cfg = resolve_config()
    spark = get_spark("batch-inference")
    events = read_table(spark, cfg.silver("events"))

    topic_model = load_spark_model(f"models:/{cfg.model_name(cfg.ml.registered_topic_model)}@champion")
    sentiment_model = load_spark_model(
        f"models:/{cfg.model_name(cfg.ml.registered_sentiment_model)}@champion"
    )
    engagement_model = load_spark_model(
        f"models:/{cfg.model_name(cfg.ml.registered_engagement_model)}@champion"
    )

    preds = run_all_inference(
        events,
        topic_model=topic_model,
        sentiment_model=sentiment_model,
        engagement_model=engagement_model,
    )
    for name, df in preds.items():
        write_delta(df, cfg.gold(name), mode="overwrite", comment=f"Gold ML output: {name}")
        logger.info("Refreshed gold ML table %s", name)


if __name__ == "__main__":
    main()
