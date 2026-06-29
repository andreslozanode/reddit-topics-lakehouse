"""Batch inference: load registered models and write gold prediction tables.

Models are resolved from the Unity Catalog model registry via MLflow. Falls back
to in-memory models when running an end-to-end pipeline in a single session.
"""

from __future__ import annotations

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.ml import engagement, sentiment, topic_modeling

logger = get_logger(__name__)


def load_spark_model(model_uri: str) -> PipelineModel:
    """Load a Spark ML PipelineModel logged with mlflow.spark."""
    import mlflow.spark

    logger.info("Loading model %s", model_uri)
    return mlflow.spark.load_model(model_uri)


def run_all_inference(
    events: DataFrame,
    *,
    topic_model: PipelineModel,
    sentiment_model: PipelineModel,
    engagement_model: PipelineModel,
) -> dict[str, DataFrame]:
    """Produce the three gold prediction frames from in-memory models."""
    return {
        "post_topics": topic_modeling.assign_topics(topic_model, events),
        "post_sentiment": sentiment.score_sentiment(sentiment_model, events),
        "engagement_predictions": engagement.predict_engagement(engagement_model, events),
    }
