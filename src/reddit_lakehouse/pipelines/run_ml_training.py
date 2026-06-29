"""ML training job: topic model, sentiment classifier, engagement classifier.

Each model is logged to MLflow and registered in the Unity Catalog model
registry. Topic terms and gold prediction tables are written so dashboards can
consume model output immediately.
"""

from __future__ import annotations

from reddit_lakehouse.common.io import read_table
from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.common.spark import get_spark
from reddit_lakehouse.common.tables import write_delta
from reddit_lakehouse.ml import engagement, sentiment, topic_modeling
from reddit_lakehouse.ml.batch_inference import run_all_inference
from reddit_lakehouse.pipelines._common import resolve_config

logger = get_logger(__name__)


def _mlflow():
    import mlflow
    import mlflow.spark  # noqa: F401  (registers the mlflow.spark submodule)

    mlflow.set_registry_uri("databricks-uc")
    return mlflow


def main() -> None:
    cfg = resolve_config()
    spark = get_spark("ml-training")
    events = read_table(spark, cfg.silver("events")).cache()
    events.count()

    mlflow = _mlflow()

    # ---- Topic modeling (LDA) -------------------------------------------- #
    with mlflow.start_run(run_name="topic_lda"):
        topic_model, terms = topic_modeling.train_topic_model(
            events,
            k=cfg.ml.topic_count,
            vocab_size=cfg.ml.vocab_size,
            min_token_length=cfg.ml.min_token_length,
            min_doc_freq=cfg.ml.min_doc_freq,
            max_iter=cfg.ml.topic_max_iterations,
            seed=cfg.ml.random_seed,
        )
        mlflow.log_params({"k": cfg.ml.topic_count, "vocab_size": cfg.ml.vocab_size})
        mlflow.spark.log_model(
            topic_model,
            artifact_path="model",
            registered_model_name=cfg.model_name(cfg.ml.registered_topic_model),
        )
        terms_df = spark.createDataFrame(terms)
        write_delta(terms_df, cfg.gold("topic_terms"), mode="overwrite", comment="LDA topic terms")

    # ---- Sentiment classifier -------------------------------------------- #
    with mlflow.start_run(run_name="sentiment_clf"):
        sentiment_model, s_metrics = sentiment.train_sentiment(
            events, vocab_size=cfg.ml.vocab_size, min_doc_freq=cfg.ml.min_doc_freq, seed=cfg.ml.random_seed
        )
        mlflow.log_metrics(s_metrics)
        mlflow.spark.log_model(
            sentiment_model,
            artifact_path="model",
            registered_model_name=cfg.model_name(cfg.ml.registered_sentiment_model),
        )

    # ---- Engagement classifier ------------------------------------------- #
    with mlflow.start_run(run_name="engagement_clf"):
        engagement_model, e_metrics = engagement.train_engagement(
            events,
            vocab_size=cfg.ml.vocab_size,
            min_doc_freq=cfg.ml.min_doc_freq,
            percentile=cfg.ml.engagement_percentile,
            cv_folds=cfg.ml.cv_folds,
            seed=cfg.ml.random_seed,
        )
        mlflow.log_metrics(e_metrics)
        mlflow.spark.log_model(
            engagement_model,
            artifact_path="model",
            registered_model_name=cfg.model_name(cfg.ml.registered_engagement_model),
        )

    # ---- Write gold prediction tables ------------------------------------ #
    preds = run_all_inference(
        events,
        topic_model=topic_model,
        sentiment_model=sentiment_model,
        engagement_model=engagement_model,
    )
    for name, df in preds.items():
        write_delta(df, cfg.gold(name), mode="overwrite", comment=f"Gold ML output: {name}")
        logger.info("Gold ML table %s written", name)

    events.unpersist()
    logger.info("ML training complete")


if __name__ == "__main__":
    main()
