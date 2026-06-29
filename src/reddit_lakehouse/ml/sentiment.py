"""Supervised sentiment classification (negative / neutral / positive).

Weak labels are derived from the dataset's in-house ``sentiment_source`` score
when available, otherwise from the engagement ``score`` distribution. A
Logistic Regression over TF-IDF + structured features is then trained.

Design note: the label ``StringIndexer`` is fit *outside* the served pipeline so
the model needs no label column at inference. An ``IndexToString`` stage maps
the numeric prediction back to a human-readable label, making the model fully
self-contained for batch inference. This is a classical, reproducible ML
pipeline (no external LLM / chatbot).
"""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.ml.features import add_derived_features, supervised_feature_pipeline

logger = get_logger(__name__)

LABELS = ["negative", "neutral", "positive"]
PRED_COL = "predicted_sentiment"


def derive_labels(df: DataFrame) -> DataFrame:
    """Attach a 3-class ``sentiment_label`` from source sentiment or score sign."""
    src = F.col("sentiment_source").cast("double")
    has_source = src.isNotNull()

    label = (
        F.when(has_source & (src > 0.05), F.lit("positive"))
        .when(has_source & (src < -0.05), F.lit("negative"))
        .when(has_source, F.lit("neutral"))
        # Fallback: use score sign as a weak proxy.
        .when(F.col("score") > 5, F.lit("positive"))
        .when(F.col("score") < 0, F.lit("negative"))
        .otherwise(F.lit("neutral"))
    )
    return df.withColumn("sentiment_label", label)


def train_sentiment(
    events: DataFrame,
    *,
    vocab_size: int,
    min_doc_freq: int,
    seed: int = 42,
) -> tuple[PipelineModel, dict]:
    """Train the sentiment classifier; return (model, metrics).

    The returned model transforms raw (derived-feature) events directly into a
    ``predicted_sentiment`` string column — no label input required.
    """
    labeled = add_derived_features(derive_labels(events))

    # Fit the label indexer separately so it never enters the served pipeline.
    label_indexer = StringIndexer(inputCol="sentiment_label", outputCol="label", handleInvalid="skip").fit(
        labeled
    )
    indexed = label_indexer.transform(labeled)

    train_df, test_df = indexed.randomSplit([0.8, 0.2], seed=seed)

    feature_pipeline = supervised_feature_pipeline(
        vocab_size=vocab_size, min_doc_freq=min_doc_freq, label_index_input=None
    )
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0,
        family="multinomial",
    )
    inverse = IndexToString(inputCol="prediction", outputCol=PRED_COL, labels=label_indexer.labels)

    pipeline = Pipeline(stages=feature_pipeline.getStages() + [lr, inverse])

    logger.info("Training sentiment classifier")
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    metrics = {}
    for name in ("f1", "accuracy"):
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=name
        )
        metrics[name] = float(evaluator.evaluate(predictions))
    logger.info("Sentiment metrics: %s", metrics)
    return model, metrics


def score_sentiment(model: PipelineModel, events: DataFrame) -> DataFrame:
    """Apply the model; emit a string ``sentiment_label`` per event."""
    featured = add_derived_features(events)
    scored = model.transform(featured)
    return scored.select(
        "event_id",
        "topic",
        "subreddit",
        "created_date",
        F.col(PRED_COL).alias("sentiment_label"),
    )
