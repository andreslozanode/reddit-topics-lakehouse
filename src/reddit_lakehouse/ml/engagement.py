"""Engagement prediction: will an event be high-engagement?

Binary classification where the positive class is "score in the top decile"
(threshold configurable). Uses a Gradient-Boosted Trees classifier over TF-IDF +
structured features with k-fold cross-validated hyper-parameter tuning. Designed
for MLflow autologging and Unity Catalog model registration.
"""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.ml.features import add_derived_features, supervised_feature_pipeline

logger = get_logger(__name__)


def make_engagement_label(events: DataFrame, *, percentile: float = 0.90) -> DataFrame:
    """Label the top-`(1-percentile)` of scores (per topic) as high engagement."""
    from pyspark.sql import Window

    w = Window.partitionBy("topic")
    threshold = F.expr(f"percentile_approx(score, {percentile})").over(w)
    return events.withColumn("score_threshold", threshold).withColumn(
        "label", F.when(F.col("score") >= F.col("score_threshold"), 1.0).otherwise(0.0)
    )


def train_engagement(
    events: DataFrame,
    *,
    vocab_size: int,
    min_doc_freq: int,
    percentile: float = 0.90,
    cv_folds: int = 3,
    seed: int = 42,
) -> tuple[PipelineModel, dict]:
    """Train + cross-validate the engagement classifier; return (model, metrics)."""
    labeled = add_derived_features(make_engagement_label(events, percentile=percentile))
    train_df, test_df = labeled.randomSplit([0.8, 0.2], seed=seed)

    feature_pipeline = supervised_feature_pipeline(vocab_size=vocab_size, min_doc_freq=min_doc_freq)
    gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=seed)

    pipeline = Pipeline(stages=feature_pipeline.getStages() + [gbt])

    grid = ParamGridBuilder().addGrid(gbt.maxDepth, [4, 6]).addGrid(gbt.maxIter, [20, 40]).build()
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=cv_folds,
        parallelism=2,
        seed=seed,
    )

    logger.info("Cross-validating engagement classifier (%s folds)", cv_folds)
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel  # PipelineModel at runtime

    predictions = best_model.transform(test_df)
    auc = float(evaluator.evaluate(predictions))
    f1 = float(
        BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR").evaluate(predictions)
    )
    metrics = {"areaUnderROC": auc, "areaUnderPR": f1, "best_avg_metric": float(max(cv_model.avgMetrics))}
    logger.info("Engagement metrics: %s", metrics)
    return best_model, metrics  # type: ignore[return-value]


def predict_engagement(model: PipelineModel, events: DataFrame) -> DataFrame:
    """Score events with engagement probability + binary prediction."""
    from pyspark.ml.functions import vector_to_array

    featured = add_derived_features(events)
    scored = model.transform(featured)
    return (
        scored.withColumn("proba_array", vector_to_array("probability"))  # type: ignore[arg-type]
        .withColumn("engagement_probability", F.col("proba_array")[1])
        .withColumnRenamed("prediction", "is_high_engagement")
        .select(
            "event_id",
            "topic",
            "subreddit",
            "created_date",
            "score",
            "is_high_engagement",
            "engagement_probability",
        )
    )
