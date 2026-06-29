"""Unsupervised topic modeling with Spark ML LDA over TF-IDF/BoW features.

Produces:
  * a fitted ``PipelineModel`` (tokenizer -> stopwords -> CountVectorizer -> LDA)
  * a topics terms table (top words per topic) for dashboard labeling
  * per-event ``dominant_topic`` + ``topic_confidence`` for gold trend tables

Logged to MLflow and (optionally) registered in the Unity Catalog model registry.
"""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, RegexTokenizer, StopWordsRemover
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from reddit_lakehouse.common.logging import get_logger

logger = get_logger(__name__)


def build_topic_pipeline(
    *,
    k: int,
    vocab_size: int,
    min_token_length: int,
    min_doc_freq: int,
    max_iter: int,
    seed: int = 42,
) -> Pipeline:
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern=r"[^a-zA-Z]+",
        minTokenLength=min_token_length,
        toLowercase=True,
    )
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="bow",
        vocabSize=vocab_size,
        minDF=float(min_doc_freq),
    )
    lda = LDA(
        k=k,
        maxIter=max_iter,
        featuresCol="bow",
        topicDistributionCol="topic_distribution",
        optimizer="online",
        seed=seed,
    )
    return Pipeline(stages=[tokenizer, remover, cv, lda])


def topic_terms(model: PipelineModel, *, words_per_topic: int = 10) -> list[dict]:
    """Extract the top terms per topic for human-readable labels."""
    cv_model = model.stages[2]
    lda_model = model.stages[3]
    vocab = cv_model.vocabulary  # type: ignore[attr-defined]
    described = lda_model.describeTopics(words_per_topic).collect()  # type: ignore[attr-defined]
    out: list[dict] = []
    for row in described:
        terms = [vocab[i] for i in row["termIndices"]]
        out.append(
            {
                "topic_id": int(row["topic"]),
                "terms": terms,
                "term_weights": [float(w) for w in row["termWeights"]],
                "label": ", ".join(terms[:4]),
            }
        )
    return out


def _argmax_topic_udf():
    from pyspark.sql.functions import udf

    @udf(returnType=IntegerType())
    def _argmax(vec):
        if vec is None:
            return None
        arr = vec.toArray()
        return int(arr.argmax())

    return _argmax


def assign_topics(model: PipelineModel, events: DataFrame) -> DataFrame:
    """Score events and attach ``dominant_topic`` + ``topic_confidence``."""
    scored = model.transform(events)
    argmax = _argmax_topic_udf()

    from pyspark.ml.functions import vector_to_array

    scored = scored.withColumn("topic_array", vector_to_array("topic_distribution"))  # type: ignore[arg-type]
    return (
        scored.withColumn("dominant_topic", argmax(F.col("topic_distribution")))
        .withColumn("topic_confidence", F.array_max("topic_array"))
        .select(
            "event_id",
            "topic",
            "subreddit",
            "created_date",
            "score",
            "dominant_topic",
            "topic_confidence",
        )
    )


def train_topic_model(
    events: DataFrame,
    *,
    k: int,
    vocab_size: int,
    min_token_length: int,
    min_doc_freq: int,
    max_iter: int,
    seed: int = 42,
) -> tuple[PipelineModel, list[dict]]:
    """Fit the LDA pipeline and return (model, topic_terms)."""
    pipeline = build_topic_pipeline(
        k=k,
        vocab_size=vocab_size,
        min_token_length=min_token_length,
        min_doc_freq=min_doc_freq,
        max_iter=max_iter,
        seed=seed,
    )
    logger.info("Training LDA with k=%s on %s events", k, "<deferred>")
    model = pipeline.fit(events)
    terms = topic_terms(model)
    return model, terms
