"""Reusable Spark ML feature-engineering building blocks.

Two families of features:
  * Text: Tokenizer -> StopWordsRemover -> CountVectorizer -> IDF (TF-IDF).
  * Structured: length/word-count/time/url/question flags assembled to a vector.

All stages are Photon-friendly (no Python UDFs in the hot path) and reused by
the topic, sentiment and engagement models so behavior stays consistent.
"""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    IDF,
    CountVectorizer,
    RegexTokenizer,
    StandardScaler,
    StopWordsRemover,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

STRUCTURED_NUMERIC = ["text_length", "word_count", "created_hour", "created_dow"]
STRUCTURED_FLAGS = ["has_url", "is_question", "is_post"]


def add_derived_features(df: DataFrame) -> DataFrame:
    """Add cheap, deterministic structured features used by several models."""
    return (
        df.withColumn("has_url", F.when(F.col("url").isNotNull() & (F.col("url") != ""), 1.0).otherwise(0.0))
        .withColumn("is_question", F.when(F.col("text").contains("?"), 1.0).otherwise(0.0))
        .withColumn("is_post", F.when(F.col("event_type") == "post", 1.0).otherwise(0.0))
    )


def text_feature_stages(
    *,
    input_col: str = "text",
    vocab_size: int = 20_000,
    min_token_length: int = 3,
    min_doc_freq: int = 5,
    output_col: str = "tf_idf",
) -> list:
    """Return [tokenizer, stopwords, count-vectorizer, idf] stages."""
    tokenizer = RegexTokenizer(
        inputCol=input_col,
        outputCol="tokens",
        pattern=r"[^a-zA-Z]+",
        minTokenLength=min_token_length,
        toLowercase=True,
    )
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=vocab_size,
        minDF=float(min_doc_freq),
    )
    idf = IDF(inputCol="raw_features", outputCol=output_col)
    return [tokenizer, remover, cv, idf]


def structured_feature_stages(output_col: str = "structured_features") -> list:
    """Assemble + scale structured numeric/flag features."""
    assembler = VectorAssembler(
        inputCols=STRUCTURED_NUMERIC + STRUCTURED_FLAGS,
        outputCol="structured_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="structured_raw",
        outputCol=output_col,
        withMean=True,
        withStd=True,
    )
    return [assembler, scaler]


def supervised_feature_pipeline(
    *,
    vocab_size: int = 20_000,
    min_doc_freq: int = 5,
    label_index_input: str | None = None,
) -> Pipeline:
    """Full pipeline combining TF-IDF + structured features into ``features``.

    If ``label_index_input`` is provided, a StringIndexer is prepended to turn a
    string label column into the numeric ``label`` column models expect.
    """
    stages: list = []
    if label_index_input:
        stages.append(StringIndexer(inputCol=label_index_input, outputCol="label", handleInvalid="skip"))
    stages += text_feature_stages(vocab_size=vocab_size, min_doc_freq=min_doc_freq)
    stages += structured_feature_stages()
    stages.append(
        VectorAssembler(
            inputCols=["tf_idf", "structured_features"],
            outputCol="features",
            handleInvalid="keep",
        )
    )
    return Pipeline(stages=stages)
