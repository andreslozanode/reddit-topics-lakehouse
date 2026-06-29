"""Canonical schemas + raw->canonical normalization.

The SocialGrep public datasets expose posts and comments in separate parquet
files that share most columns. Column names use dotted notation (e.g.
``subreddit.name``) which Spark treats specially, so normalization renames them
to flat, snake_case fields and unifies posts/comments into one event schema.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---- Raw schemas (as published by SocialGrep on Hugging Face) ------------- #
RAW_POST_SCHEMA = StructType(
    [
        StructField("type", StringType(), True),
        StructField("id", StringType(), True),
        StructField("subreddit.id", StringType(), True),
        StructField("subreddit.name", StringType(), True),
        StructField("subreddit.nsfw", BooleanType(), True),
        StructField("created_utc", LongType(), True),
        StructField("permalink", StringType(), True),
        StructField("domain", StringType(), True),
        StructField("url", StringType(), True),
        StructField("selftext", StringType(), True),
        StructField("title", StringType(), True),
        StructField("score", LongType(), True),
    ]
)

RAW_COMMENT_SCHEMA = StructType(
    [
        StructField("type", StringType(), True),
        StructField("id", StringType(), True),
        StructField("subreddit.id", StringType(), True),
        StructField("subreddit.name", StringType(), True),
        StructField("subreddit.nsfw", BooleanType(), True),
        StructField("created_utc", LongType(), True),
        StructField("permalink", StringType(), True),
        StructField("body", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("score", LongType(), True),
    ]
)

# ---- Canonical silver event schema ---------------------------------------- #
EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("event_type", StringType(), False),  # post | comment
        StructField("topic", StringType(), True),
        StructField("subreddit", StringType(), True),
        StructField("subreddit_nsfw", BooleanType(), True),
        StructField("created_ts", TimestampType(), True),
        StructField("created_date", StringType(), True),
        StructField("created_hour", LongType(), True),
        StructField("created_dow", LongType(), True),
        StructField("title", StringType(), True),
        StructField("text", StringType(), True),
        StructField("text_length", LongType(), True),
        StructField("word_count", LongType(), True),
        StructField("score", LongType(), True),
        StructField("permalink", StringType(), True),
        StructField("domain", StringType(), True),
        StructField("url", StringType(), True),
        StructField("sentiment_source", StringType(), True),
    ]
)


def _flatten_subreddit(df: DataFrame) -> DataFrame:
    """Rename dotted ``subreddit.*`` columns to flat fields.

    Handles both the dotted-string column form and a genuine nested struct.
    """
    cols = set(df.columns)
    if "subreddit.name" in cols:
        df = df.withColumnRenamed("subreddit.name", "subreddit_name")
    if "subreddit.nsfw" in cols:
        df = df.withColumnRenamed("subreddit.nsfw", "subreddit_nsfw")
    if "subreddit.id" in cols:
        df = df.withColumnRenamed("subreddit.id", "subreddit_id")
    if "subreddit" in df.columns and dict(df.dtypes)["subreddit"].startswith("struct"):
        df = (
            df.withColumn("subreddit_name", F.col("subreddit.name"))
            .withColumn("subreddit_nsfw", F.col("subreddit.nsfw"))
            .drop("subreddit")
        )
    return df


def normalize_raw(df: DataFrame, *, event_type: str, topic: str) -> DataFrame:
    """Map a raw posts/comments DataFrame to the canonical event schema.

    Posts contribute ``title`` + ``selftext``; comments contribute ``body``.
    A stable ``event_id`` is derived from the Reddit base-36 id and the type.
    """
    df = _flatten_subreddit(df)

    has = set(df.columns)
    title_col = F.col("title") if "title" in has else F.lit(None).cast("string")
    body_col = F.coalesce(
        F.col("selftext") if "selftext" in has else F.lit(None).cast("string"),
        F.col("body") if "body" in has else F.lit(None).cast("string"),
    )
    sentiment_col = F.col("sentiment") if "sentiment" in has else F.lit(None).cast("string")

    return df.select(
        F.concat_ws("_", F.lit(event_type), F.col("id")).alias("event_id"),
        F.lit(event_type).alias("event_type"),
        F.lit(topic).alias("topic"),
        F.col("subreddit_name").alias("subreddit"),
        F.col("subreddit_nsfw").cast(BooleanType()).alias("subreddit_nsfw"),
        F.col("created_utc").cast(LongType()).alias("created_utc"),
        title_col.alias("title"),
        body_col.alias("body_text"),
        F.col("score").cast(LongType()).alias("score"),
        F.col("permalink") if "permalink" in has else F.lit(None).cast("string").alias("permalink"),
        (F.col("domain") if "domain" in has else F.lit(None).cast("string")).alias("domain"),
        (F.col("url") if "url" in has else F.lit(None).cast("string")).alias("url"),
        sentiment_col.alias("sentiment_source"),
    )
