"""Gold layer: curated, BI-ready aggregates and serving tables.

Every function returns a tidy, denormalized DataFrame designed to back a
dashboard tile or a star-schema fact/dim. Grains are explicit in the names.
All transforms are pure and unit-testable.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def subreddit_daily(events: DataFrame) -> DataFrame:
    """Grain: one row per (topic, subreddit, created_date). Core trend fact."""
    return (
        events.groupBy("topic", "subreddit", "created_date")
        .agg(
            F.count("*").alias("event_count"),
            F.sum(F.when(F.col("event_type") == "post", 1).otherwise(0)).alias("post_count"),
            F.sum(F.when(F.col("event_type") == "comment", 1).otherwise(0)).alias("comment_count"),
            F.avg("score").alias("avg_score"),
            F.expr("percentile_approx(score, 0.5)").alias("median_score"),
            F.max("score").alias("max_score"),
            F.sum("score").alias("total_score"),
            F.avg("text_length").alias("avg_text_length"),
            F.avg("word_count").alias("avg_word_count"),
        )
        .withColumn("comments_per_post", F.col("comment_count") / F.greatest(F.col("post_count"), F.lit(1)))
    )


def subreddit_summary(events: DataFrame) -> DataFrame:
    """Grain: one row per (topic, subreddit). Leaderboard / overview dim+facts."""
    return (
        events.groupBy("topic", "subreddit")
        .agg(
            F.count("*").alias("event_count"),
            F.countDistinct("created_date").alias("active_days"),
            F.avg("score").alias("avg_score"),
            F.expr("percentile_approx(score, 0.9)").alias("p90_score"),
            F.max("score").alias("max_score"),
            F.min("created_ts").alias("first_seen"),
            F.max("created_ts").alias("last_seen"),
            F.first("subreddit_nsfw", ignorenulls=True).alias("nsfw"),
        )
        .withColumn(
            "events_per_active_day", F.col("event_count") / F.greatest(F.col("active_days"), F.lit(1))
        )
    )


def hourly_activity(events: DataFrame) -> DataFrame:
    """Grain: one row per (topic, created_dow, created_hour). Heatmap source."""
    return (
        events.groupBy("topic", "created_dow", "created_hour")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("score").alias("avg_score"),
        )
        .orderBy("topic", "created_dow", "created_hour")
    )


def topic_trends(events_with_topics: DataFrame) -> DataFrame:
    """Grain: one row per (topic, dominant_topic, created_date).

    Requires the LDA output column ``dominant_topic`` joined onto events.
    Backs the "what themes are rising over time" dashboard.
    """
    return (
        events_with_topics.groupBy("topic", "dominant_topic", "created_date")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("score").alias("avg_score"),
            F.avg("topic_confidence").alias("avg_confidence"),
        )
        .orderBy("topic", "created_date", "dominant_topic")
    )


def sentiment_daily(events_with_sentiment: DataFrame) -> DataFrame:
    """Grain: one row per (topic, subreddit, created_date) with sentiment mix.

    Requires a ``sentiment_label`` column in {negative, neutral, positive}.
    """
    return (
        events_with_sentiment.groupBy("topic", "subreddit", "created_date")
        .agg(
            F.count("*").alias("event_count"),
            F.sum(F.when(F.col("sentiment_label") == "positive", 1).otherwise(0)).alias("positive"),
            F.sum(F.when(F.col("sentiment_label") == "neutral", 1).otherwise(0)).alias("neutral"),
            F.sum(F.when(F.col("sentiment_label") == "negative", 1).otherwise(0)).alias("negative"),
        )
        .withColumn(
            "net_sentiment",
            (F.col("positive") - F.col("negative")) / F.greatest(F.col("event_count"), F.lit(1)),
        )
    )
