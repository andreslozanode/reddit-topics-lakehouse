"""Silver layer: unify posts + comments into one cleaned ``events`` table.

Design decision (mirrors the unified-events approach): rather than maintaining
separate post/comment tables, both are normalized to a single canonical schema
and combined. This keeps every downstream aggregate, topic model and feature
pipeline working off one well-defined grain (one row per Reddit event).

All functions here are pure ``DataFrame -> DataFrame`` transforms with no IO,
so they are fully unit-testable on a local Spark session.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from reddit_lakehouse.common.schemas import normalize_raw

# Reddit placeholders for content removed by users/mods.
_REMOVED_TOKENS = ["[removed]", "[deleted]", "", "n/a"]


def clean_text_expr(col: F.Column) -> F.Column:
    """Normalize whitespace and strip control characters from a text column."""
    cleaned = F.regexp_replace(col, r"\s+", " ")
    cleaned = F.regexp_replace(cleaned, r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "")
    return F.trim(cleaned)


def _is_valid_content(title: F.Column, text: F.Column) -> F.Column:
    """A row is valid when it has at least one non-removed text field."""
    title_ok = title.isNotNull() & ~F.lower(F.trim(title)).isin(_REMOVED_TOKENS)
    text_ok = text.isNotNull() & ~F.lower(F.trim(text)).isin(_REMOVED_TOKENS)
    return title_ok | text_ok


def to_events(
    posts: DataFrame | None,
    comments: DataFrame | None,
    *,
    topic: str,
) -> DataFrame:
    """Normalize, union, clean and enrich posts/comments into canonical events."""
    frames: list[DataFrame] = []
    if posts is not None:
        frames.append(normalize_raw(posts, event_type="post", topic=topic))
    if comments is not None:
        frames.append(normalize_raw(comments, event_type="comment", topic=topic))
    if not frames:
        raise ValueError("to_events requires at least one of posts/comments")

    unified = frames[0]
    for frame in frames[1:]:
        unified = unified.unionByName(frame, allowMissingColumns=True)

    title = clean_text_expr(F.col("title"))
    body = clean_text_expr(F.col("body_text"))
    # Combined text used for NLP: post title + selftext, or comment body.
    combined = F.trim(F.concat_ws(" ", F.coalesce(title, F.lit("")), F.coalesce(body, F.lit(""))))

    enriched = (
        unified.withColumn("title", title)
        .withColumn("text", combined)
        .withColumn("created_ts", F.to_timestamp(F.from_unixtime(F.col("created_utc"))))
        .filter(_is_valid_content(F.col("title"), F.col("text")))
        .filter(F.col("score").isNotNull())
    )

    enriched = (
        enriched.withColumn("created_date", F.to_date("created_ts").cast("string"))
        .withColumn("created_hour", F.hour("created_ts").cast("long"))
        .withColumn("created_dow", F.dayofweek("created_ts").cast("long"))
        .withColumn("text_length", F.length("text").cast("long"))
        .withColumn(
            "word_count",
            F.when(F.col("text") == "", F.lit(0))
            .otherwise(F.size(F.split(F.col("text"), r"\s+")))
            .cast("long"),
        )
    )

    # Deduplicate on the stable event id (idempotent re-ingestion safe).
    deduped = enriched.dropDuplicates(["event_id"])

    return deduped.select(
        "event_id",
        "event_type",
        "topic",
        "subreddit",
        "subreddit_nsfw",
        "created_ts",
        "created_date",
        "created_hour",
        "created_dow",
        "title",
        "text",
        "text_length",
        "word_count",
        "score",
        "permalink",
        "domain",
        "url",
        "sentiment_source",
    )
