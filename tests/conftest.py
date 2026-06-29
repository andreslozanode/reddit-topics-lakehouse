"""Shared pytest fixtures: a local Spark session and synthetic frames."""

from __future__ import annotations

import random
import time

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


def _rand_word() -> str:
    return random.choice(["spark", "delta", "model", "data", "pipeline", "cloud", "etl", "vector"])


@pytest.fixture
def raw_posts(spark):
    rows = []
    for i in range(50):
        rows.append(
            {
                "type": "post",
                "id": f"p{i:05d}",
                "subreddit.id": "sr_ds",
                "subreddit.name": "datascience" if i % 2 == 0 else "MachineLearning",
                "subreddit.nsfw": False,
                "created_utc": int(time.time()) - i * 3600,
                "permalink": f"/r/x/post/{i}",
                "title": f"{_rand_word()} {_rand_word()} title {i}" + ("?" if i % 3 == 0 else ""),
                "selftext": f"{_rand_word()} {_rand_word()} body" if i % 4 else "[removed]",
                "domain": "self.reddit",
                "url": f"https://e.com/{i}" if i % 2 else None,
                "score": i - 5,
            }
        )
    return spark.createDataFrame(rows)


@pytest.fixture
def raw_comments(spark):
    rows = []
    for i in range(60):
        rows.append(
            {
                "type": "comment",
                "id": f"c{i:05d}",
                "subreddit.id": "sr_ds",
                "subreddit.name": "dataengineering",
                "subreddit.nsfw": False,
                "created_utc": int(time.time()) - i * 1800,
                "permalink": f"/r/x/comment/{i}",
                "body": f"{_rand_word()} {_rand_word()} comment {i}" if i % 5 else "[deleted]",
                "sentiment": "0.5" if i % 2 == 0 else "-0.3",
                "score": i % 20,
            }
        )
    return spark.createDataFrame(rows)
