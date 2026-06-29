from pyspark.sql import functions as F

from reddit_lakehouse.transform.silver import to_events


def test_to_events_unifies_and_cleans(raw_posts, raw_comments):
    events = to_events(raw_posts, raw_comments, topic="sample")
    cols = set(events.columns)
    assert {"event_id", "text", "created_ts", "text_length", "word_count"} <= cols
    # Both event types present
    types = {r["event_type"] for r in events.select("event_type").distinct().collect()}
    assert types == {"post", "comment"}


def test_to_events_removes_deleted_content(raw_comments):
    events = to_events(None, raw_comments, topic="t")
    # No row should have text equal to a removed token
    bad = events.filter(F.lower(F.col("text")).isin(["[deleted]", "[removed]"])).count()
    assert bad == 0


def test_to_events_is_deduplicated(raw_posts):
    doubled = raw_posts.unionByName(raw_posts)
    events = to_events(doubled, None, topic="t")
    assert events.count() == events.select("event_id").distinct().count()


def test_to_events_requires_input():
    import pytest

    with pytest.raises(ValueError):
        to_events(None, None, topic="t")
