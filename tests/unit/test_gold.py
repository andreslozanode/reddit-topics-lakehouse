from reddit_lakehouse.transform import gold
from reddit_lakehouse.transform.silver import to_events


def test_subreddit_daily_grain(raw_posts, raw_comments):
    events = to_events(raw_posts, raw_comments, topic="sample")
    daily = gold.subreddit_daily(events)
    assert {"topic", "subreddit", "created_date", "event_count", "avg_score"} <= set(daily.columns)
    # Grain uniqueness
    keys = daily.select("topic", "subreddit", "created_date")
    assert keys.count() == keys.distinct().count()


def test_subreddit_summary_has_lifetime_metrics(raw_posts, raw_comments):
    events = to_events(raw_posts, raw_comments, topic="sample")
    summary = gold.subreddit_summary(events)
    assert {"event_count", "first_seen", "last_seen", "p90_score"} <= set(summary.columns)
    assert summary.count() >= 1


def test_hourly_activity(raw_posts):
    events = to_events(raw_posts, None, topic="t")
    hourly = gold.hourly_activity(events)
    assert {"created_hour", "created_dow", "event_count"} <= set(hourly.columns)
