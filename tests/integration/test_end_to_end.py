"""End-to-end: synthetic raw -> silver -> gold -> train a small engagement model.

Runs entirely on a local Spark session with tiny data so it stays fast and
needs no Delta or network. Validates that the stages compose correctly.
"""

from reddit_lakehouse.ml import engagement
from reddit_lakehouse.transform import gold
from reddit_lakehouse.transform.silver import to_events


def test_full_local_pipeline(raw_posts, raw_comments):
    events = to_events(raw_posts, raw_comments, topic="sample")
    assert events.count() > 0

    daily = gold.subreddit_daily(events)
    assert daily.count() > 0

    model, metrics = engagement.train_engagement(
        events, vocab_size=200, min_doc_freq=1, percentile=0.7, cv_folds=2, seed=1
    )
    assert "areaUnderROC" in metrics
    preds = engagement.predict_engagement(model, events)
    assert {"is_high_engagement", "engagement_probability"} <= set(preds.columns)
    assert preds.count() == events.count()
