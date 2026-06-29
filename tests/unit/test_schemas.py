from reddit_lakehouse.common.schemas import normalize_raw


def test_normalize_posts_flattens_subreddit(raw_posts):
    out = normalize_raw(raw_posts, event_type="post", topic="sample")
    cols = set(out.columns)
    assert {"event_id", "event_type", "topic", "subreddit", "score"} <= cols
    assert "subreddit.name" not in cols
    row = out.filter("event_id = 'post_p00000'").collect()[0]
    assert row["event_type"] == "post"
    assert row["topic"] == "sample"
    assert row["subreddit"] in {"datascience", "MachineLearning"}


def test_normalize_comments_uses_body(raw_comments):
    out = normalize_raw(raw_comments, event_type="comment", topic="t")
    row = out.filter("event_id = 'comment_c00001'").collect()[0]
    assert row["body_text"] is not None
    assert row["sentiment_source"] is not None
