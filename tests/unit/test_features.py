from reddit_lakehouse.ml.features import add_derived_features, supervised_feature_pipeline
from reddit_lakehouse.transform.silver import to_events


def test_add_derived_features(raw_posts):
    events = to_events(raw_posts, None, topic="t")
    out = add_derived_features(events)
    assert {"has_url", "is_question", "is_post"} <= set(out.columns)
    # is_post must be 1.0 for posts
    assert out.select("is_post").distinct().collect()[0]["is_post"] == 1.0


def test_supervised_pipeline_fits_and_produces_features(raw_posts, raw_comments):
    events = add_derived_features(to_events(raw_posts, raw_comments, topic="t"))
    events = events.withColumn("sentiment_label", events["event_type"])  # 2-class proxy
    pipeline = supervised_feature_pipeline(
        vocab_size=100, min_doc_freq=1, label_index_input="sentiment_label"
    )
    model = pipeline.fit(events)
    transformed = model.transform(events)
    assert "features" in transformed.columns
    assert "label" in transformed.columns
    assert transformed.select("features").take(1)
