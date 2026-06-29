-- =============================================================================
-- Reddit Topics Lakehouse — Gold BI views
-- -----------------------------------------------------------------------------
-- These views sit on top of the gold aggregate + ML output tables and present
-- a clean, denormalized, BI-friendly surface for Power BI / Tableau / Looker.
-- They are environment-agnostic: substitute the catalog at deploy time, e.g.
--   databricks sql --warehouse <id> -f sql/gold_views.sql --param catalog=reddit_prod
-- or run through the SQL editor after a find/replace of ${catalog}.
--
-- Underlying gold tables (written by the gold + ml pipelines):
--   <catalog>.gold.subreddit_daily
--   <catalog>.gold.subreddit_summary
--   <catalog>.gold.hourly_activity
--   <catalog>.gold.topic_terms
--   <catalog>.gold.post_topics
--   <catalog>.gold.post_sentiment
--   <catalog>.gold.engagement_predictions
-- =============================================================================

USE CATALOG ${catalog};
CREATE SCHEMA IF NOT EXISTS gold;
USE SCHEMA gold;

-- -----------------------------------------------------------------------------
-- 1. Daily activity, enriched with day-over-day deltas and rolling averages.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_subreddit_daily_trend AS
SELECT
    topic,
    subreddit,
    created_date,
    event_count,
    post_count,
    comment_count,
    avg_score,
    median_score,
    total_score,
    avg_word_count,
    SUM(event_count) OVER (
        PARTITION BY topic, subreddit
        ORDER BY created_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS events_rolling_7d,
    AVG(avg_score) OVER (
        PARTITION BY topic, subreddit
        ORDER BY created_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS avg_score_rolling_7d,
    event_count - LAG(event_count) OVER (
        PARTITION BY topic, subreddit ORDER BY created_date
    ) AS event_count_dod_delta
FROM subreddit_daily;

-- -----------------------------------------------------------------------------
-- 2. Leaderboard: top subreddits per topic by total engagement.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_subreddit_leaderboard AS
SELECT
    topic,
    subreddit,
    event_count,
    active_days,
    avg_score,
    p90_score,
    max_score,
    nsfw,
    first_seen,
    last_seen,
    ROUND(event_count / NULLIF(active_days, 0), 2) AS events_per_active_day,
    RANK() OVER (PARTITION BY topic ORDER BY event_count DESC) AS rank_by_volume,
    RANK() OVER (PARTITION BY topic ORDER BY avg_score DESC)    AS rank_by_avg_score
FROM subreddit_summary;

-- -----------------------------------------------------------------------------
-- 3. Activity heatmap source (day-of-week x hour-of-day).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_activity_heatmap AS
SELECT
    topic,
    created_dow AS day_of_week,
    CASE created_dow
        WHEN 1 THEN 'Sun' WHEN 2 THEN 'Mon' WHEN 3 THEN 'Tue'
        WHEN 4 THEN 'Wed' WHEN 5 THEN 'Thu' WHEN 6 THEN 'Fri'
        WHEN 7 THEN 'Sat'
    END AS day_label,
    created_hour AS hour_of_day,
    event_count,
    avg_score
FROM hourly_activity;

-- -----------------------------------------------------------------------------
-- 4. Topic model output joined to human-readable labels.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_post_topics_labeled AS
SELECT
    pt.event_id,
    pt.topic,
    pt.subreddit,
    pt.created_date,
    pt.score,
    pt.dominant_topic,
    pt.topic_confidence,
    tt.label AS topic_label,
    tt.terms AS topic_terms
FROM post_topics pt
LEFT JOIN topic_terms tt
    ON pt.dominant_topic = tt.topic_id;

-- -----------------------------------------------------------------------------
-- 5. Daily topic share (what share of activity each LDA topic captures).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_topic_share_daily AS
WITH daily AS (
    SELECT topic, created_date, dominant_topic, COUNT(*) AS n
    FROM post_topics
    GROUP BY topic, created_date, dominant_topic
)
SELECT
    d.topic,
    d.created_date,
    d.dominant_topic,
    tt.label AS topic_label,
    d.n AS event_count,
    ROUND(d.n / SUM(d.n) OVER (PARTITION BY d.topic, d.created_date), 4) AS topic_share
FROM daily d
LEFT JOIN topic_terms tt ON d.dominant_topic = tt.topic_id;

-- -----------------------------------------------------------------------------
-- 6. Sentiment breakdown per subreddit per day (counts + ratios).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_sentiment_breakdown AS
WITH agg AS (
    SELECT
        topic,
        subreddit,
        created_date,
        COUNT(*) AS total,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive,
        SUM(CASE WHEN sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS neutral,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative
    FROM post_sentiment
    GROUP BY topic, subreddit, created_date
)
SELECT
    *,
    ROUND(positive / NULLIF(total, 0), 4) AS positive_ratio,
    ROUND(negative / NULLIF(total, 0), 4) AS negative_ratio,
    ROUND((positive - negative) / NULLIF(total, 0), 4) AS net_sentiment
FROM agg;

-- -----------------------------------------------------------------------------
-- 7. Engagement model — precision-style summary by decile bucket.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_engagement_summary AS
SELECT
    topic,
    subreddit,
    created_date,
    COUNT(*) AS scored_events,
    SUM(CAST(is_high_engagement AS INT)) AS predicted_high,
    ROUND(AVG(engagement_probability), 4) AS avg_engagement_proba,
    ROUND(AVG(score), 2) AS avg_actual_score
FROM engagement_predictions
GROUP BY topic, subreddit, created_date;

-- -----------------------------------------------------------------------------
-- 8. Executive one-row-per-topic KPI surface for report headers.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_topic_kpis AS
SELECT
    s.topic,
    COUNT(DISTINCT s.subreddit) AS subreddits,
    SUM(s.event_count) AS total_events,
    ROUND(AVG(s.avg_score), 2) AS avg_score,
    MIN(s.first_seen) AS earliest_event,
    MAX(s.last_seen) AS latest_event,
    ROUND(AVG(b.net_sentiment), 4) AS avg_net_sentiment
FROM subreddit_summary s
LEFT JOIN (
    SELECT topic, AVG((positive - negative) / NULLIF(total, 0)) AS net_sentiment
    FROM (
        SELECT topic,
               COUNT(*) AS total,
               SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive,
               SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative
        FROM post_sentiment GROUP BY topic
    ) GROUP BY topic
) b ON s.topic = b.topic
GROUP BY s.topic;
