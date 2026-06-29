-- =============================================================================
-- Reddit Topics Lakehouse — Dashboard query pack
-- -----------------------------------------------------------------------------
-- Ready-to-paste queries for a Databricks SQL / Lakeview dashboard (or any BI
-- tool). Each query maps to one dashboard tile. They depend on the views in
-- sql/gold_views.sql. Replace ${catalog} at deploy time.
-- =============================================================================

USE CATALOG ${catalog};
USE SCHEMA gold;

-- TILE 1 — KPI cards: totals per topic ----------------------------------------
-- Counter visuals: total_events, subreddits, avg_net_sentiment.
SELECT topic, total_events, subreddits, avg_score, avg_net_sentiment
FROM vw_topic_kpis
ORDER BY total_events DESC;

-- TILE 2 — Daily volume trend (line chart) ------------------------------------
-- X: created_date, Y: events_rolling_7d, series: topic.
SELECT topic, created_date, events_rolling_7d
FROM vw_subreddit_daily_trend
WHERE created_date >= DATEADD(DAY, -90, CURRENT_DATE())
ORDER BY created_date;

-- TILE 3 — Top 15 subreddits (bar chart) --------------------------------------
SELECT subreddit, topic, event_count, avg_score
FROM vw_subreddit_leaderboard
WHERE rank_by_volume <= 15
ORDER BY event_count DESC;

-- TILE 4 — Activity heatmap (heatmap: day_of_week x hour_of_day) ---------------
SELECT day_label, hour_of_day, SUM(event_count) AS events
FROM vw_activity_heatmap
GROUP BY day_label, hour_of_day, day_of_week
ORDER BY day_of_week, hour_of_day;

-- TILE 5 — Topic share over time (stacked area) -------------------------------
SELECT created_date, topic_label, SUM(topic_share) AS topic_share
FROM vw_topic_share_daily
WHERE created_date >= DATEADD(DAY, -60, CURRENT_DATE())
GROUP BY created_date, topic_label
ORDER BY created_date;

-- TILE 6 — Topic dictionary (table) -------------------------------------------
SELECT topic_id, label, terms
FROM topic_terms
ORDER BY topic_id;

-- TILE 7 — Net sentiment trend (line chart, zero-baseline) --------------------
SELECT topic, created_date, AVG(net_sentiment) AS net_sentiment
FROM vw_sentiment_breakdown
WHERE created_date >= DATEADD(DAY, -90, CURRENT_DATE())
GROUP BY topic, created_date
ORDER BY created_date;

-- TILE 8 — Sentiment mix by subreddit (100% stacked bar) ----------------------
SELECT subreddit, SUM(positive) AS positive, SUM(neutral) AS neutral, SUM(negative) AS negative
FROM vw_sentiment_breakdown
GROUP BY subreddit
ORDER BY (SUM(positive) + SUM(neutral) + SUM(negative)) DESC
LIMIT 20;

-- TILE 9 — Predicted high-engagement rate by topic (bar) ----------------------
SELECT topic,
       SUM(predicted_high) AS predicted_high_events,
       SUM(scored_events)  AS scored_events,
       ROUND(SUM(predicted_high) / NULLIF(SUM(scored_events), 0), 4) AS high_engagement_rate
FROM vw_engagement_summary
GROUP BY topic
ORDER BY high_engagement_rate DESC;

-- TILE 10 — Engagement probability vs actual score (scatter sample) -----------
SELECT subreddit, engagement_probability, avg_actual_score AS avg_score, scored_events
FROM vw_engagement_summary
WHERE scored_events >= 5
ORDER BY engagement_probability DESC
LIMIT 500;
