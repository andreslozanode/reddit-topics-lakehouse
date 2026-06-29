# Data Dictionary

All tables live under a per-environment Unity Catalog catalog
(`reddit_dev` / `reddit_staging` / `reddit_prod`) and are organized into three
schemas following the medallion architecture: `bronze`, `silver`, `gold`.

---

## Bronze — raw, append-only

Raw landing of the public SocialGrep Reddit datasets, one row per source record,
partitioned by `_topic` and `_ingest_date`. Ingestion metadata columns are
prefixed with `_`.

### `bronze.posts_raw`

| Column | Type | Description |
|---|---|---|
| `type` | string | Source record type (`post`). |
| `id` | string | Reddit post id (base36). |
| `subreddit.id` / `subreddit.name` / `subreddit.nsfw` | struct | Subreddit metadata (dotted fields from source). |
| `created_utc` | long | Creation time, Unix epoch seconds. |
| `permalink` | string | Relative Reddit permalink. |
| `domain` | string | Linked domain (for link posts). |
| `url` | string | Linked URL. |
| `selftext` | string | Post body (self/text posts). |
| `title` | string | Post title. |
| `score` | long | Net score (ups − downs). |
| `_topic` | string | Dataset theme label (partition). |
| `_batch_id` | string | Ingestion batch id (UTC timestamp). |
| `_ingested_at` | timestamp | Ingestion wall-clock time. |
| `_ingest_date` | date | Ingestion date (partition). |

### `bronze.comments_raw`

Same envelope as `posts_raw`, with `body` instead of `title`/`selftext`, plus a
source `sentiment` score.

| Column | Type | Description |
|---|---|---|
| `body` | string | Comment text. |
| `sentiment` | double | SocialGrep in-house sentiment score (∈ [−1, 1]). |
| *(shared)* | | `type`, `id`, `subreddit.*`, `created_utc`, `permalink`, `score`, `_*` as above. |

---

## Silver — canonical unified events

### `silver.events`

One row per event (post **or** comment), unified into a single schema. This is
the analytical core consumed by every gold aggregate and ML model. Deduplicated
on `event_id` (idempotent re-ingestion).

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Stable id `{type}_{id}`. Primary key. |
| `event_type` | string | `post` or `comment`. |
| `topic` | string | Dataset theme. |
| `subreddit` | string | Subreddit name (flattened). |
| `subreddit_nsfw` | boolean | NSFW flag. |
| `created_ts` | timestamp | Event creation timestamp. |
| `created_date` | string | `created_ts` truncated to date. |
| `created_hour` | long | Hour of day, 0–23. |
| `created_dow` | long | Day of week, 1=Sun … 7=Sat. |
| `title` | string | Cleaned title (posts; empty for comments). |
| `text` | string | NLP text: title+selftext (posts) or body (comments). |
| `text_length` | long | Character length of `text`. |
| `word_count` | long | Whitespace token count of `text`. |
| `score` | long | Net score. |
| `permalink` / `domain` / `url` | string | Source references. |
| `sentiment_source` | double | Source sentiment (comments) or null. |

---

## Gold — BI-ready aggregates & ML output

### Aggregate tables (from the `gold` pipeline)

**`gold.subreddit_daily`** — daily activity per subreddit.
`topic, subreddit, created_date, event_count, post_count, comment_count,
avg_score, median_score, max_score, total_score, avg_text_length, avg_word_count`

**`gold.subreddit_summary`** — lifetime summary per subreddit.
`topic, subreddit, event_count, active_days, avg_score, p90_score, max_score,
first_seen, last_seen, nsfw`

**`gold.hourly_activity`** — activity by day-of-week × hour.
`topic, created_dow, created_hour, event_count, avg_score`

### ML output tables (from the `ml_training` / `inference` pipelines)

**`gold.topic_terms`** — LDA topic dictionary.
`topic_id, terms (array<string>), term_weights (array<double>), label`

**`gold.post_topics`** — dominant topic per event.
`event_id, topic, subreddit, created_date, score, dominant_topic, topic_confidence`

**`gold.post_sentiment`** — predicted sentiment per event.
`event_id, topic, subreddit, created_date, sentiment_label` (`negative|neutral|positive`)

**`gold.engagement_predictions`** — high-engagement probability per event.
`event_id, topic, subreddit, created_date, score, is_high_engagement,
engagement_probability`

### BI views (from `sql/gold_views.sql`)

| View | Purpose |
|---|---|
| `vw_subreddit_daily_trend` | Daily volume with 7-day rolling windows and DoD deltas. |
| `vw_subreddit_leaderboard` | Ranked subreddits per topic. |
| `vw_activity_heatmap` | Day-of-week × hour activity matrix. |
| `vw_post_topics_labeled` | Topic assignments joined to readable labels. |
| `vw_topic_share_daily` | Daily share of activity per LDA topic. |
| `vw_sentiment_breakdown` | Sentiment counts and net-sentiment ratios. |
| `vw_engagement_summary` | Predicted high-engagement rates. |
| `vw_topic_kpis` | One-row-per-topic executive KPIs. |
