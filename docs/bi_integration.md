# BI Integration

The gold layer is designed to be consumed directly by BI tools. All tiles should
read from the **views** in `sql/gold_views.sql` (not the base tables) so business
logic stays in one place. Create the views once per environment:

```bash
databricks sql -e "$(sed 's/${catalog}/reddit_prod/g' sql/gold_views.sql)" \
  --warehouse <warehouse_id>
```

Connection details below assume a **Databricks SQL Warehouse**. Grab the JDBC/ODBC
values from *SQL Warehouses → your warehouse → Connection details*.

---

## Databricks SQL / Lakeview (native)

The fastest path. Import `sql/dashboards/reddit_topics_dashboard.sql`: each query
maps to one tile (KPI cards, volume trend, leaderboard, activity heatmap, topic
share, sentiment trend, engagement rate). Schedule the dashboard to refresh after
the daily batch job completes.

---

## Power BI

1. **Get Data → Azure Databricks** (or the Databricks connector).
2. Enter the **Server Hostname** and **HTTP Path** from the warehouse connection
   details; authenticate with a PAT or Microsoft Entra ID.
3. Choose **DirectQuery** for always-fresh tiles, or **Import** for snappier
   slicers on smaller marts (`subreddit_summary`, `topic_terms`).
4. Load the `vw_*` views from the `gold` schema. Recommended starting model:
   - Fact: `vw_subreddit_daily_trend`, `vw_sentiment_breakdown`,
     `vw_engagement_summary`.
   - Dimension-like: `vw_subreddit_leaderboard`, `topic_terms`.
5. Mark `created_date` as a date table for time intelligence.

Tip: for the activity heatmap, pivot `vw_activity_heatmap` on `day_label` ×
`hour_of_day` with `event_count` as values.

---

## Tableau

1. **Connect → Databricks**. Provide Server Hostname, HTTP Path, and a PAT.
2. Select catalog `reddit_<env>` and schema `gold`.
3. Drag the `vw_*` views onto the canvas. They are pre-aggregated, so most sheets
   need no further joins.
4. Use **live** connections for trend tiles; create extracts for the leaderboard
   and topic dictionary if you want offline snapshots.
5. Suggested worksheets:
   - Line: `vw_subreddit_daily_trend.events_rolling_7d` by `created_date`.
   - Heatmap: `vw_activity_heatmap` (`day_label` × `hour_of_day`).
   - Diverging bar: `vw_sentiment_breakdown.net_sentiment`.

---

## Looker

1. Point a connection at the Databricks SQL Warehouse (Looker's Databricks
   dialect).
2. Generate a starter LookML model from the `gold` schema, then keep only the
   `vw_*` views as **explores**.
3. Define measures on top of the already-aggregated columns (e.g. `sum:
   event_count`, `average: avg_score`, `average: net_sentiment`).
4. Use `created_date` as the primary time dimension with day/week/month
   timeframes.

---

## Refresh & freshness

- The `reddit_lakehouse_batch` job rebuilds gold daily at 06:00 UTC.
- `reddit_lakehouse_inference` refreshes the ML prediction tables every 4 hours.
- Point BI refresh schedules to run shortly after these, or use DirectQuery/live
  connections to avoid a second copy.

## Semantics cheat-sheet

| Question | View / column |
|---|---|
| Which subreddits drive a topic? | `vw_subreddit_leaderboard` (`rank_by_volume`) |
| When are users most active? | `vw_activity_heatmap` |
| What is each topic *about*? | `topic_terms.label` / `vw_post_topics_labeled` |
| Is sentiment improving? | `vw_sentiment_breakdown.net_sentiment` over time |
| What's likely to go viral? | `vw_engagement_summary.high_engagement_rate` |
| One number per topic? | `vw_topic_kpis` |
