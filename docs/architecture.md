# Architecture

## Overview

An end-to-end, production-grade **lakehouse + ML platform** on Databricks over
public Reddit topic datasets. It follows the medallion architecture, governs
every asset through Unity Catalog, trains and registers three classical ML
models with MLflow, and exposes a BI-ready gold layer for Power BI / Tableau /
Looker. There is **no LLM / chatbot** anywhere in the system — all ML is
reproducible Spark ML (LDA, Logistic Regression, Gradient-Boosted Trees).

## End-to-end flow

```
                         Hugging Face (SocialGrep, CC-BY 4.0)
                                       │  snapshot_download
                                       ▼
                          ┌────────────────────────┐
                          │   Raw landing (Volume)  │   /Volumes/<cat>/bronze/raw/landing
                          └────────────┬───────────┘
                                       │  bronze pipeline (+ DQ gate)
                                       ▼
   BRONZE   posts_raw / comments_raw  (append-only, partitioned by topic+date)
                                       │  silver pipeline: normalize → union → clean → enrich
                                       ▼
   SILVER   events  (one canonical row per post/comment, deduped on event_id)
                                       │
                  ┌────────────────────┼─────────────────────┐
                  │ gold pipeline      │ ml_training pipeline │
                  ▼                    ▼                      ▼
   GOLD   subreddit_daily      topic_terms            (MLflow runs +
          subreddit_summary    post_topics             UC model registry:
          hourly_activity      post_sentiment          reddit_topic_lda,
                               engagement_predictions   reddit_sentiment_clf,
                                                         reddit_engagement_clf)
                                       │  gold_views.sql
                                       ▼
                         BI views  →  Power BI / Tableau / Looker / Lakeview
```

## Layer responsibilities

**Bronze** — faithful, append-only capture of the source parquet with ingestion
metadata (`_topic`, `_batch_id`, `_ingested_at`, `_ingest_date`). No business
logic; a minimal DQ gate (row count, key not-null) blocks empty/corrupt loads.

**Silver** — the canonical model. Posts and comments are normalized to a shared
schema and **unified into a single `events` table** (one row per event). Text is
cleaned, timestamps and calendar parts derived, and rows deduplicated on the
stable `event_id`. Transforms are **pure DataFrame functions** (no I/O) so they
are unit-testable without a cluster.

**Gold** — denormalized, aggregate and ML-output tables optimized for reads.
Aggregates (daily/summary/hourly) plus model outputs (topics, sentiment,
engagement) feed the BI views.

## ML components

| Model | Algorithm | Output table | Registered name |
|---|---|---|---|
| Topic modeling | LDA (online optimizer) over TF-IDF | `post_topics`, `topic_terms` | `reddit_topic_lda` |
| Sentiment | Logistic Regression (multinomial) | `post_sentiment` | `reddit_sentiment_clf` |
| Engagement | Gradient-Boosted Trees + CrossValidator | `engagement_predictions` | `reddit_engagement_clf` |

Each model is logged to MLflow and registered in the **Unity Catalog model
registry**. Batch inference resolves models by the `@champion` alias, so promoting
a new version requires no code change. The sentiment pipeline fits its label
`StringIndexer` *outside* the served pipeline and appends an `IndexToString`
stage, making the served model fully self-contained (no label column needed at
inference).

## Governance (Unity Catalog)

- Three-level namespace `catalog.schema.table` per environment.
- A managed **Volume** holds the raw landing zone.
- Catalog/schema/volume creation and grants are provisioned idempotently
  (`governance/unity_catalog.py`); table and column comments are applied from a
  central metadata map.
- Models are first-class UC objects, versioned and alias-promoted.

## Performance & optimization

Applied from application code and cluster config:

- **AQE** — adaptive coalescing + skew-join handling.
- **Liquid clustering / Z-ORDER** on hot predicates (e.g. `subreddit`,
  `created_date`) plus `OPTIMIZE`.
- **optimizeWrite + autoCompact** to avoid the small-file problem.
- **Deletion vectors** and other Delta table properties enabled by default.
- **MERGE** upserts for idempotent silver refresh.
- **Broadcast joins** for dimension-sized frames; **Arrow** for JVM↔Python.
- **Photon** runtime engine on job clusters.

## Code layout

```
src/reddit_lakehouse/
  common/        spark session, schemas, IO, Delta table helpers, logging
  ingestion/     Hugging Face download + bronze frame builders
  transform/     pure silver + gold transforms
  quality/       dependency-free data-quality rule framework
  ml/            features, topic modeling, sentiment, engagement, batch inference
  governance/    Unity Catalog provisioning + table metadata
  pipelines/     CLI entry points (setup/bronze/silver/gold/train/infer)
  config.py      typed, environment-aware config loader
```

## Environments

`dev`, `staging`, `prod` share one logical pipeline; only catalog, landing path,
SQL warehouse and a few thresholds differ (resolved by `config.load_config` via a
defaults ← environment deep-merge). Deployment is driven by a Databricks Asset
Bundle (`databricks.yml` + `resources/`).
