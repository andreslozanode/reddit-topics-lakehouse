# Reddit Topics Lakehouse

End-to-end **Databricks lakehouse + ML platform** over public Reddit topic
datasets. Medallion architecture on Unity Catalog, three registered Spark ML
models (topic modeling, sentiment, engagement), a BI-ready gold layer, and full
CI/CD via Databricks Asset Bundles + GitHub Actions / Jenkins.

No LLMs, no chatbots — only reproducible, governed, classical ML.

---

## Highlights

- **Medallion architecture** (bronze → silver → gold) on **Unity Catalog**.
- **Unified silver `events` table** — posts and comments normalized into one
  canonical, deduplicated schema; transforms are pure functions, unit-tested
  without a cluster.
- **Three ML models** logged to MLflow and registered in the UC model registry:
  LDA topic modeling, multinomial Logistic Regression sentiment, GBT engagement
  classifier with cross-validation. Batch inference resolves models by
  `@champion` alias.
- **BI-ready** gold tables + SQL views for Power BI / Tableau / Looker / Lakeview.
- **Performance**: AQE, liquid clustering / Z-ORDER, optimizeWrite + autoCompact,
  deletion vectors, MERGE upserts, broadcast joins, Arrow, Photon.
- **CI/CD**: lint + type-check + test matrix (3.10–3.12), `bundle validate`, and
  gated staging → prod deploys.
- **Dataset-agnostic**: any [SocialGrep](https://huggingface.co/SocialGrep)
  Reddit dataset (CC-BY 4.0) plugs in via config — no code changes.

---

## Architecture

```
Hugging Face (SocialGrep)
        │ download
        ▼
   Raw landing (Volume)
        │ bronze (+ DQ gate)
        ▼
 BRONZE  posts_raw / comments_raw          append-only, partitioned
        │ silver: normalize → union → clean → enrich → dedupe
        ▼
 SILVER  events                            one canonical row per event
        │
   ┌────┴───────────────┐
   │ gold               │ ml_training (MLflow + UC registry)
   ▼                    ▼
 GOLD  subreddit_daily       topic_terms / post_topics
       subreddit_summary     post_sentiment
       hourly_activity       engagement_predictions
        │ sql/gold_views.sql
        ▼
   BI views → Power BI / Tableau / Looker / Lakeview
```

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/data_dictionary.md`](docs/data_dictionary.md) for every table's schema.

---

## Quickstart (local, synthetic data)

No Databricks account needed — the pipeline falls back to local parquet when
Delta/Unity Catalog aren't present.

```bash
# Install (spark + ingestion + ml + dev extras)
pip install -e ".[spark,ingestion,ml,dev]"

# Generate a synthetic SocialGrep-schema dataset
python scripts/generate_sample_data.py --out .data/sample --topic demo --rows 2000

# Run the full test suite (unit + end-to-end)
LAKEHOUSE_ENV=dev pytest -q
```

Requires Python ≥ 3.10 and Java 17 for local Spark.

---

## Run on Databricks

```bash
databricks bundle validate -t dev
databricks bundle deploy   -t dev
databricks bundle run reddit_lakehouse_batch -t dev
```

Targets: `dev`, `staging`, `prod`. The bundle builds the project wheel and wires
each pipeline entry point as a `python_wheel_task` in a multi-task job. Full
instructions in [`docs/deployment.md`](docs/deployment.md).

### Pipeline entry points

| Command | Stage |
|---|---|
| `reddit-setup`  | Provision UC catalog/schemas/volume/grants |
| `reddit-bronze` | Download + land raw Delta (with DQ gate) |
| `reddit-silver` | Build the unified `events` table |
| `reddit-gold`   | Build BI aggregate tables |
| `reddit-train`  | Train + register the three ML models |
| `reddit-infer`  | Score with `@champion` models |

Each accepts `--env {dev|staging|prod}` and optional `--config <path>`.

---

## Datasets

Built on the public [SocialGrep Reddit datasets](https://huggingface.co/SocialGrep)
(CC-BY 4.0), which share one schema so themed datasets coexist in the same
lakehouse. Configured in `conf/pipeline.yml` (defaults: `the-reddit-dataset-dataset`,
`the-reddit-nft-dataset`, `the-reddit-covid-dataset`). Add a topic by appending a
`SourceSpec` — no code change required.

---

## BI integration

Gold views power dashboards in Power BI, Tableau, Looker, or Databricks Lakeview.
Connection steps and a tile-by-tile semantics guide are in
[`docs/bi_integration.md`](docs/bi_integration.md); ready-to-paste queries are in
[`sql/dashboards/reddit_topics_dashboard.sql`](sql/dashboards/reddit_topics_dashboard.sql).

---

## Project structure

```
reddit-topics-lakehouse/
├── conf/                  pipeline.yml (env config), logging.yaml
├── databricks.yml         Asset Bundle root
├── resources/             job + cluster definitions
├── src/reddit_lakehouse/  the package (common, ingestion, transform,
│                          quality, ml, governance, pipelines, config)
├── sql/                   gold views + dashboard queries
├── notebooks/             Databricks exploration notebooks
├── scripts/               synthetic data generator
├── tests/                 unit + integration tests
├── docs/                  architecture, BI, deployment, data dictionary
├── .github/workflows/     CI + CD
└── Jenkinsfile            Jenkins CI/CD mirror
```

---

## Development

```bash
make install     # editable install with all extras
make lint        # ruff + black --check
make typecheck   # mypy
make test        # pytest
make sample      # generate synthetic data
make validate    # databricks bundle validate -t dev
```

Pre-commit hooks (ruff, black, mypy) are configured in
`.pre-commit-config.yaml` — run `pre-commit install` once.

---

## License

[MIT](LICENSE). Datasets are licensed separately under CC-BY 4.0 by SocialGrep.
