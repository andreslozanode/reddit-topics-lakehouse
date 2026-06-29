# Deployment

Two ways to run the platform: **locally** (synthetic data, no cloud) for
development and CI, and on **Databricks** via a Databricks Asset Bundle for
staging/production.

---

## Prerequisites

- Python ≥ 3.10
- Java 17 (for local Spark)
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) ≥ 0.220 (for cloud)
- A Databricks workspace with Unity Catalog enabled (for cloud)

---

## Local (synthetic data)

No Databricks account required — the pipeline falls back to local parquet when
Delta/UC are unavailable.

```bash
# 1. Install with dev + spark extras
pip install -e ".[spark,ingestion,ml,dev]"

# 2. Generate a synthetic SocialGrep-schema dataset
python scripts/generate_sample_data.py --out .data/sample --topic demo --rows 2000

# 3. Run the unit + integration test suite
LAKEHOUSE_ENV=dev pytest -q
```

The integration test (`tests/integration/test_end_to_end.py`) exercises the full
silver → gold → model-training path on the synthetic data.

---

## Databricks (Asset Bundle)

### One-time setup

1. Authenticate the CLI: `databricks auth login --host https://<workspace>`.
2. Set the workspace host per target (CLI profile or `DATABRICKS_HOST`).
3. Provision Unity Catalog objects (catalog, schemas, volume, grants):
   ```bash
   databricks bundle run reddit_setup -t dev   # if exposed as a job
   # or run the reddit-setup entrypoint on an interactive cluster
   ```

### Validate & deploy

```bash
databricks bundle validate -t dev      # build wheel + check config
databricks bundle deploy   -t dev      # upload wheel + create jobs
databricks bundle run reddit_lakehouse_batch -t dev
```

Swap `-t dev` for `-t staging` or `-t prod`. The bundle builds the project wheel
(`artifacts.default_wheel`) and wires each pipeline entry point as a
`python_wheel_task` in a multi-task job on a shared Photon job cluster.

### Jobs created

| Job | Tasks | Schedule |
|---|---|---|
| `reddit_lakehouse_batch` | bronze → silver → gold → ml_training | daily 06:00 UTC (prod) |
| `reddit_lakehouse_inference` | batch_inference (`@champion` models) | every 4h (prod) |

Schedules are **paused** on dev/staging and **unpaused** on prod
(`schedule_pause_status` variable).

---

## CI/CD

**GitHub Actions**

- `.github/workflows/ci.yml` — on every push/PR: ruff, black, mypy, pytest across
  Python 3.10/3.11/3.12, plus `databricks bundle validate`.
- `.github/workflows/cd.yml` — on `main`: deploy to **staging**; on a `v*` tag:
  deploy to **prod** (guarded by a GitHub Environment with required reviewers).

Required repository secrets:
`DATABRICKS_HOST_DEV/STAGING/PROD`, `DATABRICKS_TOKEN_DEV/STAGING/PROD`.

**Jenkins**

`Jenkinsfile` mirrors the same gates (quality → test → build wheel → deploy
staging → manual approval → deploy prod) using Jenkins credentials
`databricks-host-*` / `databricks-token-*`.

---

## Promotion workflow (models)

1. `ml_training` registers new model versions in Unity Catalog.
2. Validate the candidate (metrics in MLflow, optional challenger comparison).
3. Move the `@champion` alias to the new version.
4. `reddit_lakehouse_inference` picks it up automatically on the next run — no
   redeploy needed.

---

## Rollback

- **Code/jobs**: redeploy a previous tag (`databricks bundle deploy -t prod` from
  the prior commit) — bundles are declarative.
- **Models**: repoint the `@champion` alias to the previous version.
- **Data**: silver/gold are Delta tables — use time travel
  (`RESTORE TABLE ... TO VERSION AS OF n`) if a bad batch lands.
