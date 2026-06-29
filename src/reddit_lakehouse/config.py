"""Typed, environment-aware configuration loaded from ``conf/pipeline.yml``.

The same logical pipeline runs across ``dev`` / ``staging`` / ``prod`` targets.
Only catalog, paths and a few thresholds change between environments; everything
else (schemas, source datasets, ML hyper-parameters) is shared.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "conf" / "pipeline.yml"


@dataclass(frozen=True)
class SourceSpec:
    """A single public Reddit dataset to ingest.

    ``repo_id`` points at a Hugging Face dataset that follows the SocialGrep
    schema. ``topic`` is a human-readable label propagated into the lakehouse so
    several themed datasets can coexist in the same tables.
    """

    topic: str
    repo_id: str
    posts_glob: str = "*posts*.parquet"
    comments_glob: str = "*comments*.parquet"


@dataclass(frozen=True)
class DataQualityConfig:
    min_bronze_rows: int = 1
    max_null_ratio: float = 0.02
    min_score: int = -100_000
    max_score: int = 10_000_000
    freshness_max_age_days: int = 3650


@dataclass(frozen=True)
class MLConfig:
    # Topic modeling (LDA)
    topic_count: int = 12
    topic_max_iterations: int = 30
    vocab_size: int = 20_000
    min_token_length: int = 3
    min_doc_freq: int = 5
    # Engagement classifier
    engagement_percentile: float = 0.90
    cv_folds: int = 3
    random_seed: int = 42
    # Registry
    registered_topic_model: str = "reddit_topic_lda"
    registered_sentiment_model: str = "reddit_sentiment_clf"
    registered_engagement_model: str = "reddit_engagement_clf"


@dataclass(frozen=True)
class PipelineConfig:
    env: str
    catalog: str
    bronze_schema: str
    silver_schema: str
    gold_schema: str
    volume: str
    raw_landing_path: str
    sql_warehouse_id: str
    sources: list[SourceSpec]
    dq: DataQualityConfig = field(default_factory=DataQualityConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    # -- Fully-qualified name helpers ------------------------------------ #
    def bronze(self, table: str) -> str:
        return f"{self.catalog}.{self.bronze_schema}.{table}"

    def silver(self, table: str) -> str:
        return f"{self.catalog}.{self.silver_schema}.{table}"

    def gold(self, table: str) -> str:
        return f"{self.catalog}.{self.gold_schema}.{table}"

    def model_name(self, short: str) -> str:
        """Unity Catalog model name: ``catalog.gold_schema.<model>``."""
        return f"{self.catalog}.{self.gold_schema}.{short}"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(env: str | None = None, path: str | os.PathLike[str] | None = None) -> PipelineConfig:
    """Load and resolve configuration for ``env``.

    Resolution order: ``defaults`` block <- ``environments.<env>`` block.
    Environment may also be supplied via the ``LAKEHOUSE_ENV`` variable.
    """
    env = env or os.environ.get("LAKEHOUSE_ENV", "dev")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    defaults = raw.get("defaults", {})
    env_block = raw.get("environments", {}).get(env)
    if env_block is None:
        raise KeyError(f"Environment '{env}' not found in {cfg_path}")

    merged = _deep_merge(defaults, env_block)

    sources = [SourceSpec(**s) for s in merged.get("sources", [])]
    dq = DataQualityConfig(**merged.get("data_quality", {}))
    ml = MLConfig(**merged.get("ml", {}))

    return PipelineConfig(
        env=env,
        catalog=merged["catalog"],
        bronze_schema=merged.get("bronze_schema", "bronze"),
        silver_schema=merged.get("silver_schema", "silver"),
        gold_schema=merged.get("gold_schema", "gold"),
        volume=merged.get("volume", "raw"),
        raw_landing_path=merged["raw_landing_path"],
        sql_warehouse_id=merged.get("sql_warehouse_id", ""),
        sources=sources,
        dq=dq,
        ml=ml,
    )
