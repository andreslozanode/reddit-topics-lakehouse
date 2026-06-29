"""Shared CLI scaffolding for pipeline entrypoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from reddit_lakehouse.config import PipelineConfig, load_config


def parse_env() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None, help="dev | staging | prod")
    parser.add_argument("--config", default=None, help="Path to pipeline.yml")
    return parser.parse_known_args()[0]


def resolve_config() -> PipelineConfig:
    args = parse_env()
    return load_config(env=args.env, path=args.config)


def batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
