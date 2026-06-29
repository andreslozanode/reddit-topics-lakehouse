"""Land public Reddit datasets from Hugging Face into a raw landing zone.

Runs on the Spark driver (or any node with network access). Uses
``huggingface_hub`` to snapshot the parquet files into ``<landing>/<topic>/``.
On Databricks the landing path is typically a Unity Catalog Volume, e.g.
``/Volumes/<catalog>/bronze/raw/``.
"""

from __future__ import annotations

import os

from reddit_lakehouse.common.logging import get_logger
from reddit_lakehouse.config import SourceSpec

logger = get_logger(__name__)


def download_source(source: SourceSpec, landing_root: str) -> str:
    """Download a single dataset's parquet files; return its local directory."""
    from huggingface_hub import snapshot_download

    target = os.path.join(landing_root, source.topic)
    os.makedirs(target, exist_ok=True)
    logger.info("Downloading %s -> %s", source.repo_id, target)
    snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["*.parquet", "*.csv", "*.csv.gz"],
    )
    return target


def download_all(sources: list[SourceSpec], landing_root: str) -> dict[str, str]:
    """Download every configured source; return ``{topic: local_dir}``."""
    return {s.topic: download_source(s, landing_root) for s in sources}
