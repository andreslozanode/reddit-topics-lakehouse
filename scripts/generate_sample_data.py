"""Generate synthetic Reddit parquet matching the SocialGrep schema.

Used for local development, CI and unit tests so no network/HF download is
required. Produces ``<out>/<topic>/posts.parquet`` and ``comments.parquet``.

Usage:
    python scripts/generate_sample_data.py --out ./.data --topic sample --rows 2000
"""
from __future__ import annotations

import argparse
import os
import random
import time

import pandas as pd

SUBREDDITS = ["datascience", "MachineLearning", "dataengineering", "Python", "aws", "databricks"]
WORDS = (
    "spark delta lakehouse pipeline model feature cluster kafka stream batch "
    "warehouse dashboard etl python sql cloud data governance catalog vector "
    "training inference latency throughput partition optimize schema quality"
).split()
SENTIMENTS = ["0.8", "0.2", "-0.1", "-0.6", "0.0", "0.45"]


def _text(n_words: int) -> str:
    return " ".join(random.choice(WORDS) for _ in range(n_words))


def _base_row(i: int, kind: str) -> dict:
    sub = random.choice(SUBREDDITS)
    created = int(time.time()) - random.randint(0, 60 * 60 * 24 * 365)
    return {
        "type": kind,
        "id": f"{kind[0]}{i:08x}",
        "subreddit.id": f"sr_{sub}",
        "subreddit.name": sub,
        "subreddit.nsfw": False,
        "created_utc": created,
        "permalink": f"/r/{sub}/{kind}/{i}",
        "score": int(random.lognormvariate(2.0, 1.4)) - 3,
    }


def make_posts(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        r = _base_row(i, "post")
        r["title"] = _text(random.randint(4, 12)).capitalize() + ("?" if random.random() < 0.3 else "")
        r["selftext"] = _text(random.randint(0, 60)) if random.random() < 0.7 else ""
        r["domain"] = random.choice(["self.reddit", "github.com", "medium.com", "youtube.com"])
        r["url"] = f"https://example.com/{i}" if random.random() < 0.4 else None
        rows.append(r)
    return pd.DataFrame(rows)


def make_comments(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        r = _base_row(i, "comment")
        r["body"] = _text(random.randint(3, 40))
        r["sentiment"] = random.choice(SENTIMENTS)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./.data")
    parser.add_argument("--topic", default="sample")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = os.path.join(args.out, args.topic)
    os.makedirs(out_dir, exist_ok=True)

    make_posts(args.rows).to_parquet(os.path.join(out_dir, "posts.parquet"), index=False)
    make_comments(args.rows * 2).to_parquet(os.path.join(out_dir, "comments.parquet"), index=False)
    print(f"Wrote sample data to {out_dir} ({args.rows} posts / {args.rows * 2} comments)")


if __name__ == "__main__":
    main()
