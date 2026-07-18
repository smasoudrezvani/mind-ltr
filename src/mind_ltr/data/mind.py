"""Loaders for the MIND (Microsoft News) dataset.

Files (after scripts/download_mind.py):

  data/raw/MINDsmall_train/news.tsv       one row per news article
  data/raw/MINDsmall_train/behaviors.tsv  one row per IMPRESSION

behaviors.tsv columns (tab-separated, no header):
  impression_id, user_id, time, history, impressions
  - history:     space-separated news IDs the user clicked BEFORE this
                 impression (their reading history)
  - impressions: space-separated "N12345-1" / "N54321-0" tokens:
                 the articles SHOWN in this impression, suffixed with
                 1 = clicked, 0 = shown-but-not-clicked.

This is gold for LTR: the "-0" items are REAL negatives (the user saw them
and passed), so we don't have to invent a negative-sampling scheme. The
token order within `impressions` is the display order, which we use as the
position for bias correction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]
BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(split_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(split_dir / "news.tsv", sep="\t", names=NEWS_COLUMNS)
    df["abstract"] = df["abstract"].fillna("")
    df["text"] = df["title"] + ". " + df["abstract"]
    return df


def load_behaviors(split_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(split_dir / "behaviors.tsv", sep="\t", names=BEHAVIORS_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["history"] = df["history"].fillna("").str.split()
    return df


def explode_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """One row per (impression, shown item): the LTR training table skeleton.

    Adds `position` (1-indexed display order) and `clicked` (label).
    """
    rows = []
    for imp_id, user_id, time, history, impressions in behaviors[
        ["impression_id", "user_id", "time", "history", "impressions"]
    ].itertuples(index=False):
        for pos, token in enumerate(impressions.split(), start=1):
            news_id, clicked = token.rsplit("-", 1)
            rows.append((imp_id, user_id, time, news_id, pos, int(clicked)))
    return pd.DataFrame(
        rows,
        columns=["impression_id", "user_id", "time", "news_id", "position", "clicked"],
    )
