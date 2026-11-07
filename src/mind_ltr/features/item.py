"""Time-lagged item popularity features (Phase A1, docs/SCIENCE.md section 8).

Every feature for an impression in time bucket b uses only events from
buckets strictly before b (an exclusive cumulative sum), so neither the
row's own label nor anything later can leak in. Run it over one continuous
timeline (train + dev logs together) so features are computed the same way
for every split, like an online system with an hourly feature job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("news_id", "time", "clicked")


def build_item_popularity(
    rows: pd.DataFrame,
    prior_ctr: float,
    prior_strength: float = 100.0,
    freq: str = "h",
) -> pd.DataFrame:
    """Compute leakage-safe popularity features for each impression row.

    Args:
        rows: One row per (impression, shown item), as returned by
            `explode_impressions`; needs `news_id`, `time` and `clicked`.
        prior_ctr: Global CTR p used as the prior mean. Compute it on the
            training window only; computing it over test rows would leak
            test labels into every feature.
        prior_strength: Prior weight m in pseudo-impressions
            (alpha = m * p, beta = m * (1 - p)).
        freq: Time bucket size as a pandas offset alias ("h" = hourly).

    Returns:
        DataFrame with the same index as `rows` and columns:
        `item_ctr_smooth` ((C + alpha) / (N + m) from earlier buckets),
        `item_log_impressions` (log(1 + N)) and `item_age_hours` (hours
        since the item was first seen in `rows`).

    Raises:
        ValueError: If a required column is missing or the prior is invalid.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"rows is missing columns: {missing}")
    if not 0.0 < prior_ctr < 1.0:
        raise ValueError(f"prior_ctr must be in (0, 1), got {prior_ctr}")
    if prior_strength <= 0.0:
        raise ValueError(f"prior_strength must be > 0, got {prior_strength}")

    bucket = rows["time"].dt.floor(freq)

    # Clicks and impressions per (item, bucket), then an EXCLUSIVE running
    # total: cumsum includes the current bucket, so subtract it back out.
    per_bucket = (
        pd.DataFrame({"news_id": rows["news_id"], "bucket": bucket, "clicked": rows["clicked"]})
        .groupby(["news_id", "bucket"], sort=True)["clicked"]
        .agg(clicks="sum", imps="size")
    )
    by_item = per_bucket.groupby(level="news_id")
    per_bucket["clicks_before"] = by_item["clicks"].cumsum() - per_bucket["clicks"]
    per_bucket["imps_before"] = by_item["imps"].cumsum() - per_bucket["imps"]

    # Align back to the input rows by (item, bucket), never by position.
    keys = pd.MultiIndex.from_arrays([rows["news_id"], bucket], names=["news_id", "bucket"])
    before = per_bucket.reindex(keys)
    clicks_before = before["clicks_before"].to_numpy(dtype=float)
    imps_before = before["imps_before"].to_numpy(dtype=float)

    alpha = prior_strength * prior_ctr
    # The first appearance of an item is never later than any of its rows,
    # so age uses no future information.
    first_seen = rows.groupby("news_id")["time"].transform("min")

    return pd.DataFrame(
        {
            "item_ctr_smooth": (clicks_before + alpha) / (imps_before + prior_strength),
            "item_log_impressions": np.log1p(imps_before),
            "item_age_hours": (rows["time"] - first_seen).dt.total_seconds() / 3600.0,
        },
        index=rows.index,
    )
