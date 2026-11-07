"""Exploratory analysis behind Phase A1 (docs/SCIENCE.md, section 8).

    uv run python scripts/explore_popularity.py [--data data/raw]

Reports MINDsmall date ranges, cold-start share and article lifetimes, then
compares popularity scores with and without leakage by their per-impression
AUC on the last train day.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mind_ltr.data.mind import explode_impressions, load_behaviors

log = logging.getLogger(__name__)

PRIOR_STRENGTH = 100.0  # m, in pseudo-impressions


def mean_impression_auc(df: pd.DataFrame, col: str) -> float:
    """MIND-style AUC: computed per impression, averaged over impressions.

    Args:
        df: Rows with `impression_id`, `clicked` and the score column.
        col: Name of the score column.

    Returns:
        Mean AUC over impressions that contain both a click and a non-click.
    """
    aucs = []
    for _, g in df.groupby("impression_id"):
        y = g["clicked"].to_numpy()
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, g[col].to_numpy()))
    return float(np.mean(aucs))


def describe_splits(tr: pd.DataFrame, dv: pd.DataFrame) -> None:
    """Log date ranges, sizes, CTR, cold-start share and article lifetimes.

    Args:
        tr: Exploded train impressions.
        dv: Exploded dev impressions.
    """
    for name, d in [("train", tr), ("dev", dv)]:
        log.info(
            "%s: %s .. %s | impressions=%d rows=%d CTR=%.4f",
            name,
            d.time.min(),
            d.time.max(),
            d.impression_id.nunique(),
            len(d),
            d.clicked.mean(),
        )
    cold = ~dv.news_id.isin(set(tr.news_id.unique()))
    log.info(
        "dev rows with article unseen in train: %.1f%% | of dev clicks: %.1f%%",
        100 * cold.mean(),
        100 * cold[dv.clicked == 1].mean(),
    )
    life = tr.groupby("news_id").time.agg(["min", "max"])
    hours = (life["max"] - life["min"]).dt.total_seconds() / 3600
    log.info(
        "article lifetime in train (hours): median=%.1f p75=%.1f p90=%.1f",
        hours.median(),
        hours.quantile(0.75),
        hours.quantile(0.9),
    )


def compare_popularity_scores(tr: pd.DataFrame) -> None:
    """Log per-impression AUC of leaky vs time-lagged popularity on the last train day.

    Args:
        tr: Exploded train impressions.
    """
    tr = tr.copy()
    tr["hour"] = tr.time.dt.floor("h")
    tr["day"] = tr.time.dt.floor("D")
    last_day = tr["day"].max()
    eval_df = tr[tr["day"] == last_day].copy()

    p = tr.clicked.mean()
    a, b = PRIOR_STRENGTH * p, PRIOR_STRENGTH * (1 - p)

    # Leaky: CTR over the whole train week, including the row itself.
    full = tr.groupby("news_id").clicked.agg(["sum", "size"])
    eval_df["ctr_leaky"] = eval_df.news_id.map(full["sum"] / full["size"])

    # Daily lag: counts from days strictly before the impression's day.
    before = tr[tr["day"] < last_day].groupby("news_id").clicked.agg(["sum", "size"])
    clk = eval_df.news_id.map(before["sum"]).fillna(0)
    imp = eval_df.news_id.map(before["size"]).fillna(0)
    eval_df["ctr_daylag_raw"] = np.where(imp > 0, clk / imp.replace(0, 1), p)
    eval_df["ctr_daylag_smooth"] = (clk + a) / (imp + a + b)

    # Hourly lag: exclusive cumulative counts up to the impression's hour.
    hourly = tr.groupby(["news_id", "hour"]).clicked.agg(["sum", "size"]).reset_index()
    hourly = hourly.sort_values(["news_id", "hour"])
    hourly["cum_clk"] = hourly.groupby("news_id")["sum"].cumsum() - hourly["sum"]
    hourly["cum_imp"] = hourly.groupby("news_id")["size"].cumsum() - hourly["size"]
    eval_df = eval_df.merge(
        hourly[["news_id", "hour", "cum_clk", "cum_imp"]], on=["news_id", "hour"], how="left"
    )
    eval_df["ctr_hourlag_smooth"] = (eval_df.cum_clk + a) / (eval_df.cum_imp + a + b)
    eval_df["imp_hourlag_log"] = np.log1p(eval_df.cum_imp)

    log.info("per-impression AUC on %s:", last_day.date())
    for col in [
        "ctr_leaky",
        "ctr_daylag_raw",
        "ctr_daylag_smooth",
        "ctr_hourlag_smooth",
        "imp_hourlag_log",
    ]:
        log.info("  %-20s %.4f", col, mean_impression_auc(eval_df, col))
    log.info("rows with no history before that day:  %.1f%%", 100 * (imp == 0).mean())
    log.info("rows with no history before that hour: %.1f%%", 100 * (eval_df.cum_imp == 0).mean())


def main() -> None:
    """Run the exploratory analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tr = explode_impressions(load_behaviors(args.data / "MINDsmall_train"))
    dv = explode_impressions(load_behaviors(args.data / "MINDsmall_dev"))
    describe_splits(tr, dv)
    compare_popularity_scores(tr)


if __name__ == "__main__":
    main()
