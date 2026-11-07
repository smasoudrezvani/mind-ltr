"""Step 4 EDA on MIND.

    uv run python scripts/eda_mind.py [--split data/raw/MINDsmall_train]

Prints a stats report and writes plots to reports/.

The centerpiece is the POSITION-BIAS TEST. We do not assume that the token
order inside the `impressions` column equals display order — we let the
data testify. If CTR is flat across token positions, order carries no
position information and IPW on it is meaningless. If CTR decays, order is
meaningful, and the log-log slope gives a data-driven eta for
propensity = 1/pos^eta (replacing the eta=1.0 we assumed in Step 3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mind_ltr.data.mind import explode_impressions, load_behaviors, load_news

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def position_bias_analysis(rows: pd.DataFrame, max_pos: int = 30) -> float:
    """CTR by token position + power-law fit. Returns fitted eta."""
    by_pos = (
        rows[rows.position <= max_pos]
        .groupby("position")["clicked"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "ctr"})
    )
    # Fit log(ctr) = c - eta * log(pos) by least squares, weighted by count.
    valid = by_pos[by_pos.ctr > 0]
    x = np.log(valid.index.to_numpy(dtype=float))
    y = np.log(valid["ctr"].to_numpy())
    w = valid["count"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1, w=np.sqrt(w))
    eta = -slope

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(by_pos.index, by_pos.ctr, marker="o", lw=1)
    ax[0].set(xlabel="token position", ylabel="CTR", title="CTR vs position")
    ax[1].plot(x, y, "o", label="observed")
    ax[1].plot(x, intercept + slope * x, "-", label=f"fit: eta={eta:.2f}")
    ax[1].set(xlabel="log(position)", ylabel="log(CTR)", title="power-law fit")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "position_bias.png", dpi=150)
    plt.close(fig)
    return float(eta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/raw/MINDsmall_train")
    args = parser.parse_args()
    split = Path(args.split)
    REPORTS.mkdir(exist_ok=True)

    news = load_news(split)
    behaviors = load_behaviors(split)
    rows = explode_impressions(behaviors)

    n_shown = len(rows)
    n_clicks = int(rows.clicked.sum())
    per_imp = rows.groupby("impression_id")["clicked"].agg(["size", "sum"])

    print(f"\n=== {split.name} ===")
    print(f"impressions           {len(behaviors):>10,}")
    print(f"users                 {behaviors.user_id.nunique():>10,}")
    print(f"news articles         {len(news):>10,}")
    print(f"shown items (rows)    {n_shown:>10,}")
    print(f"overall CTR           {n_clicks / n_shown:>10.4f}")
    print(
        f"items shown / impr    {per_imp['size'].mean():>10.1f} (median {per_imp['size'].median():.0f})"
    )
    print(f"clicks / impr         {per_imp['sum'].mean():>10.2f}")
    print(f"date range            {behaviors.time.min()}  ->  {behaviors.time.max()}")

    # impressions per day -> informs the temporal split boundary
    per_day = behaviors.set_index("time").resample("D").size()
    print("\nimpressions per day:")
    for day, cnt in per_day.items():
        print(f"  {day.date()}  {cnt:>8,}")

    # history length -> how much signal user-taste features will have
    hist_len = behaviors.history.str.len()
    print(
        f"\nhistory length: mean {hist_len.mean():.1f}, median {hist_len.median():.0f}, "
        f"zero-history users: {(hist_len == 0).mean():.1%} of impressions"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_len.clip(upper=100).hist(bins=50, ax=ax)
    ax.set(
        xlabel="history length (clipped at 100)", ylabel="impressions", title="User history length"
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "history_length.png", dpi=150)
    plt.close(fig)

    # category skew: shown vs clicked share -> which categories over/under-perform
    cat = rows.merge(news[["news_id", "category"]], on="news_id", how="left")
    shown_share = cat.category.value_counts(normalize=True)
    click_share = cat[cat.clicked == 1].category.value_counts(normalize=True)
    skew = (
        pd.DataFrame({"shown": shown_share, "clicked": click_share})
        .fillna(0.0)
        .assign(lift=lambda d: d.clicked / d.shown.replace(0, np.nan))
        .sort_values("shown", ascending=False)
    )
    print("\ncategory share (top 10 by exposure):")
    print(skew.head(10).round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    skew.head(10)[["shown", "clicked"]].plot.bar(ax=ax)
    ax.set(title="Category share: shown vs clicked", ylabel="share")
    fig.tight_layout()
    fig.savefig(REPORTS / "category_share.png", dpi=150)
    plt.close(fig)

    # the centerpiece
    eta = position_bias_analysis(rows)
    print(f"\nPOSITION BIAS: fitted eta = {eta:.3f}   (plot: reports/position_bias.png)")
    if eta < 0.05:
        print("  -> CTR is ~flat across token positions: order carries little/no")
        print("     position information; IPW on token order is NOT justified.")
    else:
        print("  -> CTR decays with token position: order is informative; use this")
        print("     eta in ipw_sample_weights instead of the assumed 1.0.")


if __name__ == "__main__":
    main()
