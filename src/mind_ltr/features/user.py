"""User features for MIND (Step 5, Task 1).

In MIND, `history` is the user's pre-log-window click history, so it is
identical on every impression row of the same user within a split — one
row per user is well-defined and we keep the first occurrence.
"""

from __future__ import annotations

import pandas as pd


def build_user_features(behaviors: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Build per-user features from parsed behaviors.

    Args:
        behaviors: Output of load_behaviors (has user_id and parsed history).
        news: Output of load_news (has news_id and category).

    Returns:
        One row per user_id: `history_length` plus one `cat_share_<category>`
        column per category observed in user histories. Shares are the
        fraction of the user's history clicks in that category (rows sum to
        1.0); zero-history users get history_length 0 and all shares 0.0.
    """
    users = behaviors.drop_duplicates(subset="user_id")[["user_id", "history"]].copy()
    users["history_length"] = users["history"].str.len()

    # One row per (user, clicked article); empty histories drop out here and
    # are restored with zero shares by the left-merge below.
    clicks = users[["user_id", "history"]].explode("history").dropna(subset=["history"])
    clicks = clicks.merge(
        news[["news_id", "category"]], how="left", left_on="history", right_on="news_id"
    )

    counts = clicks.groupby(["user_id", "category"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [f"cat_share_{c}" for c in shares.columns]

    # Merge by user_id, never by positional index alignment.
    return (
        users[["user_id", "history_length"]]
        .merge(shares.reset_index(), how="left", on="user_id")
        .fillna(0.0)
    )
