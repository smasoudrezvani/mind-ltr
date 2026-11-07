"""Tests for time-lagged item popularity (docs/SCIENCE.md section 8).

Two kinds of tests:
  * a hand-computed fixture: every expected value is worked out by hand;
  * leakage properties: changing the present or the future must never
    change a feature, checked on a random log.
"""

import numpy as np
import pandas as pd
import pytest

from mind_ltr.features.item import build_item_popularity

FEATURES = ["item_ctr_smooth", "item_log_impressions", "item_age_hours"]


def _fixture() -> pd.DataFrame:
    """Two items over three hours, with a non-default index."""
    return pd.DataFrame(
        {
            "news_id": ["A", "A", "A", "B", "A", "B"],
            "time": pd.to_datetime(
                [
                    "2019-11-09 10:05",
                    "2019-11-09 10:40",
                    "2019-11-09 11:10",
                    "2019-11-09 11:20",
                    "2019-11-09 12:35",
                    "2019-11-09 12:50",
                ]
            ),
            "clicked": [1, 0, 0, 1, 1, 0],
        },
        index=[100, 101, 102, 103, 104, 105],
    )


def test_hand_computed_values():
    """p = 0.1, m = 10 -> alpha = 1, so smoothed CTR = (C + 1) / (N + 10).

    A @10h: nothing before          -> C=0, N=0 -> 1/10
    A @11h: hour 10 had 2 imps, 1 click -> C=1, N=2 -> 2/12
    B @11h: B never seen before      -> 1/10
    A @12h: hours 10-11: 3 imps, 1 click -> C=1, N=3 -> 2/13
    B @12h: hour 11: 1 imp, 1 click  -> C=1, N=1 -> 2/11
    """
    out = build_item_popularity(_fixture(), prior_ctr=0.1, prior_strength=10.0)

    expected_ctr = [1 / 10, 1 / 10, 2 / 12, 1 / 10, 2 / 13, 2 / 11]
    expected_n = [0, 0, 2, 0, 3, 1]
    # A first seen 10:05, B first seen 11:20.
    expected_age = [0.0, 35 / 60, 65 / 60, 0.0, 150 / 60, 90 / 60]

    np.testing.assert_allclose(out["item_ctr_smooth"], expected_ctr)
    np.testing.assert_allclose(out["item_log_impressions"], np.log1p(expected_n))
    np.testing.assert_allclose(out["item_age_hours"], expected_age)


def test_output_aligned_to_input_index_and_input_not_mutated():
    rows = _fixture()
    before = rows.copy()
    out = build_item_popularity(rows, prior_ctr=0.1)
    assert list(out.index) == list(rows.index)
    pd.testing.assert_frame_equal(rows, before)


def _random_log(seed: int = 0, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-11-09")
    minutes = np.sort(rng.integers(0, 48 * 60, size=n))
    return pd.DataFrame(
        {
            "news_id": rng.choice([f"N{i}" for i in range(30)], size=n),
            "time": start + pd.to_timedelta(minutes, unit="min"),
            "clicked": rng.integers(0, 2, size=n),
        }
    )


def test_future_and_present_labels_do_not_change_past_features():
    """Randomize every label from the cutoff hour on; earlier rows must not change."""
    rows = _random_log()
    cutoff = pd.Timestamp("2019-11-10 05:00")
    base = build_item_popularity(rows, prior_ctr=0.3)

    perturbed = rows.copy()
    later = perturbed["time"] >= cutoff
    rng = np.random.default_rng(1)
    perturbed.loc[later, "clicked"] = rng.integers(0, 2, size=int(later.sum()))
    out = build_item_popularity(perturbed, prior_ctr=0.3)

    earlier = rows["time"] < cutoff
    pd.testing.assert_frame_equal(out[earlier], base[earlier])


def test_appending_future_rows_does_not_change_past_features():
    rows = _random_log()
    base = build_item_popularity(rows, prior_ctr=0.3)

    future = _random_log(seed=2, n=500)
    future["time"] = future["time"] + pd.Timedelta(days=5)
    future.index = future.index + len(rows)
    out = build_item_popularity(pd.concat([rows, future]), prior_ctr=0.3)

    pd.testing.assert_frame_equal(out.loc[rows.index], base)


def test_own_label_does_not_change_own_bucket_features():
    """Flipping one row's label leaves every row in its hour unchanged."""
    rows = _random_log()
    base = build_item_popularity(rows, prior_ctr=0.3)
    for i in [0, 500, 1999]:
        flipped = rows.copy()
        flipped.loc[i, "clicked"] = 1 - flipped.loc[i, "clicked"]
        out = build_item_popularity(flipped, prior_ctr=0.3)
        same_hour = rows["time"].dt.floor("h") == rows.loc[i, "time"].floor("h")
        pd.testing.assert_frame_equal(out[same_hour], base[same_hour])


@pytest.mark.parametrize(
    "kwargs",
    [{"prior_ctr": 0.0}, {"prior_ctr": 1.0}, {"prior_ctr": 0.1, "prior_strength": 0.0}],
)
def test_invalid_prior_raises(kwargs):
    with pytest.raises(ValueError):
        build_item_popularity(_fixture(), **kwargs)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="clicked"):
        build_item_popularity(_fixture().drop(columns="clicked"), prior_ctr=0.1)
