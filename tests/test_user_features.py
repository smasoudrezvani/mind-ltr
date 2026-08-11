"""Hand-computed fixture tests for build_user_features (Task 1).

Same philosophy as test_lambdas.py: every expected value is known in advance.
"""

import pandas as pd

from mind_ltr.features.user import build_user_features


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    # U1 appears in TWO impressions with the same history (the real MIND
    # shape) and a non-default index — this is what catches accidental
    # positional-index alignment and duplicate-user bugs.
    behaviors = pd.DataFrame(
        {
            "user_id": ["U1", "U2", "U1", "U3"],
            "history": [
                ["n1", "n2", "n3"],
                ["n4"],
                ["n1", "n2", "n3"],
                [],
            ],
        },
        index=[10, 11, 12, 13],
    )
    news = pd.DataFrame(
        {
            "news_id": ["n1", "n2", "n3", "n4"],
            "category": ["sports", "sports", "news", "sports"],
        }
    )
    return behaviors, news


def test_history_length_and_category_shares():
    """Hand-computed expectations:

    U1: history n1,n2,n3 = 2 sports + 1 news -> length 3, shares 2/3, 1/3
    U2: history n4       = 1 sports          -> length 1, share sports 1.0
    U3: empty history                        -> length 0, all shares 0.0
    """
    behaviors, news = _fixture()
    out = build_user_features(behaviors, news)

    expected = pd.DataFrame(
        {
            "user_id": ["U1", "U2", "U3"],
            "history_length": [3, 1, 0],
            "cat_share_news": [1 / 3, 0.0, 0.0],
            "cat_share_sports": [2 / 3, 1.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected, check_like=True)


def test_shares_sum_to_one_for_users_with_history():
    behaviors, news = _fixture()
    out = build_user_features(behaviors, news)
    share_cols = [c for c in out.columns if c.startswith("cat_share_")]
    with_history = out[out["history_length"] > 0]
    assert (with_history[share_cols].sum(axis=1) - 1.0).abs().max() < 1e-12


def test_input_frame_is_not_mutated():
    behaviors, news = _fixture()
    n_rows_before = len(behaviors)
    build_user_features(behaviors, news)
    assert len(behaviors) == n_rows_before
