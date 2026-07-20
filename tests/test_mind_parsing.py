"""Parser tests against a tiny hand-written MIND-format fixture.

Same philosophy as the lambda tests: know every expected value in advance.
"""

import pandas as pd
import pytest

from mind_ltr.data.mind import explode_impressions, load_behaviors, load_news

NEWS_TSV = (
    "N1\tsports\tsoccer\tTitle one\tAbstract one\tu1\t[]\t[]\n"
    "N2\tnews\tpolitics\tTitle two\t\tu2\t[]\t[]\n"          # empty abstract
    "N3\tfinance\tmarkets\tTitle three\tAbstract three\tu3\t[]\t[]\n"
)

BEHAVIORS_TSV = (
    # imp_id  user  time                    history      impressions
    "1\tU10\t11/11/2019 9:05:58 AM\tN1 N2\tN3-1 N2-0\n"
    "2\tU10\t11/12/2019 6:11:30 PM\tN1 N2 N3\tN1-0 N2-0 N3-1\n"
    "3\tU20\t11/13/2019 1:00:00 PM\t\tN1-0\n"                 # empty history
)


@pytest.fixture
def split_dir(tmp_path):
    (tmp_path / "news.tsv").write_text(NEWS_TSV)
    (tmp_path / "behaviors.tsv").write_text(BEHAVIORS_TSV)
    return tmp_path


def test_load_news(split_dir):
    news = load_news(split_dir)
    assert len(news) == 3
    assert news.loc[news.news_id == "N2", "abstract"].item() == ""   # NaN -> ""
    assert news.loc[news.news_id == "N1", "text"].item() == "Title one. Abstract one"


def test_load_behaviors(split_dir):
    b = load_behaviors(split_dir)
    assert len(b) == 3
    assert b.loc[2, "history"] == []                    # empty history -> []
    assert b.loc[0, "history"] == ["N1", "N2"]
    assert b["time"].dt.year.eq(2019).all()


def test_explode_impressions(split_dir):
    rows = explode_impressions(load_behaviors(split_dir))
    # 2 + 3 + 1 shown items across the three impressions
    assert len(rows) == 6
    # positions are 1-indexed token order within each impression
    imp2 = rows[rows.impression_id == 2]
    assert imp2.position.tolist() == [1, 2, 3]
    assert imp2.clicked.tolist() == [0, 0, 1]
    # group sizes for LightGBM come straight from this table
    assert rows.groupby("impression_id").size().tolist() == [2, 3, 1]
