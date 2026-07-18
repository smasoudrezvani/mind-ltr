"""Step 1 training loop: pointwise vs LambdaMART vs LambdaMART + IPW.

Runs entirely on synthetic data (no download needed):

    python -m mind_ltr.ranking.train_lambdarank

What to look at in the output
-----------------------------
We evaluate every model against the HIDDEN TRUE RELEVANCE (only possible
because the data is synthetic). Expected pattern:

  1. lambdarank + IPW clearly beats both click-trained models: the synthetic
     legacy ranker over-exposes a weakly-relevant "popularity" feature, so
     raw clicks make that feature look predictive; IPW undoes the confound.
  2. Plain lambdarank vs pointwise is roughly a TIE here. Honest finding:
     with BINARY labels, a pointwise GBDT is a strong baseline; lambdarank's
     advantage appears mainly with graded relevance (label_gain over
     multiple levels) and top-heavy cutoffs. Great interview nuance —
     "when is pointwise good enough?" has a real answer: often, for binary
     click data. We'll revisit with graded (watch-time-like) labels.

When we move to MIND (step 4), true relevance is unknown, so we'll evaluate
on held-out clicks instead — and we'll discuss why that's an imperfect but
standard compromise.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from mind_ltr.ranking.metrics import evaluate_groups
from mind_ltr.ranking.synthetic import make_impressions
from mind_ltr.rerank.position_bias import ipw_sample_weights


def split_by_impression(data, train_frac: float = 0.8):
    """Split whole impressions (never split rows of one impression!)."""
    n_train_groups = int(len(data.group_sizes) * train_frac)
    n_train_rows = int(np.sum(data.group_sizes[:n_train_groups]))
    tr = slice(0, n_train_rows)
    te = slice(n_train_rows, None)
    return tr, te, n_train_groups


def main() -> None:
    data = make_impressions(n_impressions=3000, items_per_impression=50)
    tr, te, n_tr_groups = split_by_impression(data)

    X_tr, X_te = data.X[tr], data.X[te]
    y_tr = data.clicks[tr]                      # we train on BIASED clicks
    y_te_true = data.true_relevance[te]         # we evaluate on TRUTH
    groups_tr = data.group_sizes[:n_tr_groups]
    groups_te = data.group_sizes[n_tr_groups:]

    results = {}

    # ------------------------------------------------------------------
    # Model 1: pointwise baseline (binary classification, ignores groups)
    # ------------------------------------------------------------------
    pointwise = lgb.LGBMClassifier(
        objective="binary", n_estimators=200, learning_rate=0.05, verbose=-1
    )
    pointwise.fit(X_tr, y_tr)
    scores = pointwise.predict_proba(X_te)[:, 1]
    results["pointwise (binary logloss)"] = evaluate_groups(
        y_te_true, scores, groups_te
    )

    # ------------------------------------------------------------------
    # Model 2: LambdaMART (listwise-flavored pairwise, optimizes NDCG)
    # ------------------------------------------------------------------
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        label_gain=[0, 1],          # gains for labels 0 and 1
        verbose=-1,
    )
    ranker.fit(X_tr, y_tr, group=groups_tr)
    scores = ranker.predict(X_te)
    results["lambdarank"] = evaluate_groups(y_te_true, scores, groups_te)

    # ------------------------------------------------------------------
    # Model 3: LambdaMART + IPW position-bias correction
    # ------------------------------------------------------------------
    weights = ipw_sample_weights(y_tr, data.positions[tr], eta=1.0)
    ranker_ipw = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        label_gain=[0, 1],
        verbose=-1,
    )
    ranker_ipw.fit(X_tr, y_tr, group=groups_tr, sample_weight=weights)
    scores = ranker_ipw.predict(X_te)
    results["lambdarank + IPW"] = evaluate_groups(y_te_true, scores, groups_te)

    # ------------------------------------------------------------------
    print("\nEvaluated against HIDDEN true relevance (synthetic ground truth):\n")
    header = f"{'model':<28} {'ndcg@10':>8} {'mrr':>8} {'recall@10':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<28} {m['ndcg@10']:>8.4f} {m['mrr']:>8.4f} {m['recall@10']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
