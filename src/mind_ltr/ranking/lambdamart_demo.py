"""Step 2 finale: our from-scratch lambdas AS the LightGBM objective.

    python -m mind_ltr.ranking.lambdamart_demo

LightGBM's gradient boosting is objective-agnostic: each round it asks the
objective for (grad, hess) per row and fits a tree to them. Feeding it OUR
lambda computation therefore builds a genuine LambdaMART out of parts we
wrote and hand-verified. If the result tracks the native "lambdarank"
objective, we've demonstrably understood the algorithm.

(Exact equality isn't expected: native lambdarank adds refinements —
lambda normalization, truncation to top pairs, its own sigmoid scaling —
that are performance tweaks, not conceptual differences.)
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from mind_ltr.ranking.lambdas import make_lgb_objective
from mind_ltr.ranking.metrics import evaluate_groups
from mind_ltr.ranking.synthetic import make_impressions


def main() -> None:
    data = make_impressions(n_impressions=1500, items_per_impression=20, seed=7)
    n_tr_groups = int(len(data.group_sizes) * 0.8)
    n_tr_rows = int(np.sum(data.group_sizes[:n_tr_groups]))
    tr, te = slice(0, n_tr_rows), slice(n_tr_rows, None)
    g_tr, g_te = data.group_sizes[:n_tr_groups], data.group_sizes[n_tr_groups:]

    common = dict(learning_rate=0.05, num_leaves=31, verbose=-1)
    n_rounds = 150

    # --- native LambdaMART -------------------------------------------------
    ds_native = lgb.Dataset(data.X[tr], label=data.clicks[tr], group=g_tr)
    native = lgb.train(
        {**common, "objective": "lambdarank", "label_gain": [0, 1]},
        ds_native,
        num_boost_round=n_rounds,
    )

    # --- OUR LambdaMART: same trees, our gradients -------------------------
    ds_ours = lgb.Dataset(data.X[tr], label=data.clicks[tr], group=g_tr, free_raw_data=False)
    ours = lgb.train(
        {**common, "objective": make_lgb_objective(g_tr, sigma=1.0)},
        ds_ours,
        num_boost_round=n_rounds,
    )

    # --- compare against hidden true relevance -----------------------------
    y_true = data.true_relevance[te]
    print("\nSame trees, two gradient sources — evaluated on hidden truth:\n")
    header = f"{'objective':<30} {'ndcg@10':>8} {'mrr':>8}"
    print(header)
    print("-" * len(header))
    for name, model in [("native lambdarank", native), ("our from-scratch lambdas", ours)]:
        m = evaluate_groups(y_true, model.predict(data.X[te]), g_te)
        print(f"{name:<30} {m['ndcg@10']:>8.4f} {m['mrr']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
