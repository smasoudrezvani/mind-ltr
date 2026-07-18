"""Feature engineering for MIND — built together in Step 5.

I mirror the three feature groups from the ByteByteGo diagrams
(user / item / user-item interaction — the interaction group is the one
that usually carries the model):

USER features
  [ ] history length (bucketized), days since first activity
  [ ] category-preference vector: normalized counts of categories in history
  [ ] mean embedding of history article texts ("user taste vector")

ITEM (news) features
  [ ] category, subcategory (one-hot / target-encoded)
  [ ] title+abstract sentence-transformer embedding (precomputed once)
  [ ] popularity: rolling CTR and click count (computed with a TIME LAG —
      only from data before the impression time, or we leak the future)

USER-ITEM interaction features (the money features)
  [ ] cosine(user taste vector, item embedding)   <- the "description
      similarity" feature from Fig 7.10, transplanted to news
  [ ] user's historical CTR on this item's category
  [ ] recency match: item age at impression time

Rules we enforce here (and will test):
  * Every feature must be computable from data strictly BEFORE the
    impression timestamp (no leakage).
  * Fit encoders/scalers on train only; transform dev with them.
"""

from __future__ import annotations

# TODO(step 5): implement alongside docs/LEARNING_PATH.md
