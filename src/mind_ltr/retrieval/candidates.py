"""Candidate generation — built together in Step 7.

MIND impressions already come with a shown-items list, so the ranker can be
trained without retrieval. I add this stage anyway because (a) it's how the
real system works (billions -> thousands, Fig 6.23), and (b) "how do you
measure retrieval separately from ranking?" is a standard interview question
(answer: recall@k of the candidate set, before any ranking).

Plan:
  [ ] BM25 over title+abstract (rank_bm25) — lexical channel
  [ ] FAISS ANN index over sentence-transformer embeddings — semantic channel
  [ ] union + dedupe -> ~200 candidates per user
  [ ] evaluate recall@200 against the user's actual clicks
"""

from __future__ import annotations

# TODO(step 7)
