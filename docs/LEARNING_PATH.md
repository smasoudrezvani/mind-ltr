# Learning Path

Work through the steps in order. Each step has: theory to understand, code to
write or run, and the interview questions it prepares you to answer. Don't
move on until you can answer the interview questions out loud without notes.

## Step 1 — Ranking metrics & the LTR problem framing  ✅ (scaffolded)
- **Theory:** why ranking ≠ classification; what a "query group"/impression
  is; DCG's gain and discount; why NDCG normalizes; MRR vs NDCG vs recall@k
  and which stage each measures.
- **Code:** read `src/mind_ltr/ranking/metrics.py` line by line, then run
  `pytest`. Exercise: add `map_at_k` (mean average precision) yourself and a
  test for it.
- **Interview Qs:** "Why NDCG over accuracy for ranking?" "What does the
  log2 discount mean?" "Your NDCG went up but recall went down — what
  happened?"

## Step 2 — Pointwise vs pairwise vs listwise  ✅ (scaffolded)
- **Theory:** pointwise = predict a score per item independently (logloss);
  pairwise = learn from item pairs "A should beat B" (RankNet); listwise-ish =
  weight each pair's gradient by how much swapping the pair changes NDCG
  (LambdaRank → LambdaMART when the learner is gradient-boosted trees).
- **Code:** run `python -m mind_ltr.ranking.train_lambdarank` and study the
  results table. Exercise: change `items_per_impression` and click rarity;
  watch when the gap between pointwise and lambdarank grows or shrinks.
- **Interview Qs:** "Explain LambdaMART to me." "When is pointwise good
  enough?" "What is the `group` parameter in LightGBM and what breaks if you
  get it wrong?"

## Step 3 — Position bias  ✅ (scaffolded)
- **Theory:** examination model P(click)=P(examined|pos)·P(rel); why naive
  click training imitates the legacy ranker; IPW reweighting; propensity
  estimation (randomization, or parametric 1/pos^eta); weight clipping and
  the bias/variance trade-off.
- **Code:** `src/mind_ltr/rerank/position_bias.py`. Exercise: in
  `synthetic.py`, raise `position_bias_eta` to 2.0 and re-run training —
  the IPW gap should widen. Then set eta wrong in IPW (e.g. 0.3) and see
  partial correction.
- **Interview Qs:** "Your offline NDCG on clicks improved but the online A/B
  was flat — one hypothesis?" (position bias in the offline labels).

## Step 4 — Real data: MIND
- **Theory:** impression logs; why "shown-but-not-clicked" negatives beat
  sampled negatives; temporal splits and leakage.
- **Code:** `python scripts/download_mind.py`, then explore with
  `mind_ltr.data.mind`. Write an EDA notebook: clicks per impression,
  position-CTR curve (plot CTR vs display position — you will SEE the bias),
  category distribution, history-length distribution.
- **Interview Qs:** "How would you construct negatives without impression
  logs?" "Why not a random train/test split?"

## Step 5 — Feature engineering
- Follow the checklist in `src/mind_ltr/features/build_features.py`.
  Golden rule: every feature computable strictly before impression time.
- Deliverable: a parquet training table + a features README section.
- **Interview Qs:** "What's your most important feature and why?" "How do
  you prevent leakage in a popularity feature?"

## Step 6 — Train on MIND + ablations
- Train the same three models on real data; evaluate on the dev split by
  held-out clicks. Produce the ablation table (± interaction features,
  ± IPW). Estimate eta from MIND's own position-CTR curve instead of
  assuming 1.0.

## Step 7 — Candidate generation
- BM25 + FAISS union; measure recall@200 separately from ranking quality.
- **Interview Qs:** "How do you evaluate retrieval vs ranking?" "Why two
  channels (lexical + semantic)?"

## Step 8 — Serving + MMR + repo polish
- Wire /recommend in FastAPI; MMR on top; Dockerfile; README as a design
  doc, including the "How this design extends to video" section.
- **Interview Qs:** "Walk me through one request end to end." "What would
  change at 10x scale?"
