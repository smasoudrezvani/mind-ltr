# mind-ltr — Learning-to-Rank news recommender on MIND

An end-to-end recommendation system built around a **LambdaMART ranker**,
with **position-bias correction (IPW)** and **MMR diversity re-ranking**,
trained on the [MIND](https://msnews.github.io/) news dataset.

Built as a deep-dive into learning-to-rank: every component is implemented
with the goal of being able to explain it, not just run it.

## Architecture

```
                        ┌────────────────────────────────────────────┐
 news corpus ──────►    │  CANDIDATE GENERATION                      │
 (~100k items)          │  BM25 (lexical) ∪ FAISS ANN (semantic)     │ ~200 candidates
                        └────────────────────┬───────────────────────┘
                                             ▼
                        ┌────────────────────────────────────────────┐
 user history ──────►   │  RANKING — LightGBM LambdaMART             │
 item features          │  user / item / user-item features          │ ~top 50
 (feature store lite)   │  trained with IPW position-bias weights    │
                        └────────────────────┬───────────────────────┘
                                             ▼
                        ┌────────────────────────────────────────────┐
                        │  RE-RANKING — MMR diversity                │ top k
                        └────────────────────────────────────────────┘
```

Two-stage design rationale: a heavy model over the full corpus is too slow;
a light model alone ranks poorly. Retrieval optimizes **recall@k** (don't
lose relevant items), ranking optimizes **NDCG** (order them well),
re-ranking handles list-level objectives (diversity) that per-item scores
can't express.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Runs immediately — no download needed (synthetic impression logs):
python -m mind_ltr.ranking.train_lambdarank

pytest                          # metric + bias-correction unit tests

python scripts/download_mind.py # real data (MINDsmall, ~100MB)
```

The synthetic experiment trains three models on **biased clicks** and
evaluates against **hidden true relevance**, demonstrating both core ideas:

| model | trained on | NDCG@10 vs hidden truth |
|---|---|---|
| pointwise (binary logloss) | raw clicks | ~0.755 (inherits popularity confound) |
| LambdaMART (lambdarank) | raw clicks | ~0.745 (ties pointwise — see note) |
| LambdaMART + IPW | reweighted clicks | **~0.802** — debiasing wins clearly |

Honest note: with binary labels, pointwise GBDT is a strong baseline and
lambdarank does not automatically beat it; its edge appears with graded
relevance and top-heavy cutoffs. Documented deliberately — it's a common
interview probe ("when is pointwise good enough?").

## Key design decisions

- **Negatives:** MIND impression logs contain shown-but-not-clicked items —
  real negatives, no sampling heuristics needed.
- **Splits:** temporal (train on earlier days, evaluate on later) — random
  splits leak the future through popularity and history features.
- **Position bias:** clicks confound relevance with exposure. We reweight
  clicked examples by inverse examination propensity (1/pos^eta, clipped).
- **Leakage rule:** every feature is computable strictly before the
  impression timestamp; encoders fit on train only.

## Metrics

- **recall@200** — candidate generation quality
- **NDCG@10, MRR** — ranking quality (per impression, averaged)
- Ablation table: ± user-item interaction features, ± IPW correction

## How this design extends to video

Nothing in the ranking layer is text-specific — swapping the domain means
swapping the *feature preparation* front-end (cf. ByteByteGo Fig 6.7/6.23):

- **Item features:** title/abstract embeddings → title embeddings (BERT) +
  tag embeddings (CBOW, aggregated) + duration + language/video-ID
  embeddings, concatenated exactly as in the video feature-prep diagram.
  Visual content adds frame-level embeddings (e.g. CLIP on sampled frames,
  mean-pooled) as one more concatenated block.
- **User features:** reading history → watch history / impressions /
  search queries, each aggregated from item embeddings the same way our
  "user taste vector" is built.
- **Labels:** click → watch-time-weighted labels (graded relevance instead
  of binary; LambdaMART handles this via `label_gain`), which also
  mitigates clickbait in a way clicks can't.
- **Pipeline:** identical multi-stage shape — candidate generation over
  billions, scoring over thousands, re-ranking over hundreds — with a video
  feature store replacing our parquet "feature store lite".
- **What genuinely changes:** feature freshness matters more (new videos
  cold-start faster), and the re-ranker gains responsibilities (freshness
  boosting, deduplication of near-identical uploads) beyond MMR diversity.

## Repo layout

```
src/mind_ltr/
  data/       MIND parsers (impression logs → LTR training table)
  features/   user / item / user-item feature builders
  retrieval/  BM25 + FAISS candidate generation
  ranking/    metrics (from scratch), synthetic sandbox, LambdaMART training
  rerank/     IPW position-bias weights, MMR diversity
  serving/    FastAPI /recommend
docs/LEARNING_PATH.md   step-by-step curriculum with interview questions
```

## Roadmap

- [x] Step 1–3: metrics, LambdaMART vs pointwise, IPW (synthetic sandbox)
- [ ] Step 4: MIND EDA (position-CTR curve, sparsity)
- [ ] Step 5: feature engineering (leakage-safe)
- [ ] Step 6: train on MIND + ablation table + data-driven eta
- [ ] Step 7: BM25 + FAISS retrieval, recall@200
- [ ] Step 8: FastAPI serving + MMR + Docker
