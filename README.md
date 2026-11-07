# mind-ltr — Learning-to-Rank news recommender on MIND

A learning-to-rank news recommender on the [MIND](https://msnews.github.io/)
dataset, being built into a two-stage **retrieve → rank → serve** system:
candidate retrieval, a **LambdaMART ranker**, **MMR diversity re-ranking**,
and a FastAPI service. Position-bias correction (IPW) is studied in a
controlled synthetic simulator where the true relevance is known.

Every component is implemented with the goal of being able to explain it,
not just run it. The status table below separates what is built from what
is planned.

## Status

| Component | Status |
|---|---|
| Ranking metrics from scratch (NDCG, MRR, MAP, recall), unit-tested | built |
| LambdaRank gradients from scratch, used as a LightGBM custom objective | built |
| Synthetic click simulator + IPW position-bias experiment | built |
| MIND parsers + EDA | built |
| User features (history length, category shares) | built |
| Item, embedding and user-item features (leakage-safe) | planned |
| LambdaMART on real MIND dev, with confidence intervals and ablations | planned |
| Candidate retrieval (FAISS + popularity), recall@K | planned |
| FastAPI `/recommend` service + Docker | planned |
| CI (GitHub Actions), monitoring | planned |

## Target architecture

```
                        ┌────────────────────────────────────────────┐
 news corpus ──────►    │  CANDIDATE RETRIEVAL            (planned)  │
                        │  FAISS ANN (semantic) ∪ recent popularity  │ ~200 candidates
                        └────────────────────┬───────────────────────┘
                                             ▼
                        ┌────────────────────────────────────────────┐
 user history ──────►   │  RANKING — LightGBM LambdaMART  (planned)  │
 item features          │  user / item / user-item features          │ ~top 50
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
uv sync

# Runs immediately — no download needed (synthetic impression logs):
uv run python -m mind_ltr.ranking.train_lambdarank

uv run pytest                                  # unit tests

uv run python scripts/download_mind.py         # real data (MINDsmall, ~250MB)
```

## Controlled synthetic experiment: position bias

The simulator generates clicks from a known examination model
(click = examined × relevant, with examination decaying by position), so
models trained on **biased clicks** can be evaluated against the **hidden
true relevance**:

| model | trained on | NDCG@10 vs hidden truth |
|---|---|---|
| pointwise (binary logloss) | raw clicks | ~0.755 (inherits popularity confound) |
| LambdaMART (lambdarank) | raw clicks | ~0.745 (ties pointwise — see note) |
| LambdaMART + IPW | reweighted clicks | **~0.802** |

These numbers come from the simulator, not from real MIND data.

Note: with binary labels, pointwise GBDT is a strong baseline and
lambdarank does not automatically beat it; its edge appears with graded
relevance and top-heavy cutoffs.

## Key design decisions

- **Negatives:** MIND impression logs contain shown-but-not-clicked items —
  real negatives, no sampling heuristics needed.
- **Splits:** temporal (train on earlier days, evaluate on later) — random
  splits leak the future through popularity and history features.
- **Position bias:** IPW (reweighting clicks by 1/pos^eta, clipped) is
  applied only in the simulator. MIND shuffles the order of items within an
  impression, so the token order carries no examination signal; the
  apparent position-CTR decay in MIND is a confound of impression length,
  not position bias. No IPW is applied to real MIND data.
- **Leakage rule:** every feature is computable strictly before the
  impression timestamp; encoders fit on train only.

## Metrics (planned for real MIND)

- **recall@K** — candidate retrieval quality
- **AUC, MRR, nDCG@5, nDCG@10** — ranking quality (MIND's standard metrics),
  per impression, averaged, with bootstrap confidence intervals
- Ablations by feature group; cold-start user / item segments
- Serving latency p50/p95

## How this design extends to video

Nothing in the ranking layer is text-specific — swapping the domain means
swapping the *feature preparation* front-end:

- **Item features:** title/abstract embeddings → title embeddings (BERT) +
  tag embeddings (CBOW, aggregated) + duration + language/video-ID
  embeddings, concatenated.
  Visual content adds frame-level embeddings (e.g. CLIP on sampled frames,
  mean-pooled) as one more concatenated block.
- **User features:** reading history → watch history / impressions /
  search queries, each aggregated from item embeddings the same way our
  "user taste vector" is built.
- **Labels:** click → watch-time-weighted labels (graded relevance instead
  of binary; LambdaMART handles this via `label_gain`), which also
  mitigates clickbait in a way clicks can't.
- **Pipeline:** identical multi-stage shape — candidate generation over
  billions, scoring over thousands, re-ranking over hundreds.
- **What genuinely changes:** feature freshness matters more (new videos
  cold-start faster), and the re-ranker gains responsibilities (freshness
  boosting, deduplication of near-identical uploads) beyond MMR diversity.

## Repo layout

```
src/mind_ltr/
  data/       MIND parsers (impression logs → LTR training table)
  features/   user features (item / interaction features planned)
  retrieval/  candidate retrieval (planned)
  ranking/    metrics (from scratch), LambdaRank gradients, synthetic simulator
  rerank/     IPW position-bias weights, MMR diversity
  serving/    FastAPI app (/health only; /recommend planned)
scripts/      MIND download, EDA
tests/        unit tests
```

## Roadmap

- [x] Metrics, LambdaRank from scratch, pointwise vs LambdaMART, IPW (synthetic simulator)
- [x] MIND parsers + EDA, user features
- [ ] Item / embedding / interaction features with leakage tests
- [ ] LambdaMART on real MIND dev: baselines, confidence intervals, ablations, cold-start segments
- [ ] Two-stage system: candidate retrieval + recall@K, end-to-end evaluation
- [ ] FastAPI `/recommend` with latency budget and popular-items fallback, small Docker image
- [ ] CI: ruff + pytest + Docker build tagged with the commit SHA
- [ ] Monitoring: impression logging with position, scheduled drift job
