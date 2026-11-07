# Science and math log

The science behind every component of mind-ltr, in the order it was built.
Each section states the model or formula, why it is the right tool here, and
how it is verified. Every number quoted is produced by code in this repo;
the command to reproduce it is given next to it.

**Notation.** An *impression* is one ranking event: user *u* at time *t* is
shown a list of items *i₁ … i_L* and clicks some of them. `y ∈ {0,1}` is
the click label, `s` the model score (higher = ranked earlier), `rank(i)`
the 1-indexed position of item *i* when sorted by `s`.

---

## 1. Problem framing: learning to rank from impressions

We learn a scoring function `s = f(user, item, context)` such that, within
each impression, sorting by `s` puts clicked items first. Quality is
measured **per impression** and then averaged over impressions, because a
ranking is only ever compared against the other items shown in the same
list.

MIND's impression logs list shown-but-not-clicked items (`N123-0`), so
negatives are real (the user saw them and passed) and no negative-sampling
heuristic is needed.

---

## 2. Ranking metrics (`src/mind_ltr/ranking/metrics.py`)

All implemented from scratch and unit-tested against hand-computed values
(`tests/test_metrics.py`).

**DCG / NDCG@k**

```
DCG@k  = Σ_{r=1..k} (2^{y_r} − 1) / log₂(r + 1)
NDCG@k = DCG@k(model order) / DCG@k(ideal order)        ∈ [0, 1]
```

The gain `2^y − 1` rewards relevant items; the discount `1/log₂(r+1)`
makes rank 1 worth 1.0 and rank 10 worth ≈ 0.29. Dividing by the ideal DCG
makes impressions with different numbers of clicks comparable. Convention:
an impression with no clicks has NDCG = 0.

**MRR (reciprocal rank)**

```
RR = 1 / rank(first clicked item)
```

Only the first hit matters: a proxy for "time to first click".

**Recall@k**

```
Recall@k = |relevant ∩ top-k| / |relevant|
```

The metric for a *retrieval* stage: its only job is to not lose relevant
items before the ranker sees them.

**AP@k (MAP when averaged)**

```
AP@k = ( Σ_{r ≤ k, y_r = 1} Precision@r ) / min(k, total relevant in the list)
```

The normalizer counts relevant items in the **whole** list, not only those
found in the top-k. Normalizing by "relevant found" would forgive every
miss: a model that finds 1 of 5 relevant items would score the same as one
that finds 1 of 1.

---

## 3. From pointwise to LambdaMART (`src/mind_ltr/ranking/lambdas.py`)

**Pointwise** treats each row as independent binary classification
(log-loss on `y`). It ignores that items compete within a list.

**RankNet (pairwise).** For a pair (i, j) with `y_i > y_j`, model
`P(i ≻ j) = σ(s_i − s_j)` and minimize cross-entropy. The gradient magnitude
on the pair is

```
ρ_ij = σ_s / (1 + exp(σ_s · (s_i − s_j)))
```

large when the pair is mis-ordered, small when it is already correct.

**LambdaRank.** RankNet treats a swap at ranks 50/51 the same as one at
ranks 1/2, but NDCG does not. LambdaRank scales each pair's force by how
much NDCG would change if the two items swapped:

```
|ΔNDCG_ij| = |gain_i − gain_j| · |disc_i − disc_j| / IDCG
λ_ij       = ρ_ij · |ΔNDCG_ij|
λ_i += λ_ij   (push the more relevant item up)
λ_j −= λ_ij   (push the less relevant item down)
```

with `gain = 2^y − 1` and `disc = 1/log₂(1 + rank)` from the *current*
scores. The second-order term used for Newton steps is
`h_ij = ρ_ij (σ_s − ρ_ij) |ΔNDCG_ij|` (always ≥ 0).

**LambdaMART** = gradient-boosted trees fitted to these λ's as
pseudo-residuals. LightGBM minimizes, so our custom objective returns
`(−λ, h)`; a floor `h ≥ 1e-9` keeps the Newton step stable.
`make_lgb_objective` plugs this into LightGBM, so the same trees can be
trained with our gradients or LightGBM's native `lambdarank`.

**Finding (synthetic, section 4):** with binary labels, pointwise GBDT is a
strong baseline and plain LambdaMART does not beat it (0.7553 vs 0.7454
NDCG@10). LambdaMART's advantage comes from graded relevance and top-heavy
cutoffs, which binary clicks do not provide.

---

## 4. Position bias and IPW, in a controlled simulator

### The examination model (position-based model)

```
P(click | u, i, pos) = P(examined | pos) · P(relevant | u, i)
                     =      π(pos)       ·      r(u, i)
```

Users look at the top of a list more than the bottom, so clicks mix
relevance with exposure. A model trained on raw clicks partly learns
"whatever the old ranker put on top".

### Inverse propensity weighting (`src/mind_ltr/rerank/position_bias.py`)

If π is known, weighting each click by `1/π` removes the bias in
expectation:

```
E[ c / π(pos) ] = π(pos) · r / π(pos) = r
```

We use the parametric propensity `π(pos) = 1 / pos^η` and weight clicked
rows by `min(1/π, clip)`. Clipping trades a little bias for much lower
variance: without it, a single click at a deep position can dominate the
loss. Non-clicked rows keep weight 1, an approximation (an unclicked item
at a deep rank may simply not have been seen).

### The simulator (`src/mind_ltr/ranking/synthetic.py`)

Because real logs never reveal true relevance, IPW is validated where the
ground truth is known:

1. Each item has hidden true relevance driven by its features; feature 0
   ("popularity") is only weakly relevant.
2. A legacy ranker orders items almost entirely by feature 0.
3. Examination follows `1/pos^η`; click = examined AND relevant.

So in the click log, feature 0 *looks* predictive only because it controlled
exposure: a confound that IPW should undo.

### Result (controlled synthetic experiment)

Reproduce: `uv run python -m mind_ltr.ranking.train_lambdarank`
(3,000 impressions × 50 items, 80/20 split by impression, evaluated against
hidden true relevance)

| model | trained on | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|---|
| pointwise (binary log-loss) | raw clicks | 0.7553 | 0.8913 | 0.3859 |
| LambdaMART | raw clicks | 0.7454 | 0.8842 | 0.3811 |
| LambdaMART + IPW (η = 1) | reweighted clicks | **0.8018** | **0.9381** | **0.4052** |

These are simulator numbers, not results on real MIND data.

---

## 5. Why there is no IPW on real MIND

MIND documents that the order of items inside an impression is shuffled, so
the token position is not the display position and carries no examination
signal. Yet a naive plot of CTR against token position on MIND *does*
decay. The decay comes from a **length confound**, not position bias:

Suppose position has no effect at all and each impression of length L gets
about K clicks regardless of L. Then every item in that impression has
click probability ≈ K / L. Position p only exists in impressions with
L ≥ p, so

```
CTR(p) = E[ K / L  |  L ≥ p ]
```

As p grows, the short impressions (high K/L) drop out of the average, so
CTR(p) falls even though position does nothing. The diagnostic is to
**stratify by impression length**: within impressions of a fixed length, a
true position effect would still decay, while a pure length confound gives
a flat curve.

Consequence: IPW and other propensity methods are used only where
propensities are known (the simulator). On MIND we evaluate without
propensity weights. The production answer is to **log the display
position** at serving time, so a future propensity estimate is possible
(Phase E).

---

## 6. MMR diversity re-ranking (`src/mind_ltr/rerank/mmr.py`)

Greedy list construction that trades relevance against redundancy with
items already selected:

```
MMR(i) = λ · score(i) − (1 − λ) · max_{j ∈ selected} cos(e_i, e_j)
```

λ = 1 is pure relevance; λ = 0 is pure diversity. It is a cheap
post-processing step over the top few hundred candidates. Implemented;
wired into serving in Phase C.

---

## 7. User features (`src/mind_ltr/features/user.py`)

```
history_length(u)        = |H_u|
cat_share_c(u)           = |{ i ∈ H_u : category(i) = c }| / |H_u|      (0 if H_u empty)
```

`H_u` is the click history MIND provides for each user. It is recorded
*before* the logging window and is identical on every impression of the
user within a split, so these features use no information from the
impressions being ranked (no leakage).

---

## 8. Time-lagged item popularity (Phase A1)

Reproduce the numbers in this section:
`uv run python scripts/explore_popularity.py`

### Data facts (MINDsmall)

| fact | value |
|---|---|
| train logs | 2019-11-09 → 2019-11-14 (6 days, 5,843,444 impression rows, 156,965 impressions) |
| dev logs (our test set) | 2019-11-15 (1 day, 2,740,998 rows, 73,152 impressions) |
| base CTR | 0.0404 (train), 0.0406 (dev) |
| article lifetime in the logs | median 5.8 h, p75 20.4 h, p90 49.9 h |
| dev rows whose article never appears in train | 21.3% of rows, 24.9% of clicks |

### Why popularity

Model each article's click probability as a time-varying `θ_i(t)` and each
impression as `click ~ Bernoulli(θ_i(t))`. Much of `θ_i(t)` is shared by
all users (a breaking story is clicked by almost everyone), so the article's
click record *so far* is a strong estimate of it. News is perishable
(median lifetime under 6 hours), so "so far" must be computed carefully in
time and kept fresh.

### Leakage

A feature for an impression at time t may only use information available
before t:

```
x(u, i, t) must be a function of F_t = { events logged before t }
```

- **Target leakage:** the feature contains the row's own label (e.g. an
  item CTR computed over data that includes this click).
- **Future leakage:** the feature uses events after t (e.g. "CTR over the
  whole week" for a Tuesday impression).

A leaky feature inflates offline metrics, the model learns to rely on it,
and at serving time the feature cannot be computed the same way
(training–serving skew), so quality drops silently.

### Time-lagged cumulative counts

Cut time into hourly buckets b. With `c_i(b)` clicks and `n_i(b)`
impressions of item i in bucket b, an impression in bucket b gets

```
C_i(b) = Σ_{b' < b} c_i(b')        clicks strictly before this bucket
N_i(b) = Σ_{b' < b} n_i(b')        impressions strictly before this bucket
```

an *exclusive* cumulative sum: the current bucket, and therefore the row's
own label, never enters. The hourly bucket also models a feature pipeline
that refreshes every hour (features are 0–60 minutes stale).

### Smoothing (Beta–Binomial shrinkage)

Raw `C/N` is noisy for small N (1 click in 1 view = 100%). Put a Beta prior
on each item's CTR, centered on the global CTR p with strength m
pseudo-impressions:

```
θ_i ~ Beta(α, β),   α = m·p,   β = m·(1 − p)
E[θ_i | C, N] = (C + α) / (N + α + β)
```

With p = 0.04 and m = 100 (α = 4, β = 96):

| item | raw CTR | smoothed |
|---|---|---|
| 1 click / 1 view | 1.000 | 5/101 = 0.050 |
| 500 / 1,000 | 0.500 | 504/1,100 = 0.458 |
| 0 / 0 (new item) | undefined | 4/100 = 0.040 (the prior) |

Small samples are pulled toward the global mean; large samples keep their
own value. m is tuned on the validation day, never on the test set.

### Evidence on MIND

Per-impression mean AUC of each popularity score on the last train day
(2019-11-14):

| popularity score | leakage | AUC |
|---|---|---|
| CTR over the whole train week (includes the row and the future) | leaky | 0.6670 |
| CTR from days before today, raw | none | 0.5621 |
| CTR from days before today, smoothed (m = 100) | none | 0.5734 |
| CTR from hours before now, smoothed (m = 100) | none | **0.6347** |
| log(1 + impressions before now) | none | 0.4607 |

- The leaky score looks best; that is exactly the trap.
- Hourly beats daily because articles are short-lived: with a daily lag,
  29.2% of last-day rows belong to articles with no history at all; with
  an hourly lag, 0.7%.
- Exposure volume alone is *below* random (0.46): heavily shown articles
  are older and get fewer clicks per view. Volume is not quality; the
  ranker receives both and can learn the interaction.

### Design decision: streaming features for the test day

Features are computed over one continuous timeline (train and dev logs
together), always with the hourly lag. A 14:00 test impression on Nov 15
may use clicks from Nov 15 before 14:00, as an online system with an hourly
feature job would. Train, validation and test features are computed
identically, so there is no skew. Rejected alternative: freezing features
at the end of Nov 14 (a nightly batch job), which would give 21% of test
rows only the prior; it is kept as a robustness check ("what if the feature
pipeline stalls for a day?").

### Features

| feature | definition |
|---|---|
| `item_ctr_smooth` | (C + α) / (N + α + β), hourly lag |
| `item_log_impressions` | log(1 + N) |
| `item_age_hours` | hours since the article was first seen in the logs (MIND has no publish time) |

### How leakage is tested

1. **Future perturbation:** randomize every label in the current hour and
   later; features of earlier rows must not change.
2. **Own-label invariance:** flip one row's label; its features must not
   change.
3. **Hand-computed fixture** with exact expected values, including the
   smoothing.

The tests were themselves checked by mutation: making the cumulative sum
include the current hour (a deliberate leak) makes the hand-computed and
own-label tests fail.

### Implementation results (`src/mind_ltr/features/item.py`)

`build_item_popularity` over the full streaming timeline (train + dev,
8,584,442 rows, prior p from the training window only) runs in seconds on
a laptop CPU with no missing values. Per-impression AUC of each feature on
its own:

| feature | 2019-11-14 (validation day) | 2019-11-15 (test day) |
|---|---|---|
| `item_ctr_smooth` | 0.6347 | 0.6439 |
| `item_log_impressions` | 0.4607 | 0.5169 |
| `item_age_hours` | 0.4616 | 0.4859 |

The validation-day CTR value matches the independent computation in the
evidence table above. Volume and age are weak on their own; their value,
if any, comes from interactions inside the ranker and will be measured by
the feature-group ablations (Phase A4).

Cold start: about a quarter of test clicks go to articles never seen in
train. Popularity helps them only after their first hour; content features
(Phase A2) cover the rest, and results will be reported separately for cold
and warm items.

---

## Changelog

| date | section | change |
|---|---|---|
| 2026-11-07 | 1–7 | Documented components built before the restart |
| 2026-11-07 | 8 | Phase A1 science: time-lagged popularity, leakage, smoothing |
| 2026-11-07 | 8 | Phase A1 implemented: `features/item.py`, leakage tests, real-data results |
