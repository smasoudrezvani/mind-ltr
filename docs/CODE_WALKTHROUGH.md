# Code walkthrough

How the code for each feature works, file by file, in the order it was
built. The science and math behind each feature are in
[SCIENCE.md](SCIENCE.md); this document explains the implementation and how
it is tested.

---

## Phase A1: time-lagged item popularity

Science: [SCIENCE.md, section 8](SCIENCE.md#8-time-lagged-item-popularity-phase-a1).

Files:
- `src/mind_ltr/features/item.py`: `build_item_popularity`
- `tests/test_item_features.py`: 9 tests
- `scripts/explore_popularity.py`: exploratory analysis and real-data check

### `src/mind_ltr/features/item.py`

**1. Signature and input checks**

```python
def build_item_popularity(rows, prior_ctr, prior_strength=100.0, freq="h") -> pd.DataFrame:
```

- `rows` is the exploded impression log from `explode_impressions`: one row
  per (impression, shown article), with `news_id`, `time` and `clicked`.
- `prior_ctr` (p) is required, with no default. The caller must compute it
  on the **training window only**: the global CTR is itself a statistic of
  the labels, so computing it over test rows would leak a little test-label
  information into every feature.
- `prior_strength` is m, the prior weight in pseudo-impressions.
- `freq="h"` sets hourly buckets (any pandas offset alias works).
- Missing columns or an invalid prior raise a `ValueError` with a clear
  message.

**2. Per-bucket counts and the exclusive cumulative sum**

```python
bucket = rows["time"].dt.floor(freq)                       # 10:40 -> 10:00
per_bucket = (
    pd.DataFrame({"news_id": ..., "bucket": bucket, "clicked": ...})
    .groupby(["news_id", "bucket"], sort=True)["clicked"]
    .agg(clicks="sum", imps="size")                         # c_i(b), n_i(b)
)
by_item = per_bucket.groupby(level="news_id")
per_bucket["clicks_before"] = by_item["clicks"].cumsum() - per_bucket["clicks"]  # C_i(b)
per_bucket["imps_before"] = by_item["imps"].cumsum() - per_bucket["imps"]        # N_i(b)
```

- `sort=True` orders buckets in time within each article, so `cumsum()` is a
  running total through time.
- `cumsum()` includes the current bucket; subtracting it gives the sum over
  strictly earlier buckets. This subtraction is the line that prevents
  leakage: without it, the row's own label and every other click in the same
  hour would enter the feature.

**3. Aligning back to the rows by key**

```python
keys = pd.MultiIndex.from_arrays([rows["news_id"], bucket], names=["news_id", "bucket"])
before = per_bucket.reindex(keys)
```

Each input row looks up its own (article, hour) entry, so the result is
correct whatever the order or index of `rows`. Aligning by position instead
of by key is a common silent pandas bug; the test fixture uses a
non-default index to catch it.

**4. The three features**

```python
alpha = prior_strength * prior_ctr
item_ctr_smooth      = (clicks_before + alpha) / (imps_before + prior_strength)
item_log_impressions = np.log1p(imps_before)
item_age_hours       = (rows["time"] - first_seen).dt.total_seconds() / 3600
```

- The smoothed CTR is (C + α) / (N + α + β). Since α + β = m·p + m·(1 − p) =
  m, the denominator uses `prior_strength` directly.
- `first_seen` is the earliest time the article appears in `rows`. It is
  never later than any of the article's rows, so it uses no future
  information. Limitation: articles that existed before the first logged day
  get an age measured from that day (left-censoring).
- The output keeps the **same index as the input**, so the caller can
  `rows.join(features)`.

### `tests/test_item_features.py`

| test | what it proves |
|---|---|
| `test_hand_computed_values` | Two articles over three hours, every expected value worked out by hand in the docstring (e.g. article A at 11h: 1 click in 2 earlier views → (1 + 1)/(2 + 10) = 2/12) |
| `test_output_aligned_to_input_index_and_input_not_mutated` | Output index equals input index; the input frame is unchanged |
| `test_future_and_present_labels_do_not_change_past_features` | Randomizing every label from a cutoff hour on leaves all earlier features identical |
| `test_appending_future_rows_does_not_change_past_features` | Adding 500 rows five days later leaves all earlier features identical |
| `test_own_label_does_not_change_own_bucket_features` | Flipping one row's label changes nothing in its hour (target leakage) |
| `test_invalid_prior_raises` (3 cases) | p ∉ (0, 1) or m ≤ 0 raise `ValueError` |
| `test_missing_column_raises` | A missing required column raises `ValueError` naming it |

The leakage tests are random-log property tests: they check a rule for every
row instead of a few hand-picked values. They were verified by mutation:
replacing the exclusive cumulative sum with an inclusive one makes
`test_hand_computed_values` and `test_own_label_does_not_change_own_bucket_features`
fail.

### `scripts/explore_popularity.py`

Reproduces every MIND number in SCIENCE.md section 8:

```bash
uv run python scripts/explore_popularity.py
```

- `describe_splits`: date ranges, sizes, CTR, cold-start share, article
  lifetimes.
- `compare_popularity_scores`: leaky vs day-lagged vs hour-lagged popularity
  on the last train day, computed independently of `features/item.py`.
- `check_item_features`: runs `build_item_popularity` over the streaming
  timeline (train + dev, prior from train only) and reports each feature's
  AUC on the validation and test days.
- `mean_impression_auc`: MIND-style AUC (per impression, then averaged),
  vectorized with the Mann–Whitney rank formula
  `(sum of positive ranks − P(P+1)/2) / (P·Q)`, using average ranks for ties
  like scikit-learn's `roc_auc_score`.

---

## Changelog

| date | section | change |
|---|---|---|
| 2026-11-07 | Phase A1 | `features/item.py`, its tests, and the analysis script |
