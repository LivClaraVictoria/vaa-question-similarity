# Forbidden Interval Coverage — Data README

## Background

This data comes from an analysis of **Clone-Robust Weighting (CRW)** applied to the
SmartVote Swiss voting advice application. CRW reweights questionnaire items by integrating
a graph-based weighting function over a radius parameter r ∈ [0, α]. At each radius r,
two questions x and y are treated as "equivalent" (same graph class) only if they share
identical neighborhoods — i.e., every other question z is either within distance r of both
or neither.

The experiment asks: **for a given pair of questions (x, y), how much of the CRW integration
range [0, α] can actually "see" the pair as equivalent?**

The context is the **mini vs. full (maxi) SmartVote questionnaire**. The mini questionnaire
contains 30 questions (column `rapide = 1` in the source data); the full questionnaire adds
45 more questions (maxi, `rapide = 0`). We analyze all 30 × 45 = 1,350 (mini, maxi) pairs.

Two distance metrics are analyzed in separate subfolders:
- `E5-INSTRUCT/` — multilingual sentence embedding distances (Euclidean on normalized vectors)
- `ANSWER-CORRELATION-ARCCOS/` — arccos(|Pearson r|) on voter answer vectors; this is
  considered the "ground truth" functional redundancy metric since it directly measures
  whether voters answer two questions the same way

---

## Key Concept: Forbidden Intervals

For a pair (x, y) and a third question z, the **forbidden interval** is:

    [min(d(x,z), d(y,z)),  max(d(x,z), d(y,z)))

At any radius r within this interval, z is a neighbor of exactly one of x or y — breaking
their equivalence. The **union** of forbidden intervals across all 73 other questions z gives
the set of radii where CRW cannot treat x and y as similar.

**Coverage fraction** at a given α = (measure of union ∩ [0, α]) / α.
A coverage of 0.8 means 80% of [0, α] is "forbidden" — CRW sees the pair as equivalent
only for the remaining 20%.

### Critical threshold: d(x, y)

CRW can only put x and y in the same graph class when r ≥ d(x, y) (only then are they
adjacent). Forbidden intervals for r < d(x, y) are geometrically interesting but
**irrelevant for CRW** — the pair isn't even adjacent there.

The **effective usable fraction** restricts to the adjacent window [d(x,y), α]:

    eff_usable_frac = measure([d(x,y), α] \ forbidden intervals) / (α − d(x,y))

This is 0 when d(x,y) ≥ α (the pair is never adjacent within [0, α] — CRW cannot
detect them as similar at all, regardless of the forbidden interval structure).

### Critical threshold: max pairwise distance

Once α exceeds the **maximum pairwise distance** in the dataset, the graph is fully
connected (every question is adjacent to every other). All forbidden intervals have
closed, so eff_usable_frac = 1 — but trivially, since everyone is in one equivalence
class with equal weights. The effective alpha range for CRW is therefore [0, max_pair_dist].

- E5-INSTRUCT: max pairwise distance ≈ **0.539**
- ANSWER-CORRELATION-ARCCOS: max pairwise distance ≈ **1.571** (≈ π/2)

---

## Files

Each metric subfolder contains (per run, identified by timestamp `MMDD_HHMM`):

### `forbidden_intervals_{metric}_{ts}.csv`

One row per **merged forbidden interval** per pair. Sorted by `d_xy` ascending
(closest pairs first = most likely approximate clones).

| Column | Description |
|--------|-------------|
| `q_mini_id` | Question ID of the mini questionnaire question |
| `q_mini_text` | Full text of the mini question |
| `q_maxi_id` | Question ID of the full-only (maxi) question |
| `q_maxi_text` | Full text of the maxi question |
| `d_xy` | Distance between x and y under this metric |
| `interval_start` | Start of this merged forbidden interval |
| `interval_end` | End of this merged forbidden interval |
| `metric` | Metric name |

A pair with many rows has many disjoint forbidden intervals. A pair with no rows has no
forbidden intervals (identical neighborhoods at all radii — but may still have d_xy > α).

### `forbidden_coverage_{metric}_{ts}.csv`

One row per **(mini, maxi) pair**. Contains both raw and effective coverage statistics
at every α value in the grid (0.1, 0.2, ..., 1.5, 1.6, 1.8, ..., 3.0).

#### Fixed columns

| Column | Description |
|--------|-------------|
| `q_mini_id` | Question ID of the mini question |
| `q_mini_text` | Full text of the mini question |
| `q_maxi_id` | Question ID of the maxi question |
| `q_maxi_text` | Full text of the maxi question |
| `d_xy` | Distance between x and y |
| `n_merged_intervals` | Number of merged forbidden intervals (after union) |
| `total_interval_measure` | Total length of all merged forbidden intervals (regardless of α) |
| `metric` | Metric name |

#### Per-alpha columns (repeated for each α in the grid)

| Column | Description |
|--------|-------------|
| `covered_measure_{α}` | Absolute measure of forbidden intervals ∩ [0, α] |
| `coverage_{α}` | Fraction of [0, α] that is forbidden = `covered_measure / α` |
| `eff_usable_measure_{α}` | Absolute measure of [d_xy, α] that is NOT forbidden |
| `eff_usable_frac_{α}` | Fraction of [0, α] that is both adjacent and usable = `eff_usable_measure / α`. **0.0 when d_xy ≥ α** (pair never adjacent — CRW cannot detect them) |

**Interpretation guide:**
- `coverage_{α}` close to 1 → nearly all of [0, α] is forbidden (pair has very different
  neighborhoods). But this includes the pre-adjacency range which is irrelevant.
- `eff_usable_frac_{α}` close to 1 → within the adjacent window [d_xy, α], the pair has
  nearly identical neighborhoods. CRW *can* detect them as similar, but whether it produces
  a meaningful weight change also depends on class size at those radii.
- `eff_usable_frac_{α}` = 0.0 AND d_xy ≥ α → pair is never adjacent; CRW is blind to this pair.
- `eff_usable_frac_{α}` = 0.0 AND d_xy < α → adjacent window exists but entirely covered by
  forbidden intervals (pair has completely different neighborhoods when adjacent).

---

## Key Finding

For the mini/maxi natural redundancy setting:

- **E5-INSTRUCT (α=0.3–0.4):** Most pairs have d_xy > α, so `eff_usable_frac = 0` —
  CRW is completely blind to these pairs. Even the closest pairs (d_xy ≈ 0.26) have a
  very narrow adjacent window at typical alphas.

- **ANSWER-CORRELATION-ARCCOS (α=1.1–1.5):** The closest pairs do have d_xy < α (e.g.
  d_xy ≈ 0.68), giving a meaningful adjacent window. `eff_usable_frac` is high (pair
  neighborhoods are similar within the window). Yet CRW still produces near-zero weight
  correction. The reason: at radii r ≥ 0.68, many other question pairs are also adjacent,
  so the equivalence classes are large and adding one more member barely changes any weight.

This demonstrates that **high `eff_usable_frac` is necessary but not sufficient** for CRW
to correct redundancy. CRW requires the redundant pair to form a *small, dense cluster*
(near-zero d_xy), not just share similar neighborhoods at large radii.

---

## Suggested Analyses

- **Sort by `d_xy`** to find the most redundant mini/maxi pairs (potential approximate clones).
- **Filter `d_xy < α`** to find pairs where CRW is at least in principle able to act.
- **Plot `eff_usable_frac_{α}` vs α** for the closest pairs to see how the usable window
  evolves with alpha.
- **Aggregate `eff_usable_frac_{α}` across all pairs** at a fixed α to get a distribution
  of CRW's detection capability across the full mini/maxi question set.
- **Cross-compare metrics:** a pair with high `eff_usable_frac` under ARCCOS but low under
  E5 indicates the functional redundancy (voter answers) is not reflected in the text embeddings.
