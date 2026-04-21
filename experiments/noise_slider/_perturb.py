"""
Vectorised noise helper for the noise slider experiment.

Each voter+candidate answer at a cloned question is, independently:
  - kept with prob (1 − λ), or
  - switched to a value v' ∈ A_q \\ {x} with P(v' | x) ∝ 1 / |v' − x|,

where A_q is the per-question admissible set (unique non-NaN values observed
in the source column across voters + candidates). No hardcoded scale.
"""

import numpy as np
import pandas as pd


def admissible_set(*series: pd.Series) -> np.ndarray:
    """Unique non-NaN values observed in one or more source columns.

    Accepts multiple series (typically voter + candidate answers for the same question).
    Returns a sorted numpy array of the observed distinct values — no hardcoded ladder,
    so 4-, 5-, 7-, 8-point scales all work automatically.
    """
    merged = pd.concat(series, ignore_index=True)
    vals = merged.dropna().unique()
    vals = np.asarray(vals, dtype=float)
    vals.sort()
    return vals


def perturb_column(
    x: np.ndarray,
    lam: float,
    admissible: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Return a perturbed copy of x.

    For each non-NaN entry: switch with probability lam to a value v' drawn from
    admissible \\ {x_i} with P(v' | x_i) ∝ 1 / |v' − x_i|; otherwise keep x_i.
    NaN → NaN. Entries whose value is not in admissible, or admissible sets with
    cardinality ≤ 1, are skipped (recorded in the returned telemetry dict).

    Parameters
    ----------
    x : 1-D array of answers (may contain NaN).
    lam : switch probability ∈ [0, 1].
    admissible : sorted array of allowable values for this question.
    rng : np.random.Generator used for both the Bernoulli and the transition draws.

    Returns
    -------
    out : 1-D array, same shape as x, with perturbations applied.
    telemetry : dict with keys
        n_total        — total entries in x
        n_nan          — number of NaN entries (unchanged)
        n_skipped_offset — entries with value ∉ admissible (unchanged, defensive)
        n_skipped_small  — entries skipped because |admissible| ≤ 1 (set-level, not per entry)
        n_kept         — entries where the Bernoulli draw said "don't switch"
        n_switched     — entries actually switched to a new value
    """
    x = np.asarray(x, dtype=float).copy()
    out = x.copy()
    n_total = x.size

    telemetry = {
        "n_total": int(n_total),
        "n_nan": int(np.isnan(x).sum()),
        "n_skipped_offset": 0,
        "n_skipped_small": 0,
        "n_kept": 0,
        "n_switched": 0,
    }

    # Edge case 2: admissible set is trivial — nowhere to go.
    if admissible.size <= 1:
        telemetry["n_skipped_small"] = int(n_total - telemetry["n_nan"])
        return out, telemetry

    # Work only with non-NaN entries.
    non_nan_mask = ~np.isnan(x)
    idx = np.flatnonzero(non_nan_mask)
    if idx.size == 0:
        return out, telemetry

    # Edge case 1: defensive — any values not in admissible should not be perturbed.
    in_adm = np.isin(x[idx], admissible)
    bad_idx = idx[~in_adm]
    idx = idx[in_adm]
    telemetry["n_skipped_offset"] = int(bad_idx.size)

    if idx.size == 0:
        return out, telemetry

    # Bernoulli draw: u ≤ lam ⇒ switch (P(u ≤ lam) = lam for u ~ U[0, 1]).
    u = rng.random(idx.size)
    switch_mask = u <= lam
    switch_idx = idx[switch_mask]

    telemetry["n_kept"] = int(idx.size - switch_idx.size)
    telemetry["n_switched"] = int(switch_idx.size)

    if switch_idx.size == 0:
        return out, telemetry

    # Transition sampling (vectorised): for each i ∈ switch_idx, sample from
    # admissible \ {x_i} with P ∝ 1 / |v' - x_i|.
    k = admissible.size
    current = x[switch_idx]                         # (m,)
    diffs = np.abs(admissible[None, :] - current[:, None])  # (m, k)

    # Mask out self (distance 0) to avoid division-by-zero and self-transitions.
    self_mask = diffs == 0.0                        # (m, k)
    with np.errstate(divide="ignore"):
        inv = np.where(self_mask, 0.0, 1.0 / diffs)
    totals = inv.sum(axis=1, keepdims=True)         # (m, 1)
    probs = inv / totals                            # (m, k)

    # Vectorised categorical draw: inverse-CDF on each row.
    cdf = np.cumsum(probs, axis=1)                  # (m, k)
    r = rng.random(switch_idx.size)                 # (m,)
    choice_idx = (cdf < r[:, None]).sum(axis=1)     # (m,) in [0, k-1]
    choice_idx = np.minimum(choice_idx, k - 1)      # guard against fp edge case

    out[switch_idx] = admissible[choice_idx]
    return out, telemetry
