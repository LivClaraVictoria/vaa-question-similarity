# Handoff — canton validation/test runs (written 2026-10-05)

Read this first, then `CLAUDE.md`. Read `configs/base_constants.py` before anything else about parameters:
it holds the project's key constants (seats per canton → Jaccard k, hash params, paths, defaults).
Do NOT dig into `dependencies/rsfp/` (the user does not want that).

## 0. How to work with this user (they flagged "careless mistakes")
- Verify state with read-only commands before acting. An **interrupted/rejected command may still have run**:
  a rejected command had already made a commit, and another had already submitted a Slurm job. After any
  interruption, check `git log`, `git status`, `squeue -u $USER` before assuming nothing happened.
- Do NOT launch jobs, start long waits, or commit/amend unless asked. Do not add compute "just in case"
  (the user rejected computing Jaccard@k for a grid of k as wasteful).
- Cancel only exact job IDs you submitted (`scancel <ids>`); `scancel -u $USER` was blocked by the safety classifier.
- The user is often on a phone: short, concrete messages; say plainly what is verified and what is not.
- Pushing to GitHub needs the user's credentials (none on this account) — they push themselves.

## 1. Goal
ZH (Zurich) = validation canton: all parameters were selected on ZH only. Other cantons = held-out test
cantons; BE (Bern) is the base test canton (shown as its own case), the rest are comparators showing stability.
All test cantons use ONE alpha (the ZH-chosen one) and no alpha sweeps. Cantons excluded: AR AI GL NW OW UR
(1–3 candidates). Details of the mechanism are in `CLAUDE.md` (VQS_DISTRICT, VQS_ALPHA, canton_results_path,
cache rules, orchestrator).

## 2. What ran tonight (run dir `experiment_results/cantons/_runs/20261004_224705/`, alpha=0.4)
Job states from `bash jobs/cantons/status.sh` (all verified):
| experiment | result |
|---|---|
| `pipeline` (distances→CRW→recs) | 19/19 cantons COMPLETED; every canton's recs JSON has `n_jaccard` = its seat count (BE 24, AG 16, … JU 2) |
| `partisan_sweep` (Phase 2, 6 parties, ZH Phase-1 selection) | 19/19 COMPLETED (7 jobs each). Output dir exists per canton; only BE's party subfolders were opened (Centre FDP GLP Green SP SVP). **Contents not checked.** |
| `rec_distortion` BE (75 q × 5 clone types, α=0.4) | COMPLETED (375 worker CSVs + 1 collected CSV, report + plots) in `experiment_results/cantons/BE/exp1/question_alpha_sweep/` |
| `rec_distortion` all 18 comparators | **ALL FAILED** (375 tasks each) — no results. Cause below. |

### Bug that killed the comparators (fixed, commit `b1ea28e`, NOT pushed)
`ResultManager.exists()` finds cache files by globbing `*{hash}.parquet` and ignores the prefix. My new base-side
CRW cache (`crw_…`) had the same hash as the pipeline's `recs_…` file (same params), so tasks loaded the pipeline's
table → `KeyError: '__voter_idx__'`. Fix: base-side cache now in `$VQS_REC_CACHE_DIR/base_side/`
(`$SCRATCH/vaa-question-similarity/cache/recommendations/base_side/`; the 17 valid existing files were moved there).
BE's own results are valid (its tasks read their own `crw_BE` file; pipeline outputs were computed independently).

## 3. FIRST TASK: check tonight's results (nothing below has been checked beyond what is stated)
1. **BE vs ZH rec_distortion at α=0.4** (done once, numbers): ZH file
   `experiment_results/exp1/question_alpha_sweep/e5_instruct_5ct_n4/compiled/*a04_5ct_n5_0311_1543.csv` (k=36),
   BE CSV `…/cantons/BE/exp1/question_alpha_sweep/*/*districtBE*_1005_0250.csv` (k=24). Mean Jaccard base/CRW:
   easy 0.811/0.900 vs 0.810/0.899; hard 0.811/0.856 vs 0.810/0.855; neg_easy 0.811/0.892 vs 0.810/0.890;
   neg_hard 0.811/0.843 vs 0.810/0.842; mix 0.811/0.857 vs 0.810/0.856. CRW helps on 75/75 questions in both;
   per-question gain correlation ZH vs BE r=0.992. **Suspiciously identical to 3 decimals** — one plausible
   reason: seat-based k is ~3.5% of the candidate list in both (24/685, 36/1029), but this is unproven.
   Sanity-check (e.g. different clone types/voters differ? is BE really on BE data — log said 34,052 voters /
   685 candidates, so yes) before trusting it.
2. **Partisan alpha (resolved, check one detail):** `launch_partisan_sweep.sh` uses config `pipeline_e5_instruct_ZH_a03`
   and `VQS_ALPHA=0.4` re-targeted it, so test cantons ran Phase 2 at α=0.4 (BE folder:
   `pipeline_e5_instruct_ZH_a03_districtBE_alpha0.4`). ZH's Phase 2 exists at BOTH alphas for all 6 parties
   (`experiment_results/party_impact/high_impact/phase2/pipeline_e5_instruct_ZH_a03` = Mar 3, `…_a04` = Mar 11), so
   **no ZH rerun is needed: compare test cantons against `…_ZH_a04`.** Still unverified: that the a04 run used the same
   Phase 1 CSV as the test cantons (`party_impact_pipeline_e5_ZH_0302_0057.csv`, set in `launch_partisan_sweep.sh`).
3. **Clone count (resolved):** 4 clones per question in the BE run — launcher `N_CLONES=4`, worker logs print
   `Clones : 4`, and perfect_mix requires a multiple of 4 (equal share of its 4 component types; this is the
   "same number of clones across scenarios" rationale; 5 would crash). The `n5` / "× 5" labels came from the collect
   step not passing `--n-clones` (cosmetic); fixed in `job_question_alpha_sweep_collect.sh` (uncommitted), so
   future runs are labelled correctly. BE's existing files still say n5 — optionally rerun only the collect step to
   regenerate the report (seconds, no cluster compute). ZH's count is inferred (folder `n4`, perfect_mix rows
   present), not logged.
4. Open the pipeline stats (`*.txt` next to each `recs_*.parquet` in `experiment_results/cantons/<C>/pipeline_outputs/`)
   and the partisan outputs per canton; confirm they are non-empty and plausible.

## 4. Decisions / answers from the user so far
- **Jaccard k: resolved.** k = seats per canton from `SEATS_PER_CANTON` in `configs/base_constants.py` (ZH 36,
  BE 24, AG 16, LU 9, SG 12, VD 19, …), via `experiments/_common._resolve_n` / `vqs/recommendation_saver._get_jaccard_n`
  (when `n_recommendations="all"` and a district is set). The supervisor's "ZH gets 37" was misremembered by the user:
  **36 is correct, no change, no rerun needed for k.** Do not compute Jaccard for a grid of k (rejected as wasteful).
- **One-seat cantons are excluded** (AR AI GL NW OW UR: 1–3 candidates); 19 cantons ran. Very small ones still in:
  SH, JU (2 seats), ZG (3), BS, SZ, NE (4) — top-2 Jaccard is coarse. Open: drop them from the clone sweep or
  just filter in the comparison (user has not decided).
- **Alpha 0.4:** user is "90% sure"; supported by the ZH final rec_distortion files being `…ZH_a04…` and by ZH Phase 2
  existing at 0.4. Paper PDF could not be read (no poppler).
- **Still to do: rerun the comparators' `rec_distortion`** (the "retry"); not running now. The last attempt was
  cancelled (orchestrator job 16124340 cancelled before starting).

## What the clone sweep (`rec_distortion`) is
For each of 75 questions × 5 clone types (easy paraphrase, hard paraphrase, negation-easy, negation-hard, perfect mix),
add 4 clones of that question (clones copy the original's answers; negation types flip them). Rank all candidates
per voter on the cloned questionnaire twice — plain (`base_*`) and with CRW at α (`crw_*`) — and compare with the same
ranking on the original questionnaire: Jaccard of the top-k lists (k = seats), plus Spearman/Kendall on the full
ranking, averaged over voters. Higher = closer to the original = less distortion. 375 tasks per canton.

## 5. How to (re)run — only when the user says so
Fresh run: `ALPHA=0.4 EXPERIMENTS="rec_distortion" bash jobs/cantons/run_all_cantons.sh` (submits a tiny orchestrator
job; log `…/_runs/<ts>/orchestrator.log`; `bash jobs/cantons/status.sh`). Comparators only, reusing the valid BE
result: a prepared resume dir exists at `experiment_results/cantons/_runs/20261005_comparators/` (BE marked done,
`peak_rec_distortion`=14.3); run with `RUN_DIR=<that dir>` (relative path works from the repo root).
Measured on Bern: each task ≈3.5–4 min, peak 11–14 GB, 1 CPU. Euler caps ~128 GB memory per user (≈8 tasks at 16 GB)
and ~1000 queued tasks. BE has priority (comparators wait for BE of the same experiment and use `--nice`).
Logs: `$SCRATCH/vaa-question-similarity/logs/`. Python: `~/miniforge3/envs/bachelor-thesis/bin/python`
(use `PYTHONPATH=.` for ad-hoc scripts). Tests: `python -m pytest tests -q` (20 pass).

## 6. Repo state
- `main` is 1 commit ahead of `origin/main` (`b1ea28e`, unpushed). Earlier commits (`dc20f6e`, `2087fcf`, `40c5670`)
  are pushed. `CLAUDE.md` is tracked (removed from `.gitignore`).
- Uncommitted edit: `jobs/perfect_clones/job_question_alpha_sweep_collect.sh` (passes `--n-clones`, label fix).
- Untracked, intentionally not committed: `experiment_results/cantons/` (results, run manifests, empty worker dirs
  from failed/cancelled runs — safe to ignore) and one ZH distance parquet in `experiment_results/pipeline_outputs/`.
- Not built yet: cross-canton comparison tables/plots (collect per-canton CSVs by path under
  `experiment_results/cantons/<C>/…`; note folder names contain `_districtXX`/`_alpha0.4` suffixes, ZH's do not).
- Not done: one-line README note about the canton workflow (see CLAUDE.md instead).
- Open design choices (also in CLAUDE.md): partisan Phase 2 uses ZH's Phase 1 selection; approximate clones
  re-select on each canton's voters; variance-based clone selectors used national data.
