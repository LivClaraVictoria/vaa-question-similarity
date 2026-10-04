# CLAUDE.md

Bachelor thesis codebase: clone-robust weighting (CRW) for Smartvote VAA recommendations. See README.md
for experiments and entry points. Run everything as modules from the repo root (`python -m main ...`).

## Environment
- Cluster: ETH Euler (profile `jobs/clusters/euler.sh`, selected in `jobs/cluster.conf`).
- Python: conda env `bachelor-thesis` at `~/miniforge3/envs/bachelor-thesis/bin/python` (no `python`
  on the login shell PATH). Tests: `python -m pytest tests -q`.
- Slurm logs: `$SCRATCH/vaa-question-similarity/logs/`. Data (confidential, never commit): `data/cleaned/`.
- `configs/base_constants.py` and `configs/create_clones/base_clone_creator.py` use CRLF line endings — keep them.

## Validation / test split across cantons
- ZH is the validation canton: all parameters (model, alpha, clone sets, thresholds) were selected on ZH
  only. Other cantons are held-out test cantons; BE is the base test canton, the rest are comparators.
  Never tune anything on test-canton results.
- Canton is chosen with env var `VQS_DISTRICT=<code>`: `vqs.config_utils.load_config` re-targets every
  config with `district != "all"` (registered as an override, so run names get `_districtXX`).
  `sbatch --export=ALL` forwards it, so existing launchers work unchanged:
  `VQS_DISTRICT=BE bash jobs/perfect_clones/launch_rec_distortion.sh`.
- Results: ZH keeps the original `experiment_results/` tree; test cantons get the same layout under
  `experiment_results/cantons/<code>/` (`canton_results_path` in Python, `results_root` in `jobs/_lib/common.sh`).
- All test cantons at once: `bash jobs/cantons/run_all_cantons.sh` (BE first; array experiments of other
  cantons wait for BE's so shared caches are warm). Manifest + job ids: `experiment_results/cantons/_runs/<ts>/`.
- One-seat cantons (AR AI GL NW OW UR) are excluded: 1–3 candidates make ranking metrics meaningless.

## Caching rules
- Cache keys (`ResultManager`, MD5 of config params) for text-embedding metrics are canton-independent
  (question text only). Answer-based metrics (`ANSWER_BASED_METRICS`) add `ANSWER_METRIC_HASH_PARAMS`
  (incl. `district`) at every stage via `respondent_hash_params(config)` — keep this when adding stages.
- Never re-implement the hash in scripts; build it from the same param lists (see
  `distance_structure_analysis._compute_dist_hash`). Never fall back to "any cached file" for answer-based metrics.
- `ResultManager.save` writes atomically (temp file + rename): many concurrent jobs share caches.

## Open design decisions (test cantons)
- Partisan Phase 2 (`launch_partisan_sweep.sh`) uses the ZH Phase 1 question selection by default
  (strict transfer); `PHASE1_CSV=` uses the canton's own Phase 1.
- Approximate clones re-select the top-5 correlated questions on each canton's own voters (rule transfer).
- Variance-based clone selectors (`combinedvar`, `highvotervar`, `highcandvar`) used national data
  (`district = "all"` in clone configs); set `district = "ZH"` there for strictly ZH-only selections.
