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
- Test cantons run ONE alpha (the one chosen on ZH, env `VQS_ALPHA`, no-op if the config already has it) and no
  alpha sweeps. All test cantons at once: `ALPHA=<zh alpha> bash jobs/cantons/run_all_cantons.sh` (submits a small
  orchestrator job; progress `experiment_results/cantons/_runs/<ts>/orchestrator.log`, `bash jobs/cantons/status.sh`).
  BE is first and has priority (others wait for BE's same experiment, then use `--nice`); memory per job is scaled from
  BE's measured peak (sacct MaxRSS) by voters x candidates; 1 CPU per job (the ranking code is single-threaded).
- One-seat cantons (AR AI GL NW OW UR) are excluded: 1–3 candidates make ranking metrics meaningless.

## Caching rules
- Cache keys (`ResultManager`, MD5 of config params) for text-embedding metrics are canton-independent
  (question text only). Answer-based metrics (`ANSWER_BASED_METRICS`) add `ANSWER_METRIC_HASH_PARAMS`
  (incl. `district`) at every stage via `respondent_hash_params(config)` — keep this when adding stages.
- Never re-implement the hash in scripts; build it from the same param lists (see
  `distance_structure_analysis._compute_dist_hash`). Never fall back to "any cached file" for answer-based metrics.
- `ResultManager.save` writes atomically (temp file + rename): many concurrent jobs share caches.
- Base-side (un-cloned) baseline/CRW recommendations are cached on disk once per canton
  (`RecommendationEngine.run_baseline_cached/run_crw_cached`, dir `VQS_REC_CACHE_DIR`, ~0.5-0.8 GB each, keep on
  scratch). Cloned-side tables are deliberately NOT cached (one per question x alpha x clone type = cache explosion).
  `ResultManager.save` drops the DataFrame index, but `analyze_from_dfs` joins on the voter index: cached tables
  must keep it (the engine helpers do).

## Open design decisions (test cantons)
- Partisan Phase 2 (`launch_partisan_sweep.sh`) uses the ZH Phase 1 question selection by default
  (strict transfer); `PHASE1_CSV=` uses the canton's own Phase 1.
- Approximate clones re-select the top-5 correlated questions on each canton's own voters (rule transfer).
- Variance-based clone selectors (`combinedvar`, `highvotervar`, `highcandvar`) used national data
  (`district = "all"` in clone configs); set `district = "ZH"` there for strictly ZH-only selections.

## Known naming pitfalls (to be fixed in the future)
Run/file/folder names are unintuitive; do not read them literally:
- Names come from the config *file* name plus override suffixes, not from the data. Test-canton runs reuse
  the ZH config, so Bern outputs are called e.g. `recs_pipeline_e5_instruct_ZH_district~BE_alpha~0.4_*` or
  `pipeline_e5_instruct_ZH_a03_districtBE_alpha0.4`: "ZH" = the config the parameters were chosen on, the data
  is BE (see `district~BE` / `districtBE` and the metadata JSON `overrides`).
- "a03" in a config name is only the filename; the alpha actually used is the one in the `alpha~`/`alpha0.4`
  suffix (`VQS_ALPHA` overrides the config). Partisan runs on test cantons use the `_a03` config at alpha 0.4:
  compare against ZH's `..._ZH_a04` results, never `..._ZH_a03`.
- Labels `n5` / "x 5" in clone-sweep names/reports are cosmetic; the real clone count is 4.
- "recommendations" (`pipeline_outputs/recommendations/`, `recs_*`) holds the plain AND CRW rankings per voter.
- `experiment_results/cantons/<C>/handover/` holds cleanly renamed copies (not the originals) for sharing.
