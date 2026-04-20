# VAA Question Similarity & Clone-Robust Weighting

This repository contains the codebase for my Bachelor Thesis analyzing semantic similarity between political questions in the Swiss Voting Advice Application (VAA), SmartVote.

**Core Research Question:** How do voter-candidate recommendations change when identical or near-identical questions are added to the VAA questionnaire, and can Clone-Robust Weighting (CRW) correct this distortion?

## Tech Stack
* **Language:** Python 3.12
* **Embeddings:** `sentence-transformers` (SBERT, E5 multilingual, Jina v3, etc.)
* **Paraphrase Generation:** `openai` (GPT-4o for approximate clones)
* **Data Processing:** `pandas`, `numpy`, `scipy`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`, `plotly`
* **External Dependency:** `dependencies/rsfp/` (see below)

---

## External Dependency

`dependencies/rsfp/` is based on the code from Dustin Brunner's master thesis ["Toward Robust Voting Advice Applications: Lessons from Smartvote"](https://gitlab.ethz.ch/disco-students/fs24/recommender-systems-for-politics). The code was incorporated as a full subfolder rather than a git submodule. Minor compatibility changes were applied (no structural changes): imports were converted to relative imports. See `dependencies/README.md` for details.

---

## Installation & Setup

To run this project locally, it is recommended to use Conda to manage your Python environment and dependencies.

```bash
# 1. Create a new conda environment
conda create --name vqs-env python=3.12 -y

# 2. Activate the environment
conda activate vqs-env

# 3. Install the required dependencies
pip install -r requirements.txt
```

---

## Repository Structure

The codebase is organized into core infrastructure, experiment scripts, and configuration:

* `vqs/`: Core library (distance metrics, CRW algorithm, recommendation engine).
* `clone_pipeline/`: Synthetic clone generation and LLM paraphrase caching.
* `cross_run_analysis/`: Tools to compare baseline vs. CRW-weighted pipeline runs.
* `experiments/`: Executable scripts for thesis experiments, organized by narrative:
  * `perfect_clones/`: Recommendation and partisan distortion experiments using synthetic clones.
  * `approximate_clones/`: Recommendation and partisan distortion experiments using correlated but textually distinct questions.
  * `explanatory/`: Supporting analyses (distance structure, question impact, model benchmark, etc.).
  * `abandoned/`: Experiments producing null or negative results.
  * `verification/`: One-off correctness checks and unit tests.
* `configs/`: Python-based inheritance configuration files.
* `jobs/`: SLURM batch scripts for cluster execution.
* `cache/`: Hash-based caching for expensive computations.
* `experiment_results/`: Human-readable outputs, plots, and summaries.

---

## Core Infrastructure & Entry Points

All scripts should be executed as Python modules (e.g., `python -m <module>`) from the project root to ensure local imports resolve correctly.

| Script | Purpose | Example Execution |
| :--- | :--- | :--- |
| `main.py` | Runs the full pipeline: distances -> CRW weights -> recommendations. | `python -m main --config configs/full_pipeline/base_data/pipeline_e5_ZH.py` |
| `clone_main.py` | Generates synthetic cloned question datasets. | `python -m clone_main --config configs/create_clones/identical_q32214_n10.py` |
| `comparator_main.py` | Compares two pipeline runs and generates metrics. | `python -m comparator_main <run_a.parquet> <run_b.parquet>` |

---

## The Experiments

The thesis narrative is built on two primary experiments and subsequent analyses. Below is a guide on where to find them and how to execute them.

### Step 0: Model Selection

Determines the optimal embedding model for all subsequent experiments by running the full alpha sweep across all 10 embedding models and 5 clone types. The selected model is then used as the fixed distance metric for Experiments 1 and 2.

**Location:** `experiments/perfect_clones/model_selection.py`

```bash
python -m experiments.perfect_clones.model_selection \
    --config_a configs/full_pipeline/base_data/pipeline_e5_ZH.py \
    --config_b configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py
```

---

### Experiment 1: Perfect Clone Experiments

Tests CRW's ability to detect and correct synthetic clones (identical copies, paraphrases, negations) of questionnaire questions.

#### Recommendation Distortion

Clones each of the 75 questions under 5 different clone conditions (identical, easy/hard paraphrase, negation variants), yielding 375 cloned datasets. For each, runs a CRW alpha sweep using the optimal embedding model (E5-INSTRUCT) and measures per-voter recommendation change (Jaccard, Spearman, Kendall).

**Location:** `experiments/perfect_clones/`

Key scripts:
* `recommendation_distortion.py`: 75-question x 5-clone-type x alpha sweep.
* `clone_count_sweep.py`: alpha x clone-count sweep for a single question.

```bash
python -m experiments.perfect_clones.model_selection \
    --config_a configs/full_pipeline/base_data/pipeline_e5_ZH.py \
    --config_b configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py
```

#### Partisan Distortion

Measures how cloning specific questions shifts party visibility across voter recommendations. Phase 1 sweeps all 75 questions individually to identify which questions most benefit which parties. Phase 2 applies cumulative CRW correction for the top-k questions benefiting a target party.

**Location:** `experiments/perfect_clones/partisan_distortion.py`

```bash
# Phase 1 (full SLURM sweep):
bash jobs/launch_party_impact.sh

# Phase 2 (requires existing Phase 1 CSV):
python -m experiments.perfect_clones.partisan_distortion \
    --mode phase2 \
    --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \
    --target-party Centre \
    --phase1-csv <path-to-phase1-csv>
```

---

### Experiment 2: Approximate Clone Experiments

Tests CRW on questions that are correlated by voter answer patterns but textually distinct. Adds the top-5 most answer-correlated full-questionnaire questions to the mini (rapide) questionnaire and measures the resulting recommendation and party-visibility distortion across CRW alpha values.

**Location:** `experiments/approximate_clones/`

#### Recommendation Distortion

```bash
python -m experiments.approximate_clones.recommendation_distortion
```

#### Partisan Distortion

```bash
python -m experiments.approximate_clones.partisan_distortion
```

---

### Explanatory Experiments

Supporting analyses that contextualize the primary results.

**Location:** `experiments/explanatory/`

Key scripts:
* `model_benchmark.py`: evaluates 10 embedding models on a synthetic benchmark dataset, providing an explanation of why certain models perform better in the alpha sweep.
  ```bash
  # Run each model on the fake benchmark dataset first (one config per model):
  python -m main --config configs/distance_method/fake/fake_e5_instruct.py
  # Repeat for each config in configs/distance_method/fake/, then evaluate:
  python -m experiments.explanatory.model_benchmark
  ```
* `distances/distance_structure_analysis.py`: analyzes the correlation structure of pairwise question distances, validates embedding models against voter-answer correlation ground truth, and compares within-topic vs. cross-topic distance distributions.
  ```bash
  python -m experiments.explanatory.distances.distance_structure_analysis --section 2,3
  ```
* `question_impact.py`: sweeps all 75 questions, measuring how much each one distorts recommendations when cloned. Used to identify the top-5 most impactful questions for Experiment 1.
  ```bash
  python -m experiments.explanatory.question_impact \
      --config configs/full_pipeline/base_data/pipeline_e5_ZH.py
  ```
* `approximate_clones_analysis/`: distance-level analyses for the approximate clone setup (distance distributions, forbidden intervals).
* Additional scripts: `category_analysis.py`, `recommendation_metric_agreement.py`, `compare_recommendation_and_party_distortion.py`, etc.

---

### Abandoned Experiments

**Location:** `experiments/abandoned/`

Experiments retained for completeness but not part of the thesis narrative.

* `question_removal.py`: tested whether CRW can compensate for removed questions (underrepresentation of a topic). CRW's mechanism is downweighting dense clusters; meaningfully upweighting sparse topics is beyond its design.

---

### Verification

**Location:** `experiments/verification/`

One-off scripts verifying correct behavior of individual pipeline components: `verify_clone_identity.py`, `verify_crw_pipeline.py`, `test_result_manager.py`, etc.

---

## Configuration System

The pipeline uses a Python-based inheritance system for configuration.
* Base defaults are defined in `configs/base_constants.py`.
* Specific runs inherit from the base and override parameters (e.g., `data_year`, `alpha`, `dist`, `embedding_instruction`).
* You can pass CLI overrides to `main.py` directly:
    `python -m main --config configs/full_pipeline/base_data/pipeline_e5_ZH.py alpha=0.8 data_year=2019`

---

## Executing on SLURM Cluster

For heavy workloads (alpha sweeps, parallel pipeline runs), use the provided `sbatch` scripts in the `jobs/` directory. The architecture uses a launcher + generic worker pattern to maximize parallelism.

**Common SLURM Workflows:**
* **Question Impact Sweep (Parallel):**
    `bash jobs/launch_question_impact.sh`
* **Party Impact Analysis (Phase 1 -> Phase 2):**
    `bash jobs/launch_party_impact.sh`
* **Alpha Sweep:**
    `bash jobs/launch_alpha_sweep_combinedvar.sh`

**Note on Shared File System:** Cache files (`cache/`) are written to a shared NFS. To prevent race conditions during paraphrase generation via the OpenAI API, clone creation jobs must be run in series.

---

## Data Confidentiality

**IMPORTANT:** Any data containing voter IDs, candidate IDs, or data linked to individual voters/candidates is strictly confidential. Only question data (texts, IDs) is tracked in version control. Do not push raw SmartVote voter parquets to this repository.
