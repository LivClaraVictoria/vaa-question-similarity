# VAA Question Similarity & Clone-Robust Weighting

This repository contains the codebase for my Bachelor Thesis analyzing semantic similarity between political questions in the Swiss Voting Advice Application (VAA), SmartVote. 

**Core Research Question:** How do voter-candidate recommendations change when identical or near-identical questions are added to the VAA questionnaire, and can Clone-Robust Weighting (CRW) correct this distortion?

## Tech Stack
* **Language:** Python 3.12 
* **Embeddings:** `sentence-transformers` (SBERT, E5 multilingual, Jina v3, etc.)
* **Paraphrase Generation:** `openai` (GPT-4o for approximate clones)
* **Data Processing:** `pandas`, `numpy`, `scipy`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`, `plotly`
* **External Dependency:** `dependencies/rsfp/` (git submodule for VAA recommender systems)

---

## Installation & Setup

To run this project locally, it is recommended to use Conda to manage your Python environment and dependencies.

```bash
# 1. Create a new conda environment (you can replace `vqs-env` with your preferred name)
conda create --name vqs-env python=3.12 -y

# 2. Activate the environment
conda activate vqs-env

# 3. Install the required dependencies
pip install -r requirements.txt
```

## Repository Structure

The codebase is organized into core infrastructure, experiment scripts, and configuration:

* `vqs/`: Core library (distance metrics, CRW algorithm, recommendation engine).
* `clone_pipeline/`: Synthetic clone generation and LLM paraphrase caching.
* `cross_run_analysis/`: Tools to compare baseline vs. CRW-weighted pipeline runs.
* `experiments/`: Executable scripts for thesis experiments (divided by narrative chapters).
* `configs/`: Python-based inheritance configuration files.
* `jobs/`: SLURM batch scripts for cluster execution.
* `cache/`: Hash-based caching for expensive computations.
* `experiment_results/`: Human-readable outputs, plots, and summaries.

---

## Core Infrastructure & Entry Points

All scripts should be executed as Python modules (e.g., `python -m <module>`) from the project root to ensure local imports resolve correctly.

| Script | Purpose | Example Execution |
| :--- | :--- | :--- |
| `main.py` | Runs the full pipeline: distances → CRW weights → recommendations. | `python -m main --config configs/full_pipeline/base_data/pipeline_e5_ZH.py` |
| `clone_main.py` | Generates synthetic cloned question datasets. | `python -m clone_main --config configs/create_clones/identical_q32214_n10.py` |
| `comparator_main.py` | Compares two pipeline runs and generates metrics. | `python -m comparator_main <run_a.parquet> <run_b.parquet>` |

---

## The Experiments

The thesis narrative is built on three primary experiments and subsequent analyses. Below is a guide on where to find them and how to execute them.

### Experiment 1: Clone Detection (Primary Result)
Tests whether CRW can detect and correct for synthetic clones (identical, paraphrased, and negated). This involves sweeping the CRW `alpha` parameter across 10 embedding models.

* **Location:** `experiments/perfect_clones/rec_change/`
* **Key Script:** `alpha_sweep.py` evaluates CRW robustness by running the full comparison pipeline across a range of alpha values.
* **Execution:**
    `python -m experiments.perfect_clones.model_selection --config_a configs/full_pipeline/base_data/pipeline_e5_ZH.py --config_b configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py`
* **Outputs:** CSVs and plots showing Jaccard/Spearman metrics vs. alpha, saved to `experiment_results/exp1/model_alpha_sweep/`.

### Experiment 2: Question Removal (Negative Result)
Tests whether CRW compensates for missing/removed questions (underrepresentation). 
* **Location:** `experiments/question_removal/`
* **Key Script:** `question_removal.py`
* **Result:** Demonstrates that CRW downweights dense clusters but does not effectively upweight sparse ones. Defines the scope limitation of the algorithm.
* **Execution:**
    `python -m experiments.abandoned.question_removal`

### Experiment 3: Natural Redundancy (Approximate Clones)
Tests whether CRW helps when naturally correlated (but textually distinct) questions are added from the full questionnaire to a mini questionnaire.
* **Location:** `experiments/approximate_clones/`
* **Key Script:** `recommendation_distortion.py` and `partisan_distortion.py`
* **Result:** Shows CRW is built for "clone detection" (near-identical distance) and cannot correct for natural topic overrepresentation without textual similarity.
* **Execution:**
    `python -m experiments.approximate_clones.recommendation_distortion`

### Party Impact Analysis
Analyzes which political parties gain or lose visibility when specific questions are cloned, and simulates strategic VAA attacks.
* **Location:** `experiments/perfect_clones/party_impact/`
* **Key Script:** `party_impact.py` (Phase 1 evaluates single-question impact; Phase 2 tests cumulative CRW correction).
* **Execution (Phase 2):**
    `python -m experiments.perfect_clones.partisan_distortion --mode phase2 --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py --target-party Centre`

### Distance & Correlation Metric Analysis
Validates embedding models against actual voter answer correlations to prove that while CRW works for textual similarity, embedding distances do not inherently capture functional redundancy.
* **Location:** `experiments/explanatory/distances/`
* **Key Script:** `distance_structure_analysis.py`
* **Execution:**
    `python -m experiments.explanatory.distances.distance_structure_analysis --config configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py`

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

**IMPORTANT:** Any data containing voter IDs, candidate IDs, or data linked to individual voters/candidates is strictly confidential. Only question data (texts, IDs) is tracked in version control. Do not push raw `SmartVote` voter parquets to this repository.