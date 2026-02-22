# Toward Robust Voting Advice Applications: Lessons from Smartvote

This repository contains all code written for the master's thesis "Toward Robust Voting Advice Applications: Lessons from Smartvote" by [Dustin Brunner](https://www.linkedin.com/in/dustinbrunner/). 
The thesis was written in Spring 2024 at the [Distributed Computing Group (DISCO)](https://disco.ethz.ch/) from ETH Zürich.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Building the Data](#1-building-the-data)
  - [2. Generating, Storing \& Loading Recommendations](#2-generating-storing--loading-recommendations)
    - [2.1 Candidate Recommendations](#21-candidate-recommendations)
    - [2.2 List Recommendations](#22-list-recommendations)
  - [3. Calculating Voter \& Candidate Metrics](#3-calculating-voter--candidate-metrics)
  - [4. Reproducing Plots \& Tables](#4-reproducing-plots--tables)

## Introduction

This project explores methods to enhance the robustness of VAAs against manipulation. We analyze different distance metrics and recommendation strategies using the Smartvote datasets from 2019 and 2023. The analysis is documented in various Jupyter notebooks, and the core functionality is implemented in Python modules.

## Project Structure

Here’s a detailed explanation of the folder structure and the purpose of each file:

```bash
recommender-systems-for-politics/  # Root directory
├── rsfp/  # Main project code
│   ├── data.py                    # SVDataFrame class, functions for building dataframes, cleaning, and preprocessing
│   ├── matching.py                # Candidate & List Recommendation functions, includes all evaluated distance metrics
│   ├── constants.py               # Important constants, mappings from IDs to names, etc.
│   ├── utils.py                   # General utility functions used throughout the project
│   ├── dimensionality_reduction.py # Functions for CA, PCA, TSVD dimensionality reduction
│   └── metrics.py                 # Rank correlation metric functions
│
├── Analysis.ipynb                 # Exploratory Data Analysis (EDA) notebook
├── Cleaning.ipynb                 # Data preprocessing & cleaning analysis
├── Matching.ipynb                 # Recommendation computation notebook
├── DistanceMethods.ipynb          # Visualizations of distance metrics
├── MethodComparison.ipynb         # Comparison of distance metrics
├── Vulnerabilities.ipynb          # Vulnerability analysis & plots
├── WhatIfAnalysis.ipynb           # Answer Calibration & Question Favoritism What-If analyses
├── AnswerOptimization.ipynb       # Answer Optimization, finding crafted candidates
├── Shift.ipynb                    # Analysis of shifts between 2019 and 2023 elections
│
├── images/                        # Plots generated for the report and paper
│
├── data/                          # All input data & generated cached data
│   ├── cache/                     # Folder used for caching precomputed voting recommendations and other files
│   ├── smart vote data/           # Smartvote data from 2019
│   ├── sv23_ETHZ/                 # Smartvote data from 2023
│   └── NRW2023-kandidierende.json # Election vote data for 2023
│
├── answer_optimization/           # Optimized crafted candidates for all cantons
└── answer_optimization_script.py  # Script for optimizing crafted candidates in all cantons
```

### Required Files
To ensure that everything works correctly in this project, the following files must be present in the specified directories:

1. **2019 Data Files:**
   - Voter Data: `data/smart vote data/sv_Voter_1xNR_V1_0_ethz.csv`
   - Candidate Data: `data/smart vote data/smartvote_2019_Candidates_NR.csv`
   - Questions Data: `data/smart vote data/smartvote_2019_NR_Questions.csv`

2. **2023 Data Files:**
   - Voter Data: `data/sv23_ETHZ/sv23 Voters-NR 2024-03-14.csv`
   - Corrected Timestamp Data: `data/sv23_ETHZ/sv23 Voters-NR_time_recDATE.csv`
   - Candidate Data: `data/sv23_ETHZ/23_ch_nr_candidates_de_2024_03_06.csv`
   - Questions Data: `data/sv23_ETHZ/23_ch_nr-questions_de-fr-it-en.xlsx`

3. **2023 Election Data:**
   - Election Vote Data: `data/NRW2023-kandidierende.json` (from https://opendata.swiss/de/dataset/eidg-wahlen-2023/resource/1cd03e48-bb87-4d89-825b-84ccd32a0b83, slightly changed over time, best to use version in repository)

## Installation

1. Clone the repository and change directory. This step may take 5-10 minutes depending on your internet connection as the datasets included in the repository are quite large (~4 GiB).
```bash
git clone https://gitlab.ethz.ch/disco-students/fs24/recommender-systems-for-politics.git
cd recommender-systems-for-politics
```
2. Create conda environment and activate it
```bash
conda env create -n rsfp python=3.12
conda activate rsfp
```
3. Install the dependencies using pip
```bash
pip install -r requirements.txt
```

**Remark:** `requirements.txt` contains only the most important packages in order to be easily installable. The full snapshot of the original environment is in `requirements-full.txt`.
If you ever run into a `ModuleNotFoundError`, you can install the missing package using `pip install <package-name>`.
There is also a detailed list of dependencies at the start of the `Analysis.ipynb` notebook.

4. a.) Start Jupyter Lab to run any notebook
```bash
jupyter lab
```
4. b.) Run scripts directly from the command line (after adjusting hyperparameters, etc.)
```bash
python answer_optimization_script.py
```

## Usage

### 1. Building the Data

To build the data you require the Smartvote datasets. The datasets should be placed in the `data/smart vote data` and `data/sv23_ETHZ` folders and the files should be named as indicated in [Required Files](#required-files). Alternatively you can also adjust the file paths in the `constants.py` module.

You can use the `build_all`, `build_voters` and `build_candidates` (2023) functions, or the `build_all19`, `build_voters19` and `build_candidates19` (2019) functions from the `data.py` module, to build the voter and candidate dataframes of the respective years.

```python
from rsfp.data import build_all, build_all19

# 2023 datasets
df_voters, df_candidates, df_questions = build_all(verbose=True)

# 2019 datasets
df_voters19, df_candidates19, df_questions19 = build_all19(verbose=True)
```

### 2. Generating, Storing & Loading Recommendations

##### 2.1 Candidate Recommendations

To generate candidate recommendations, you can use the `add_candidate_voting_recommendations` function from the `matching.py` module.

To save the candidate recommendations to cache, you can use the `save_candidate_voting_recommendations` method from the `SVDataFrame` class.
To load the candidate recommendations from cache, you can use the `load_candidate_voting_recommendations` method from the `SVDataFrame` class.

To count the number of recommendations each candidate received, you can use the `add_recommendation_counts` method from the `SVDataFrame` class to which you can pass the voter dataframe with the added candidate recommendations.
To save and load the recommendation counts you can use the `load_candidate_recommendation_counts` and `save_candidate_recommendation_counts` methods from the `SVDataFrame` class.

```python
from rsfp.matching import add_candidate_voting_recommendations

# Add candidate voting recommendations for 2023
df_voters = add_candidate_voting_recommendations(df_voters, df_candidates, distance_method='L2_sv')
# Add candidate voting recommendations for 2019
df_voters19 = add_candidate_voting_recommendations(df_voters19, df_candidates19, distance_method='L2_sv')

# save candidate voting recommendations to cache
df_voters.save_candidate_voting_recommendations('df_voters_candidate_recommendations')
# load candidate voting recommendations from cache
df_voters = df_voters.load_candidate_voting_recommendations('df_voters_candidate_recommendations')

# Add candidate recommendation counts
df_candidates = df_candidates.add_recommendation_counts(df_voters)
df_candidates19 = df_candidates19.add_recommendation_counts(df_voters19)

# save candidate recommendation counts to cache
df_candidates.save_candidate_recommendation_counts('df_candidates_recommendation_counts')
# load candidate recommendation counts from cache
df_candidates = df_candidates.load_candidate_recommendation_counts('df_candidates_recommendation_counts')
```

The candidate voting recommendations will be stored in the following columns of `df_voters`:
- `_matchID_{rank}_{distance_metric}`: The ID of the candidate recommended at rank `{rank}`, with ranks ranging from 1 to the number of seats in the given canton, using distance metric `{distance_metric}`.
- `_matchDist_{rank}_{distance_metric}`: The distance of the recommended candidate at rank `{rank}` using distance metric `{distance_metric}`.

The candidate recommendation counts will be stored in the following columns of `df_candidates`:
- `_n_recommendations_spc_{distance_metric}`: The number of voters to which the candidate was recommended in any of the top-seats ranks (spc: seats-per-canton).
- `_n_recommendations_top_{distance_metric}`: The number of voters to which the candidate was recommended in the top rank.
- `_n_recommendations_vis_{distance_metric}`: The fraction of voters (in percent) to which the candidate was recommended in any of the top-seats ranks (**visibility**).
- `_n_recommendations_spc_{distance_metric}_normalized`: The expectation-normalized number of spc recommendations a candidate received.
- `_n_recommendations_top_{distance_metric}_normalized`: The expectation-normalized number of top recommendations a candidate received.

##### 2.1 List Recommendations

To generate list recommendations, you can use the `add_list_voting_recommendations` function from the `matching.py` module.

To save the list recommendations to cache, you can use the `save_list_voting_recommendations` method from the `SVDataFrame` class.
To load the list recommendations from cache, you can use the `load_list_voting_recommendations` method from the `SVDataFrame` class.

```python
from rsfp.matching import add_list_voting_recommendations

# Add list voting recommendations for 2023
df_voters = add_list_voting_recommendations(df_voters, df_candidates, distance_method='L2_sv')
# Add list voting recommendations for 2019
df_voters19 = add_list_voting_recommendations(df_voters19, df_candidates19, distance_method='L2_sv')

# save list voting recommendations to cache
df_voters.save_list_voting_recommendations('df_voters_list_recommendations')
# load list voting recommendations from cache
df_voters = df_voters.load_list_voting_recommendations('df_voters_list_recommendations')
```

The list voting recommendations are in the following columns in `df_voters`:
- `_ListID_{rank}_{distance_metric}`: The ID of the list recommended at rank `{rank}`, with ranks ranging from 1 to the number of lists in the given canton, using distance metric `{distance_metric}`.
- `_ListDist_{rank}_{distance_metric}`: The distance to the list recommendation at rank `{rank}` using distance metric `{distance_metric}`.
- `_ListParty_{rank}_{distance_metric}`: After running `df_voters.map_lists_to_parties()` this column contains the corresponding party of the list recommended at rank `{rank}` using distance metric `{distance_metric}`.


### 3. Calculating Voter & Candidate Metrics

To calculate the metrics regarding the candidate recommendation counts (answer strength - recommendation correlation, gini), you can use the `get_recommendation_metrics` method from the `SVDataFrame` class.
For this method to work you need to have added the candidate recommendation counts to the candidate dataframe.

```python
# calculate candidate recommendation count metrics
df_candidate_metrics = df_candidates.get_recommendation_metrics()
```

To calculate the metrics regarding the voting recommendations (party match accuracy, disagreement count), you can use the `get_voter_metrics` method from the `SVDataFrame` class.
For this method to work you need to have added the list voting recommendations to the voter dataframe.

```python
# calculate list voting recommendation metrics
df_voter_metrics = df_voters.get_voter_metrics()
```

### 4. Reproducing Plots & Tables

To reproduce plots from the paper, you can use the Jupyter notebooks in the root directory.
Most of the plots are in the `Vulnerabilities.ipynb` notebook, with some of them being in the `Analysis.ipynb`, `WhatIfAnalysis.ipynb` or `DistanceMethods.ipynb` notebooks.
The notebooks are generally structured similarly to the paper with section names corresponding to the paper sections.

The steps to reproduce a given plot in a notebook are usually the following:
1. Run the cells in the `Imports` section
2. Run the necessary cells in the `Data` section (building data, loading cached recommendations, etc.)
3. Jump to the corresponding section for the plot you want to reproduce and try to run the cell that generates the plot. (Copy & pasting it will allow you to verify if you got the same output as the cell generated previously)
4. If there are errors indicating missing variables the best bet is searching for the cells where they are defined and tracing your way back in this way.
5. As soon as you have been able to regenerate the figure, you can store the plot to the `images/` folder using the `save_paper_figure` method from the `utils.py` module. This method also accepts additional layout arguments to adjust the figure size, margins, etc.

To reproduce the tables from the paper, you can use the `MethodComparison.ipynb` notebook. The notebook contains the code to calculate the metrics used to evaluate the distance metrics.

