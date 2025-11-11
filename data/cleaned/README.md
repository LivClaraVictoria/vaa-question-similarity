# Data Preprocessing

The data in this folder was generated using the code from Dustin Brunner's repository.

His project is not integrated into this one. Instead, it was run as a standalone tool to produce the necessary clean data file.

### Reproducibility Steps

To regenerate the cleaned data files:

1.  **Clone the source repository:**
    `git clone https://gitlab.ethz.ch/disco-students/fs24/recommender-systems-for-politics.git`


2.  **Set up the environment:**
    A separate conda environment was created using `requirements.txt` from his repo.
    `cd recommender-systems-for-politics`
    `conda create -n rsfp python=3.12`
    `cconda activate rsfp`
    `pip install -r requirements.txt`

3.  **Modify Configuration:**
    The file `rsfp/constants.py` was temporarily modified to point to this project's data:
    * `DATA_FOLDER` was set to `vaa-question-similarity/data/raw`
    * The output save path (CACHE_FOLDER) was set to `vaa-question-similarity/data/cleaned/`

4.  **Run the Script:**
    From his repository's directory, the following command was run:
    `python -c 'from rsfp.data import build_all, build_all19; print("--- Starting data build for 2019 ---"); build_all19(verbose=True); print("--- 2019 data build complete ---"); print("\n--- Starting data build for 2023 ---"); build_all(verbose=True); print("\n All data has been processed and saved.")'`

This process was run once on [Date, e.g., 2025-11-06]. The resulting output file is committed to this repository for use by the main analysis code.