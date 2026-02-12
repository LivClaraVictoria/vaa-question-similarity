from pathlib import Path
from networkx import config
import pandas as pd
from dependencies import SVDataFrame


# # Conceptual preview - don't implement yet
def load_parquet_by_prefix(directory, prefix):
    # Find any file that starts with "df_voters19" and ends with ".parquet"
    files = list(directory.glob(f"{prefix}*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError(f"No file found starting with {prefix}")
    if len(files) > 1:
        print(f"Warning: Multiple files found for {prefix}, taking the first one.")

    return pd.read_parquet(files[0])


def load_dataset(config) -> dict:
    data_map = {}  # voters, candidates, questions
    if config.data_choice == "cleaned":
        if config.data_year == "2023":
            q_path = config.QUESTIONS_2023_PATH
            v_prefix = config.VOTERS_PREFIX
            c_prefix = config.CANDIDATES_PREFIX
        elif config.data_year == "2019":
            q_path = config.QUESTIONS_2019_PATH
            v_prefix = config.VOTERS_19_PREFIX
            c_prefix = config.CANDIDATES_19_PREFIX
        else:
            raise ValueError(f"Unknown data_year: {config.data_year}")

        df_q = pd.read_parquet(q_path)
        data_map["questions"] = SVDataFrame(
            df_q, term=int(config.data_year)
        )  # type: ignore

        if config.load_voters:
            print(f"Loading voters data with prefix '{v_prefix}'...")
            data_map["voters"] = SVDataFrame(
                load_parquet_by_prefix(config.CLEANED_DIR, v_prefix),
                term=int(config.data_year),
            )  # type: ignore
        if config.load_candidates:
            print(f"Loading candidates data with prefix '{c_prefix}'...")
            data_map["candidates"] = SVDataFrame(
                load_parquet_by_prefix(config.CLEANED_DIR, c_prefix),
                term=int(config.data_year),
            )  # type: ignore

    elif config.data_choice == "fake":
        df_q = pd.read_csv(config.FAKE_DATA_FILE)
        data_map["questions"] = SVDataFrame(df_q)  # type: ignore

    else:
        raise NotImplementedError("Only cleaned and fake data implemented for now.")

    # Optional Canton Filtering
    if (
        config.filter_districts
        and config.data_choice != "fake"
        and ("voters" in data_map or "candidates" in data_map)
    ):
        cantonID_map = (
            config.DISTRICT2ID if config.data_year == "2023" else config.DISTRICT2ID19
        )
        target_id = cantonID_map.get(config.district)

        if target_id is None:
            print(
                f"Warning: District '{config.district}' not found for year {config.data_year}. Skipping filter."
            )
        else:
            print(f"Filtering data for district: {config.district} (ID: {target_id})")
            if "voters" in data_map:
                df = data_map["voters"]
                voter_district_col = (
                    "districtID" if config.data_year == "2023" else "ID_district"
                )
                data_map["voters"] = df[df[voter_district_col] == target_id].copy()

            # Filter Candidates (ID_election)
            if "candidates" in data_map:
                df = data_map["candidates"]
                data_map["candidates"] = df[df["ID_district"] == target_id].copy()

    # Optional subsetting for quick testing
    if hasattr(config, "subset_n") and config.subset_n is not None:
        print(f"!!! SANITY CHECK MODE: Subsetting data to {config.subset_n} rows !!!")
        if "voters" in data_map:
            data_map["voters"] = data_map["voters"].iloc[: config.subset_n]

    return data_map
