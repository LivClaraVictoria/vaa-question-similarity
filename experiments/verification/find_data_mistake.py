import pandas as pd
import numpy as np
import os
from configs.base_constants import CLEANED_DIR
from dependencies import (
    DATA_FOLDER,
    SV23_FOLDER,
    TIMESTAMP_FILE,
    CACHE_FOLDER,
    ID2PREF_PARTY,
    ID2EDUCATION,
    ID2DISTRICT,
    ID2LANGUAGE,
    get_cols,
    count_consecutive_values,
)

try:
    df_voters = pd.read_parquet(CLEANED_DIR / "df_voters_topmatch.parquet")
    print(f"STARTING COUNT: {len(df_voters)} rows")
except Exception as e:
    print(f"Error loading parquet file: {e}")

OG_linecount = len(df_voters)
before = OG_linecount

# empty strings -> nan
df_voters.replace(" ", np.nan, inplace=True)
print(
    f"0. Replacing empty strings with NaN: {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)


before = len(df_voters)
# drop redundant age column
df_voters.drop(columns=["age"], inplace=True)
print(f"0. Dropping age col: {len(df_voters)} (Lost: {before - len(df_voters)})")

before = len(df_voters)
# remove corrupted recommendations with no electionID
df_voters.dropna(subset=["electionID"], inplace=True)
print(
    f"1. After dropping missing electionID: {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)

before = len(df_voters)
# unrealistic birthYEAR -> nan
df_voters.loc[
    ((df_voters["birthYEAR"] < 1900) | (df_voters["birthYEAR"] > 2005)), "birthYEAR"
] = np.nan


before = len(df_voters)
# irregular zip -> nan
df_voters.loc[((df_voters["zip"] > 9658) | (df_voters["zip"] < 1000)), "zip"] = np.nan

before = len(df_voters)
# only keep recommendations from deluxe questionnaire with at least 60 answers and at most 75
df_voters = df_voters[(df_voters["N_answers"] >= 15) & (df_voters["N_answers"] <= 75)]
print(
    f"2. After N_answers filter (15-75):    {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)


before = len(df_voters)
# add time column
df_time = pd.read_csv(os.path.join(DATA_FOLDER, SV23_FOLDER, TIMESTAMP_FILE))
# add column _time in correct datetime format:
df_time["_time"] = pd.to_datetime(df_time["recTIME_REC"])
# add correct timestamps to df_voters by merging on recID
df_voters = df_voters.merge(df_time[["recID", "_time"]], on="recID")
df_voters.drop(columns="recTIME", inplace=True)
print(
    f"3. After merging time column: {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)

before = len(df_voters)
# remove recommendations before publishing and after election date
df_voters = df_voters[
    ("2023-08-23" <= df_voters["_time"]) & (df_voters["_time"] < "2023-10-23")
]
print(
    f"4. After date filter (2023-08-23 to 2023-10-23): {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)

before = len(df_voters)
# remove recommendations with more than 14 consecutive equal answers
answer_cols = get_cols(df_voters, col_type="answer")
df_voters = df_voters[count_consecutive_values(df_voters[answer_cols].values) <= 14]
print(
    f"5. After removing >14 consecutive equal answers: {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)

before = len(df_voters)
# remove duplicate voter_ids
df_voters = df_voters.sort_values(
    by=["N_answers", "_time", "recID"], ascending=False
).drop_duplicates(subset=["voterID"], keep="first")
print(
    f"6. After removing duplicate voter_ids: {len(df_voters)} (Lost: {before - len(df_voters)} - {(before - len(df_voters)) / before * 100:.2f}%)"
)

print("\n----FINAL CLEANING STATS:----")
print(f"TOTAL ROWS LOST: {OG_linecount - len(df_voters)}")

# before = len(df_voters)
# # add decoded columns
# decoded_columns = {
#     "_party": df_voters["pref_party"].astype("float64").map(ID2PREF_PARTY),
#     "_education": df_voters["education"].map(ID2EDUCATION),
#     "_district": df_voters["districtID"].map(ID2DISTRICT),
#     "_language": df_voters["language"].map(ID2LANGUAGE),
# }
# df_decoded = pd.concat(decoded_columns, axis=1)

# # merge decoded columns with original DataFrame
# df_voters = pd.concat([df_voters, df_decoded], axis=1)

# # add stats columns
# df_voters["_answer_strength"] = (df_voters[answer_cols] - 50).abs().mean(axis=1)
# df_voters["_answer_strength_std"] = (df_voters[answer_cols] - 50).abs().std(axis=1)

# weight_cols = get_cols(df_voters, col_type="weight")
# df_voters["_maxDist_L2_sv"] = np.sqrt(((df_voters[weight_cols] * 100) ** 2).sum(axis=1))

# max_diff = df_voters[answer_cols].map(lambda x: max(x, 100 - x)).to_numpy()
# weights = df_voters[weight_cols].to_numpy()
# df_voters["_maxDistCorrect_L2_sv"] = np.sqrt(
#     np.nansum((max_diff * weights) ** 2, axis=1)
# )

# # reset index
# df_voters = df_voters.reset_index(drop=True)
