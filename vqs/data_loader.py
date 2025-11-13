# # Conceptual preview - don't implement yet
# def load_parquet_by_prefix(directory, prefix):
#     # Find any file that starts with "df_voters19" and ends with ".parquet"
#     files = list(directory.glob(f"{prefix}*.parquet"))

#     if len(files) == 0:
#         raise FileNotFoundError(f"No file found starting with {prefix}")
#     if len(files) > 1:
#         print(f"Warning: Multiple files found for {prefix}, taking the first one.")

#     return pd.read_parquet(files[0])
