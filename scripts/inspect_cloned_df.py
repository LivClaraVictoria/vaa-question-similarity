"""
Quick inspection of a cloned questions dataframe.
Usage: python scripts/inspect_cloned_questions.py
"""

import sys
from pathlib import Path

# Make sure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from configs.base_constants import DATA_DIR

# --- CONFIG ---
clone_dir_name = "identical_q32214_n10"
source_q_id = 32214
# --------------

questions_path = DATA_DIR / "cloned" / clone_dir_name / "df_questions.parquet"
df = pd.read_parquet(questions_path)

print(f"Total questions: {len(df)}")
print(f"\n--- Source question ({source_q_id}) ---")
print(
    df[df["ID_question"] == source_q_id][
        ["ID_question", "question_EN", "type", "_n_options"]
    ].to_string()
)

print(f"\n--- Clone rows ---")
clone_mask = df["ID_question"] > 9_000_000
print(df[clone_mask][["ID_question", "question_EN", "type", "_n_options"]].to_string())
