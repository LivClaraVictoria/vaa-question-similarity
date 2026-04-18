"""
Verify that cloned questions are byte-for-byte identical to their source questions.

Checks all text columns, answer columns, and weight columns. Reports any difference
including whitespace, punctuation, encoding artifacts, or Unicode normalization issues.

Usage:
    python -m scripts.verify_clone_identity --clone-id identical_combinedvar_n10
    python -m scripts.verify_clone_identity --clone-id identical_q32214_n10
    python -m scripts.verify_clone_identity  # runs all identical clone datasets
"""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

from clone_pipeline.spec import CLONE_ID_BASE


CLONED_DIR = Path("data/cloned")
TEXT_COLS = ["question_EN", "question_en"]  # 2023 and 2019 variants


def load_clone_metadata(clone_id: str) -> dict | None:
    meta_path = CLONED_DIR / clone_id / "clone_metadata.json"
    if not meta_path.exists():
        print(f"No metadata at {meta_path}")
        return None
    with open(meta_path) as f:
        return json.load(f)


def char_diff(a: str, b: str) -> str:
    """Show exactly where two strings differ."""
    if a == b:
        return "(identical)"
    diffs = []
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            diffs.append(
                f"  pos {i}: {repr(ca)} (U+{ord(ca):04X}) vs {repr(cb)} (U+{ord(cb):04X})"
            )
    if len(a) != len(b):
        diffs.append(f"  length: {len(a)} vs {len(b)}")
    return "\n".join(diffs) if diffs else f"(differ only in length: {len(a)} vs {len(b)})"


def check_text_col(df_questions: pd.DataFrame, source_id: int, clone_ids: list[int], col: str) -> bool:
    if col not in df_questions.columns:
        return True  # column doesn't exist for this year, skip
    source_row = df_questions[df_questions["ID_question"] == source_id]
    if source_row.empty:
        print(f"  ERROR: Source question {source_id} not found")
        return False
    source_text = source_row.iloc[0][col]
    all_ok = True
    for clone_id in clone_ids:
        clone_row = df_questions[df_questions["ID_question"] == clone_id]
        if clone_row.empty:
            print(f"  ERROR: Clone {clone_id} not found in questions df")
            all_ok = False
            continue
        clone_text = clone_row.iloc[0][col]
        if source_text == clone_text:
            pass  # exact match
        else:
            print(f"  MISMATCH in '{col}' for clone {clone_id}:")
            print(char_diff(source_text, clone_text))
            # Extra checks
            if source_text.strip() == clone_text.strip():
                print("    → Texts match after strip() — leading/trailing whitespace differs")
            if unicodedata.normalize("NFC", source_text) == unicodedata.normalize("NFC", clone_text):
                print("    → Texts match after Unicode NFC normalization")
            if unicodedata.normalize("NFD", source_text) == unicodedata.normalize("NFD", clone_text):
                print("    → Texts match after Unicode NFD normalization")
            all_ok = False
    return all_ok


def check_answer_col(df: pd.DataFrame, source_id: int, clone_ids: list[int], label: str, flip: bool) -> bool:
    src_col = f"answer_{source_id}"
    if src_col not in df.columns:
        print(f"  SKIP: column {src_col} not in {label}")
        return True
    source_vals = df[src_col]
    all_ok = True
    for clone_id in clone_ids:
        clone_col = f"answer_{clone_id}"
        if clone_col not in df.columns:
            print(f"  ERROR: {clone_col} not found in {label}")
            all_ok = False
            continue
        clone_vals = df[clone_col]
        expected = 100 - source_vals if flip else source_vals
        # NaN-safe comparison: NaN == NaN should be treated as equal
        mismatches = (~((clone_vals == expected) | (clone_vals.isna() & expected.isna()))).sum()
        nan_count = expected.isna().sum()
        if mismatches > 0:
            print(f"  MISMATCH: {mismatches} answer differences for clone {clone_id} in {label} "
                  f"(NaN rows correctly excluded: {nan_count})")
            all_ok = False
        elif nan_count > 0:
            pass  # NaN rows present but all correctly copied — reported in summary
    return all_ok


def check_weight_col(df_voters: pd.DataFrame, source_id: int, clone_ids: list[int]) -> bool:
    src_col = f"weight_{source_id}"
    if src_col not in df_voters.columns:
        print(f"  SKIP: {src_col} not in voters (no voter weights for this question)")
        return True
    source_vals = df_voters[src_col]
    all_ok = True
    for clone_id in clone_ids:
        clone_col = f"weight_{clone_id}"
        if clone_col not in df_voters.columns:
            print(f"  ERROR: {clone_col} not found in voters")
            all_ok = False
            continue
        clone_vals = df_voters[clone_col]
        mismatches = (~((clone_vals == source_vals) | (clone_vals.isna() & source_vals.isna()))).sum()
        nan_count = source_vals.isna().sum()
        if mismatches > 0:
            print(f"  MISMATCH: {mismatches} weight differences for clone {clone_id} in voters "
                  f"(NaN rows correctly excluded: {nan_count})")
            all_ok = False
    return all_ok


def verify_clone_identity(clone_id: str) -> None:
    print(f"\n{'='*70}")
    print(f"Verifying clone identity for: {clone_id}")
    print(f"{'='*70}")

    meta = load_clone_metadata(clone_id)
    if meta is None:
        return

    clone_dir = CLONED_DIR / clone_id

    # Load DataFrames
    questions_path = clone_dir / "df_questions.parquet"
    candidates_path = next(clone_dir.glob("df_candidates*.parquet"), None)
    voters_path = next(clone_dir.glob("df_voters*.parquet"), None)

    if not questions_path.exists():
        print(f"  ERROR: {questions_path} not found")
        return

    df_questions = pd.read_parquet(questions_path)
    df_candidates = pd.read_parquet(candidates_path) if candidates_path else None
    df_voters = pd.read_parquet(voters_path) if voters_path else None

    print(f"Questions df: {df_questions.shape[0]} rows")
    if df_candidates is not None:
        print(f"Candidates df: {df_candidates.shape[0]} rows, {df_candidates.shape[1]} cols")
    if df_voters is not None:
        print(f"Voters df: {df_voters.shape[0]} rows, {df_voters.shape[1]} cols")

    all_ok = True

    for spec in meta["specs"]:
        source_id = spec["source_q_id"]
        clone_ids = spec["clone_ids"]
        flip = spec["flip_answers"]
        clone_type = spec["clone_type"]
        print(f"\nSpec: source=Q{source_id}, type={clone_type}, n={len(clone_ids)}, flip={flip}")

        # 1. Question text
        print(f"\n  [1] Question text:")
        for col in TEXT_COLS:
            ok = check_text_col(df_questions, source_id, clone_ids, col)
            all_ok = all_ok and ok
        if all_ok:
            print(f"    OK: all clone texts are byte-for-byte identical to source")

        # 2. Candidate answers
        if df_candidates is not None:
            print(f"\n  [2] Candidate answers (flip={flip}):")
            ok = check_answer_col(df_candidates, source_id, clone_ids, "candidates", flip)
            all_ok = all_ok and ok
            if ok:
                print(f"    OK: all candidate answer columns match (flip={flip})")

        # 3. Voter answers
        if df_voters is not None:
            src_ans_col = f"answer_{source_id}"
            nan_voters = df_voters[src_ans_col].isna().sum() if src_ans_col in df_voters.columns else 0
            print(f"\n  [3] Voter answers (flip={flip}, NaN voters: {nan_voters}/{len(df_voters)}):")
            ok = check_answer_col(df_voters, source_id, clone_ids, "voters", flip)
            all_ok = all_ok and ok
            if ok:
                print(f"    OK: all voter answer columns match (flip={flip})")

        # 4. Voter weights
        if df_voters is not None:
            src_wt_col = f"weight_{source_id}"
            nan_weights = df_voters[src_wt_col].isna().sum() if src_wt_col in df_voters.columns else 0
            print(f"\n  [4] Voter weights (NaN weights: {nan_weights}/{len(df_voters)}):")
            ok = check_weight_col(df_voters, source_id, clone_ids)
            all_ok = all_ok and ok
            if ok:
                print(f"    OK: all voter weight columns match")

        # 5. Show full source text for manual inspection
        source_row = df_questions[df_questions["ID_question"] == source_id]
        for col in TEXT_COLS:
            if col in df_questions.columns and not source_row.empty:
                text = source_row.iloc[0][col]
                print(f"\n  Source text ({col}): {repr(text)}")
                print(f"  Length: {len(text)} chars")
                print(f"  Bytes (UTF-8): {len(text.encode('utf-8'))}")
                # Show first/last few chars with their codepoints
                print(f"  First 5 chars: {[(c, hex(ord(c))) for c in text[:5]]}")
                print(f"  Last 5 chars:  {[(c, hex(ord(c))) for c in text[-5:]]}")

    print(f"\n{'PASS' if all_ok else 'FAIL'}: {clone_id}")


def main():
    parser = argparse.ArgumentParser(description="Verify cloned questions are identical to source.")
    parser.add_argument(
        "--clone-id",
        type=str,
        default=None,
        help="Clone dataset to verify. If not specified, verifies all identical clone datasets.",
    )
    args = parser.parse_args()

    if args.clone_id:
        verify_clone_identity(args.clone_id)
    else:
        for clone_id in ["identical_combinedvar_n10", "identical_q32214_n10", "identical_highcandvar_n10"]:
            verify_clone_identity(clone_id)


if __name__ == "__main__":
    main()
