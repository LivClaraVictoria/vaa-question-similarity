import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import shutil
from vqs.result_management import ResultManager


def test_result_manager():
    # 1. Setup a dummy environment
    test_dir = Path("test_cache_dir")
    test_dir.mkdir(exist_ok=True)

    # Mock Config
    config = SimpleNamespace(
        dist="SBERT",
        data_year=2024,
        alpha=1.0,
        results_file_type="csv",
        save_results=True,
    )

    params_list = ["dist", "data_year", "alpha"]

    print("--- Initializing ResultManager ---")
    rm = ResultManager(config, test_dir, params_list, prefix="test")
    initial_hash = rm.hash
    print(f"Generated Hash: {initial_hash}")

    # 2. Test Saving
    df_original = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    save_path = rm.save(df_original, readable=True)
    print(f"Saved to: {save_path.name}")

    # 3. Test Existence (The "Skip" Logic)
    print("\n--- Testing Existence Check ---")
    exists_path = rm.exists()
    if exists_path and initial_hash in exists_path.name:
        print("✅ SUCCESS: ResultManager recognized existing file by hash.")
    else:
        print("❌ FAILURE: ResultManager failed to find the file.")

    # 4. Test Loading
    print("\n--- Testing Loading ---")
    df_loaded = rm.load()
    if df_loaded is not None and df_loaded.equals(df_original):
        print("✅ SUCCESS: Data integrity maintained.")
    else:
        print("❌ FAILURE: Data mismatch or load failed.")

    # 5. Test Parameter Sensitivity (Change a param, hash should change)
    print("\n--- Testing Parameter Sensitivity ---")
    config.alpha = 0.5
    rm_new = ResultManager(config, test_dir, params_list, prefix="test")
    if rm_new.hash != initial_hash:
        print(f"✅ SUCCESS: Hash changed to {rm_new.hash} when alpha changed.")
    else:
        print("❌ FAILURE: Hash did not change despite parameter update.")

    # 6. Cleanup (Optional)
    # shutil.rmtree(test_dir)
    print("\nTest Complete!")


if __name__ == "__main__":
    test_result_manager()
