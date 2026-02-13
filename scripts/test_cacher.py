import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from vqs.cache_management import CacheManager

# Run as module:
# python -m scripts.test_cacher


# We define it as a standard function starting with 'test_'
def test_cache_functionality():
    """Verifies that CacheManager hits, misses, and saves correctly."""

    print("--- Execution Started ---")
    # 1. Setup Mock Config
    # Since this is in scripts/, we go up one level to find the root
    project_root = Path(__file__).parent.parent
    test_cache_dir = project_root / "test_cache_folder"

    config = SimpleNamespace(
        data_year=2023,
        dist="SBERT",
        alpha=0.6,
        data_choice="cleaned",
        # Ensure your manager has the paths it needs
        CACHE_DIR=test_cache_dir,
    )

    params_list = ["data_year", "dist", "alpha"]
    prefix = "test_run"

    # 2. Execution Logic
    cacher = CacheManager(config, test_cache_dir, prefix, params_list)

    # Clean up old test data if it exists for a fresh start
    if cacher.path.exists():
        cacher.path.unlink()

    # Test Save
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3)), columns=list("ABC"))
    cacher.save(df)
    assert cacher.path.exists(), "Cache file was not created."

    # Test Load (Hit)
    loaded_df = cacher.load_if_exists()
    assert loaded_df is not None, "Cache hit failed for identical parameters."

    # Test Miss (Change alpha)
    config.alpha = 0.9
    cacher_miss = CacheManager(config, test_cache_dir, prefix, params_list)
    assert (
        cacher_miss.load_if_exists() is None
    ), "Cache hit occurred despite param change."

    print("✅ All cache tests passed!")


if __name__ == "__main__":
    test_cache_functionality()
