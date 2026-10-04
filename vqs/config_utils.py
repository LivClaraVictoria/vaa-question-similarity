"""
Shared config loading utilities used by main.py and all experiment scripts.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def load_config(config_path: Path):
    """
    Loads a config.py file by executing it and capturing its variables
    into a namespace object.
    """

    # 1. Read the raw text of the config file
    try:
        config_script = config_path.read_text()
    except FileNotFoundError:
        print(f"Error: Config file not found at: {config_path}")
        sys.exit(1)

    # 2. Create an empty "container" (a dictionary)
    #    This is where the variables from the config file will live.
    config_vars = {"__file__": str(config_path)}

    # 3. Execute the config file's script.
    #    All variables (from the import and the overrides)
    #    are loaded into the 'config_vars' dictionary.
    try:
        exec(config_script, config_vars)
    except Exception as e:
        print(f"Error while loading config file {config_path}:\n{e}")
        sys.exit(1)

    # 4. Convert the dictionary into an object (SimpleNamespace).
    #    This lets you use dot notation (like config.dist)
    #    instead of dictionary notation (like config['dist']).
    config = SimpleNamespace(**config_vars)

    # We remove these two internal Python variables, just to keep it clean
    config_vars.pop("__builtins__", None)
    config_vars.pop("__name__", None)

    # `overrides` comes from `from configs.base_constants import *`, i.e. one list object shared
    # by every config loaded in this process. Copy it so overrides don't leak between configs.
    config.overrides = list(getattr(config, "overrides", []))

    _apply_district_env(config)
    _apply_alpha_env(config)

    return config


def _apply_alpha_env(config):
    """
    Sets the CRW alpha from env var VQS_ALPHA for configs that use CRW, so every experiment of a
    test canton runs at the single alpha chosen on the validation canton. A no-op when the config
    already has that alpha (names/hashes then match the ZH runs). Sweep scripts overwrite alpha
    themselves and are unaffected.
    """
    raw = os.environ.get("VQS_ALPHA")
    if not raw or not getattr(config, "apply_clone_robust_weighting", False) \
            or getattr(config, "data_choice", "") == "fake":
        return
    alpha = float(raw)
    if alpha == getattr(config, "alpha", None):
        return
    print(f"VQS_ALPHA={raw}: re-targeting config from alpha {config.alpha} to {alpha}.")
    apply_overrides(config, [f"alpha={raw}"])


def _apply_district_env(config):
    """
    Re-targets a canton-scoped config (district != "all") to the canton in env var VQS_DISTRICT.
    Configs with district="all" (fake data, national runs) are left untouched. Registered as a
    regular override, so it shows up in run names and metadata.
    """
    district = os.environ.get("VQS_DISTRICT")
    current = getattr(config, "district", "all")
    if not district or current == "all" or district == current:
        return
    if district not in config.DISTRICT2ID:
        print(f"Error: VQS_DISTRICT='{district}' is not a canton code ({', '.join(config.DISTRICT2ID)}).")
        sys.exit(1)
    print(f"VQS_DISTRICT={district}: re-targeting config from district '{current}' to '{district}'.")
    apply_overrides(config, [f"district={district}"])


def respondent_hash_params(config) -> list[str]:
    """Extra cache-hash params for answer-based metrics (empty for text-embedding metrics)."""
    from configs import base_constants

    answer_based = getattr(config, "ANSWER_BASED_METRICS", base_constants.ANSWER_BASED_METRICS)
    if config.dist.upper() in answer_based:
        return list(getattr(config, "ANSWER_METRIC_HASH_PARAMS", base_constants.ANSWER_METRIC_HASH_PARAMS))
    return []


def canton_results_path(path, config=None) -> Path:
    """
    Maps an experiment_results path to the config's canton (or, without a config, to the
    canton in VQS_DISTRICT — for compile scripts that only read results). The validation
    canton (and district="all") keeps the original tree; every test canton gets a mirror of
    the same structure under experiment_results/cantons/<code>/, so cantons compare
    path-for-path.
    """
    if config is None:
        from configs import base_constants as config
        district = os.environ.get("VQS_DISTRICT") or config.VALIDATION_DISTRICT
    else:
        district = config.district
    path = Path(path)
    if district in ("all", config.VALIDATION_DISTRICT):
        return path
    abs_path = path if path.is_absolute() else config.PROJECT_ROOT / path
    rel = abs_path.relative_to(config.RESULTS_DIR)
    return config.CANTON_RESULTS_DIR / district / rel


def apply_overrides(config, overrides):
    """
    Parses a list of "key=value" strings and updates the config object.
    """
    if not overrides:
        return config

    print(f"\n--- Applying CLI Overrides ---")
    for item in overrides:
        if "=" not in item:
            print(
                f"Warning: Ignoring malformed override '{item}'. Use 'key=value' format."
            )
            continue

        key, value_str = item.split("=", 1)

        # Check if the key exists in the config to avoid typos
        if not hasattr(config, key):
            print(
                f"Warning: New config key '{key}' is being added (was not in config file)."
            )

        # Keep track of overrides for caching purposes
        safe_value = value_str.replace("/", "_").replace("\\", "_")
        config.overrides.append(f"{key}~{safe_value}")

        # --- Type Inference ---
        # 1. Boolean
        if value_str.lower() == "true":
            val = True
        elif value_str.lower() == "false":
            val = False
        # 2. int, float, or string
        else:
            # Try to convert to int, then float, finally keep as string
            try:
                val = int(value_str)
            except ValueError:
                try:
                    val = float(value_str)
                except ValueError:
                    val = value_str  # it's a string

        # Update the SimpleNamespace config object
        setattr(config, key, val)
        print(f" -> Set '{key}' to: {val} ({type(val).__name__})")

    print("------------------------------\n")
    return config
