"""
Shared config loading utilities used by main.py and all experiment scripts.
"""

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

    return config


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
