"""
Unified CLI for all thesis experiments (VAA question similarity / clone-robust weighting).

Subcommands:
  pipeline              Run full pipeline: distances → CRW → recommendations
                          --config <cfg> [--distances-only] [key=value overrides...]
  compare               Cross-run recommendation comparator
                          <rec_a.parquet> <rec_b.parquet> [-n N]
  generate-clones       Write synthetic cloned dataset to disk
                          --config <clone_cfg>
  clone-count-sweep     Perfect clones × clone count × alpha sweep
  model-selection       Alpha sweep comparing base vs cloned config (model selection)
  rec-distortion        Per-question alpha sweep (75 q × 21 α × clone types)
  partisan-distortion   Party visibility impact from cloning
  approx-rec-distortion Alpha sweep on approximate (mini→full) questions
  approx-partisan       Mini vs full questionnaire party visibility

All subcommand-specific flags are passed through to the underlying module unchanged.
Run `python -m main <subcommand> --help` for per-subcommand help.
"""

import importlib
import sys

COMMANDS = {
    "pipeline":               ("vqs.pipeline_runner",                                   "main"),
    "compare":                ("cross_run_analysis.cli",                                "main"),
    "generate-clones":        ("clone_pipeline.cli",                                    "main"),
    "clone-count-sweep":      ("experiments.perfect_clones.clone_count_sweep",          "main"),
    "model-selection":        ("experiments.perfect_clones.model_selection",            "main"),
    "rec-distortion":         ("experiments.perfect_clones.recommendation_distortion",  "main"),
    "partisan-distortion":    ("experiments.perfect_clones.partisan_distortion",        "main"),
    "approx-rec-distortion":  ("experiments.approximate_clones.recommendation_distortion", "main"),
    "approx-partisan":        ("experiments.approximate_clones.partisan_distortion",    "main"),
}


def _usage():
    print("Usage: python -m main <subcommand> [args...]")
    print()
    print("Subcommands:")
    for cmd in COMMANDS:
        print(f"  {cmd}")
    print()
    print("Run `python -m main <subcommand> --help` for per-subcommand usage.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        _usage()
        sys.exit(0 if (len(sys.argv) >= 2 and sys.argv[1] in ("--help", "-h")) else 1)

    cmd = sys.argv[1]
    mod_path, fn = COMMANDS[cmd]
    mod = importlib.import_module(mod_path)
    getattr(mod, fn)(sys.argv[2:])
