import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from clone_pipeline.spec import CloneSpec


def write_cloned_dataset(
    dataframes: dict[str, pd.DataFrame],
    specs: list[CloneSpec],
    out_dir: Path,
    cand_filename: str,
    voters_filename: str,
    questions_filename: str,
    source_dir: Path,
    data_year: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}\n"
            f"Delete it manually if you want to regenerate."
        )
    out_dir.mkdir(parents=True)

    print(f"Writing cloned dataset to: {out_dir}")
    dataframes["candidates"].to_parquet(out_dir / cand_filename)
    dataframes["voters"].to_parquet(out_dir / voters_filename)
    dataframes["questions"].to_parquet(out_dir / questions_filename)

    metadata = {
        "data_year": data_year,
        "source_dir": str(source_dir),
        "generated_at": datetime.now().isoformat(),
        "specs": [
            {
                "source_q_id": s.source_q_id,
                "clone_type": s.clone_type,
                "n_clones": s.n_clones,
                "clone_ids": s.clone_ids,
                "flip_answers": s.flip_answers,
            }
            for s in specs
        ],
    }
    with open(out_dir / "clone_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Done. Metadata written to {out_dir / 'clone_metadata.json'}")
