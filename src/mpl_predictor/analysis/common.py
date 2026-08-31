import json
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_TABLES = (
    "teams",
    "matches",
    "games",
    "players",
    "player_identity",
    "championships",
    "drafts",
    "player_season_stats",
)


def load_canonical_tables(directory: Path) -> dict[str, pd.DataFrame]:
    """Load the complete canonical dataset from Parquet files."""
    missing = [name for name in CANONICAL_TABLES if not (directory / f"{name}.parquet").exists()]
    if missing:
        names = ", ".join(f"{name}.parquet" for name in missing)
        raise FileNotFoundError(
            f"Canonical dataset is incomplete in {directory}. Missing: {names}. "
            "Run `mpl-predictor canonicalize` first."
        )
    return {name: pd.read_parquet(directory / f"{name}.parquet") for name in CANONICAL_TABLES}


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records with ISO-formatted dates."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
