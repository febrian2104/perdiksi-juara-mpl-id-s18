import re
from dataclasses import dataclass
from pathlib import Path

_FILENAME_PATTERN = re.compile(r"^mpl_id_season_(?P<season>\d{2})_(?P<table>[a-z0-9_]+)\.csv$")


@dataclass(frozen=True, slots=True)
class DatasetFile:
    """One recognized MPL season dataset file."""

    season: int
    table: str
    path: Path


def discover_dataset_files(data_dir: Path) -> list[DatasetFile]:
    """Discover recognized season CSV files below the data directory."""
    files: list[DatasetFile] = []
    if not data_dir.exists():
        return files

    for path in data_dir.glob("mpl-season*/*.csv"):
        match = _FILENAME_PATTERN.match(path.name)
        if match is None:
            continue
        files.append(
            DatasetFile(
                season=int(match.group("season")),
                table=match.group("table"),
                path=path.resolve(),
            )
        )

    return sorted(files, key=lambda item: (item.season, item.table, str(item.path)))
