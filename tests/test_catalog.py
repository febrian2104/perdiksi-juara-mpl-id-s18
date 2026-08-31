from pathlib import Path

from mpl_predictor.cli import main
from mpl_predictor.data.audit import audit_data
from mpl_predictor.data.catalog import discover_dataset_files
from mpl_predictor.data.contracts import BASE_TABLES, PLAYER_STATS_TABLE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def test_historical_seasons_are_discoverable() -> None:
    files = discover_dataset_files(DATA_DIR)
    seasons = {item.season for item in files}

    assert set(range(1, 18)).issubset(seasons)


def test_expected_table_availability_by_era() -> None:
    files = discover_dataset_files(DATA_DIR)
    tables_by_season = {
        season: {item.table for item in files if item.season == season} for season in range(1, 18)
    }

    for season in range(1, 4):
        assert tables_by_season[season] == BASE_TABLES

    for season in range(4, 18):
        assert tables_by_season[season] == BASE_TABLES | {PLAYER_STATS_TABLE}


def test_foundation_audit_has_no_structural_errors() -> None:
    report = audit_data(DATA_DIR)

    assert not report.errors, [issue.message for issue in report.errors]
    assert report.total_rows > 0


def test_audit_cli_succeeds(capsys: object) -> None:
    exit_code = main(["audit", "--data-dir", str(DATA_DIR)])
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert exit_code == 0
    assert "Errors: 0" in output
