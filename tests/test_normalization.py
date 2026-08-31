from pathlib import Path

import pandas as pd

from mpl_predictor.data.normalization import normalize_tables, write_normalized_tables
from mpl_predictor.data.semantic_audit import audit_semantics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def test_normalization_standardizes_core_match_fields() -> None:
    tables = normalize_tables(DATA_DIR)
    matches = tables["matches"]

    assert matches["season"].dtype == pd.Int64Dtype()
    assert set(matches["stage"].dropna()) == {"regular_season", "playoffs"}
    assert set(matches["winner_side"].dropna()) == {"team_a", "team_b", "draw"}
    assert matches["team_a_id"].notna().all()
    assert matches["team_b_id"].notna().all()
    assert matches["winner_team_id"].notna().sum() == len(matches) - 14
    assert "upper_bracket_semifinals" not in set(matches["round"].dropna())
    assert "midlaner" not in set(tables["player_season_stats"]["role"].dropna())


def test_normalization_keeps_unknown_values_missing() -> None:
    tables = normalize_tables(DATA_DIR)
    games = tables["games"]

    assert games["winner_side"].isna().any()
    assert games["duration_minutes"].isna().any()
    assert "NULL" not in set(games["winner_raw"].dropna())


def test_semantic_audit_has_no_consistency_errors() -> None:
    report = audit_semantics(normalize_tables(DATA_DIR))

    assert not report.errors, [check.message for check in report.errors]
    assert {check.code for check in report.warnings} == {
        "draft_coverage_partial",
        "game_duration_partial",
        "game_winner_partial",
        "player_stats_partial",
    }
    assert any(check.code == "split_date_matches" for check in report.information)


def test_normalized_tables_can_be_written_as_parquet(tmp_path: Path) -> None:
    tables = normalize_tables(DATA_DIR)
    outputs = write_normalized_tables(tables, tmp_path)

    assert len(outputs) == 7
    assert all(path.exists() for path in outputs.values())
    assert len(pd.read_parquet(outputs["matches"])) == 1182
