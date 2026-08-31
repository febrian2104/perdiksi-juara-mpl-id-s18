from pathlib import Path

import pandas as pd

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.analysis.eda import build_eda_report
from mpl_predictor.analysis.prediction_policy import (
    build_prediction_policy_report,
    build_prediction_windows,
    load_prediction_policy,
    write_prediction_outputs,
)
from mpl_predictor.analysis.quality import build_quality_report, write_quality_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"
POLICY_PATH = PROJECT_ROOT / "config" / "prediction_policy.json"


def _tables() -> dict[str, pd.DataFrame]:
    return load_canonical_tables(CANONICAL_DIR)


def test_quality_report_marks_core_data_ready(tmp_path: Path) -> None:
    report = build_quality_report(_tables())
    output = tmp_path / "quality.json"
    write_quality_report(report, output)

    statuses = {item["feature_group"]: item["status"] for item in report["feature_groups"]}
    assert report["core_modeling_ready"] is True
    assert report["blocking_issue_count"] == 0
    assert report["dataset_scope"]["franchise_team_season_count"] == 118
    assert statuses["team_match_history"] == "ready"
    assert statuses["published_player_statistics"] == "excluded_as_of"
    assert statuses["champion_and_final_rank"] == "target_only"
    assert output.exists()


def test_eda_separates_franchise_scope_and_target() -> None:
    report, frames = build_eda_report(_tables())

    assert report["scope"]["team_season_observations"] == 150
    assert report["scope"]["franchise_team_season_observations"] == 118
    assert report["scope"]["franchise_champion_observations"] == 14
    assert report["franchise_comparison"]["difference_percentage_points"] > 0
    assert len(frames["team_season_performance"]) == 150
    assert set(frames["team_season_performance"]["champion"].unique()) == {False, True}


def test_prediction_windows_have_leakage_safe_cutoffs(tmp_path: Path) -> None:
    tables = _tables()
    policy = load_prediction_policy(POLICY_PATH)
    windows = build_prediction_windows(tables, policy)
    report = build_prediction_policy_report(policy, windows)
    report_path = tmp_path / "policy.json"
    windows_path = tmp_path / "windows.csv"
    write_prediction_outputs(report, windows, report_path, windows_path)

    preseason = windows.loc[windows["prediction_type"].eq("preseason")]
    weekly = windows.loc[windows["prediction_type"].eq("weekly")]
    assert len(windows) == 129
    assert len(preseason) == 14
    assert len(weekly) == 115
    assert preseason["feature_cutoff_date"].lt(preseason["first_regular_season_date"]).all()
    assert preseason["available_regular_match_count"].eq(0).all()
    assert weekly["completed_week"].notna().all()
    assert windows.loc[windows["snapshot_id"].eq("S11_W06"), "feature_cutoff_date"].iloc[
        0
    ] == pd.Timestamp("2023-03-26")
    assert report_path.exists()
    assert windows_path.exists()
