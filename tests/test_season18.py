from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import load_dashboard_data
from mpl_predictor.data.season18 import (
    build_season18_report,
    merge_roster_history,
    parse_roster_html,
    validate_season18_data,
)
from mpl_predictor.features.snapshots import load_feature_config
from mpl_predictor.models.explainability import (
    build_global_importance,
    build_match_explanations,
)
from mpl_predictor.models.final import predict_match_probability, train_final_match_model
from mpl_predictor.models.season18_predictions import (
    build_season18_prediction_windows,
    schedule_as_of_window,
)
from mpl_predictor.models.simulation import (
    build_season18_match_probabilities,
    load_simulation_config,
    simulate_season18,
)
from mpl_predictor.models.walk_forward import load_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEASON18_DIR = PROJECT_ROOT / "data" / "season18"
CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"


def _load_season18() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(SEASON18_DIR / "teams.csv")
    rosters = pd.read_csv(SEASON18_DIR / "rosters.csv")
    schedule = pd.read_csv(SEASON18_DIR / "schedule_results.csv")
    schedule["scheduled_at"] = pd.to_datetime(schedule["scheduled_at"], utc=True).dt.tz_convert(
        "Asia/Jakarta"
    )
    for column in ("team_a_score", "team_b_score"):
        schedule[column] = pd.to_numeric(schedule[column], errors="coerce").astype("Int64")
    return teams, rosters, schedule


def test_official_season18_snapshot_is_complete_and_time_guarded() -> None:
    teams, rosters, schedule = _load_season18()
    observed_at = date(2026, 8, 31)
    checks = validate_season18_data(teams, rosters, schedule, observed_at)
    report = build_season18_report(teams, rosters, schedule, observed_at)

    assert len(teams) == 9
    assert len(rosters) == 79
    assert rosters["member_type"].eq("player").sum() == 59
    assert rosters["valid_from"].eq("2026-08-31").all()
    assert len(schedule) == 72
    assert schedule["status"].eq("completed").sum() == 24
    assert schedule["status"].eq("scheduled").sum() == 48
    assert all(check["status"] == "pass" for check in checks)
    assert report["blocking_issue_count"] == 0


def test_roster_parser_separates_players_and_staff() -> None:
    html = """
    <section><h5>ROSTER SEASON 18</h5>
      <div class="col-md-3 col-6">
        <div class="player-name">Test Player</div><div class="player-role">Mid Lane</div>
      </div>
      <div class="col-md-3 col-6">
        <div class="player-name">Test Coach</div><div class="player-role">Coach</div>
      </div>
    </section>
    """
    roster = parse_roster_html(html, "AE", date(2026, 8, 31))

    assert roster["member_type"].tolist() == ["player", "staff"]
    assert roster["role"].tolist() == ["mid_lane", "coach"]
    assert roster["valid_from"].eq(date(2026, 8, 31)).all()


def test_roster_updates_preserve_first_seen_and_close_removed_members() -> None:
    original_html = """
    <section><h5>ROSTER SEASON 18</h5>
      <div class="col-md-3 col-6">
        <div class="player-name">Retained</div><div class="player-role">Mid Lane</div>
      </div>
      <div class="col-md-3 col-6">
        <div class="player-name">Removed</div><div class="player-role">Jungle</div>
      </div>
    </section>
    """
    current_html = """
    <section><h5>ROSTER SEASON 18</h5>
      <div class="col-md-3 col-6">
        <div class="player-name">Retained</div><div class="player-role">Mid Lane</div>
      </div>
      <div class="col-md-3 col-6">
        <div class="player-name">New Player</div><div class="player-role">Gold Lane</div>
      </div>
    </section>
    """
    original = parse_roster_html(original_html, "AE", date(2026, 8, 31))
    current = parse_roster_html(current_html, "AE", date(2026, 9, 7))
    merged = merge_roster_history(original, current, date(2026, 9, 7)).set_index("nickname")

    assert merged.loc["Retained", "valid_from"] == date(2026, 8, 31)
    assert merged.loc["New Player", "valid_from"] == date(2026, 9, 7)
    assert merged.loc["Removed", "valid_to"] == date(2026, 9, 7)


def test_preseason_and_weekly_windows_hide_future_results() -> None:
    _, _, schedule = _load_season18()
    windows = build_season18_prediction_windows(schedule)

    assert windows["snapshot_id"].tolist() == ["S18_PRE", "S18_W01", "S18_W02", "S18_W03"]
    assert windows["feature_cutoff_date"].astype(str).tolist() == [
        "2026-08-13",
        "2026-08-16",
        "2026-08-23",
        "2026-08-30",
    ]
    expected_completed = {"S18_PRE": 0, "S18_W01": 8, "S18_W02": 16, "S18_W03": 24}
    for _, window in windows.iterrows():
        snapshot = schedule_as_of_window(schedule, window)
        assert snapshot["status"].eq("completed").sum() == expected_completed[window.snapshot_id]
        assert snapshot.loc[snapshot["status"].eq("scheduled"), "winner_side"].isna().all()


@pytest.fixture(scope="module")
def final_simulation_inputs():
    teams, _, schedule = _load_season18()
    tables = load_canonical_tables(CANONICAL_DIR)
    feature_config = load_feature_config(PROJECT_ROOT / "config" / "feature_config.json")
    model_config = load_model_config(PROJECT_ROOT / "config" / "model_config.json")
    match_features = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "features" / "match_features.parquet"
    )
    artifact = train_final_match_model(match_features, model_config)
    probabilities, tracker, history = build_season18_match_probabilities(
        tables, schedule, teams, feature_config, artifact
    )
    return tables, teams, schedule, artifact, probabilities, tracker, history


def test_final_model_is_symmetric_and_s18_results_are_fixed(final_simulation_inputs) -> None:
    _, _, _, artifact, probabilities, _, _ = final_simulation_inputs
    features = {column: 0.0 for column in artifact["feature_columns"]}
    features["elo_rating_diff"] = 125.0
    reverse = {column: -value for column, value in features.items()}

    probability = predict_match_probability(artifact, features)
    reverse_probability = predict_match_probability(artifact, reverse)

    assert probability == pytest.approx(1.0 - reverse_probability, abs=1e-12)
    assert probabilities["team_a_win_probability"].between(0, 1).all()
    assert (
        probabilities.loc[probabilities["status"].eq("completed"), "probability_basis"]
        .eq("historical_pre_match_state")
        .all()
    )
    assert (
        probabilities.loc[probabilities["status"].eq("scheduled"), "probability_basis"]
        .eq("current_as_of_state")
        .all()
    )


def test_season18_simulation_is_reproducible_and_normalized(final_simulation_inputs) -> None:
    tables, teams, schedule, artifact, probabilities, tracker, history = final_simulation_inputs
    config = load_simulation_config(PROJECT_ROOT / "config" / "simulation_config.json")
    config["iterations"] = 250

    first, first_report = simulate_season18(
        tables, schedule, teams, probabilities, tracker, history, artifact, config
    )
    second, second_report = simulate_season18(
        tables, schedule, teams, probabilities, tracker, history, artifact, config
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["champion_probability"].sum() == pytest.approx(1.0)
    assert first["playoff_probability"].sum() == pytest.approx(6.0)
    assert first_report["validation"]["blocking_issue_count"] == 0
    assert second_report["validation"]["blocking_issue_count"] == 0


def test_explainability_and_dashboard_outputs_are_loadable(final_simulation_inputs) -> None:
    _, _, _, artifact, probabilities, _, _ = final_simulation_inputs
    probabilities = probabilities.copy()
    probabilities["snapshot_id"] = "S18_W03"
    probabilities["snapshot_order"] = 3
    global_importance = build_global_importance(artifact)
    local = build_match_explanations(artifact, probabilities)
    dashboard_data, missing = load_dashboard_data(get_project_paths(PROJECT_ROOT))

    assert len(global_importance) == 28
    assert global_importance["absolute_importance"].is_monotonic_decreasing
    assert local["match_id"].nunique() == 48
    assert local.groupby("match_id")["contribution_rank"].min().eq(1).all()
    assert missing == []
    assert dashboard_data["predictions"]["snapshot_id"].nunique() == 4
    sums = dashboard_data["predictions"].groupby("snapshot_id")["champion_probability"].sum()
    assert sums.sub(1.0).abs().lt(1e-9).all()
