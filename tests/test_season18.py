from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import load_dashboard_data
from mpl_predictor.data.season18 import (
    build_season18_asof_snapshot,
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
from mpl_predictor.models.online_learning import (
    OnlineTemperatureLearner,
    adapt_probability,
)
from mpl_predictor.models.season18_predictions import (
    build_match_accuracy_metrics,
    build_online_learning_summary,
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


def test_august_21_snapshot_has_ten_results_and_no_backdated_roster() -> None:
    teams, rosters, schedule = _load_season18()
    cutoff = date(2026, 8, 21)
    windows = build_season18_prediction_windows(schedule, as_of=cutoff)
    _, snapshot_rosters, snapshot_schedule, report = build_season18_asof_snapshot(
        teams, rosters, schedule, cutoff
    )

    assert windows["snapshot_id"].tolist() == ["S18_PRE", "S18_W01", "S18_D20260821"]
    assert windows["available_result_count"].tolist() == [0, 8, 10]
    assert windows.iloc[-1]["partial_week"] == 2
    assert snapshot_schedule["status"].eq("completed").sum() == 10
    assert snapshot_schedule["status"].eq("scheduled").sum() == 62
    assert snapshot_rosters.empty
    assert report["scope"]["completed_weeks"] == [1]
    assert report["scope"]["partial_weeks"] == [2]
    results = snapshot_schedule.loc[snapshot_schedule["status"].eq("completed")]
    assert set(results.loc[results["week"].eq(2), "winner_team_id"]) == {"NAVI", "BTR"}


def test_august_31_snapshot_has_week_three_and_verified_roster() -> None:
    teams, rosters, schedule = _load_season18()
    cutoff = date(2026, 8, 31)
    _, snapshot_rosters, snapshot_schedule, report = build_season18_asof_snapshot(
        teams, rosters, schedule, cutoff
    )

    assert snapshot_schedule["status"].eq("completed").sum() == 24
    assert snapshot_schedule["status"].eq("scheduled").sum() == 48
    assert len(snapshot_rosters) == 79
    assert report["scope"]["completed_weeks"] == [1, 2, 3]
    assert report["scope"]["partial_weeks"] == []
    source_observed_date = pd.to_datetime(schedule["observed_at"].iloc[0]).date()
    expected_retrospective = source_observed_date > cutoff
    expected_basis = (
        "retrospective_end_of_day_wib" if expected_retrospective else "official_observation_date"
    )
    assert report["retrospective_reconstruction"] is expected_retrospective
    assert snapshot_schedule["snapshot_basis"].eq(expected_basis).all()


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
    assert probabilities["base_team_a_win_probability"].between(0, 1).all()
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
    completed = probabilities.loc[probabilities["status"].eq("completed")]
    scheduled = probabilities.loc[probabilities["status"].eq("scheduled")]
    assert completed["online_learning_observation_count"].tolist() == list(range(24))
    assert scheduled["online_learning_observation_count"].eq(24).all()
    assert completed["online_learning_update_applied"].all()
    assert scheduled["online_learning_update_applied"].eq(False).all()
    assert completed.iloc[0]["team_a_win_probability"] == pytest.approx(
        completed.iloc[0]["base_team_a_win_probability"]
    )
    assert (
        completed.iloc[1:]["team_a_win_probability"]
        .ne(completed.iloc[1:]["base_team_a_win_probability"])
        .any()
    )
    actual_team_a_win = completed["winner_team_id"].eq(completed["team_a_id"]).astype(float)
    assert completed["online_learning_error_signal"].to_numpy() == pytest.approx(
        (actual_team_a_win - completed["team_a_win_probability"]).to_numpy()
    )
    expected_correct = completed["predicted_winner_team_id"].eq(completed["winner_team_id"])
    pd.testing.assert_series_equal(
        completed["prediction_correct"].astype(bool),
        expected_correct,
        check_names=False,
    )
    assert set(completed["accuracy_status"]) == {"correct", "incorrect"}
    assert completed["result_update_status"].eq("incorporated_after_pre_match_prediction").all()
    assert scheduled["prediction_correct"].isna().all()
    assert scheduled["accuracy_status"].eq("pending_result").all()
    assert scheduled["result_update_status"].eq("awaiting_result").all()

    accuracy = build_match_accuracy_metrics(probabilities)
    assert accuracy["evaluated_match_count"] == 24
    assert accuracy["correct_prediction_count"] == int(expected_correct.sum())
    assert accuracy["incorrect_prediction_count"] == int((~expected_correct).sum())
    assert accuracy["match_accuracy"] == pytest.approx(expected_correct.mean())
    assert 0 <= accuracy["brier_score"] <= 1
    assert accuracy["log_loss"] > 0

    learning = build_online_learning_summary(probabilities)
    assert learning["enabled"] is True
    assert learning["update_count"] == 24
    assert learning["final_observation_count"] == 24
    assert 0.5 <= learning["final_confidence_scale"] <= 2.0
    assert learning["adaptive_prequential_metrics"] == accuracy


def test_online_learning_is_symmetric_and_uses_prediction_error() -> None:
    learner = OnlineTemperatureLearner.from_config(
        {
            "enabled": True,
            "learning_rate": 0.05,
            "l2_regularization": 0.05,
            "minimum_scale": 0.5,
            "maximum_scale": 2.0,
        }
    )
    probability = learner.predict(0.8)
    initial_scale = learner.scale
    learner.update(0.8, 0.0)

    assert probability == pytest.approx(0.8)
    assert learner.observation_count == 1
    assert learner.scale < initial_scale
    assert adapt_probability(0.8, learner.scale) == pytest.approx(
        1.0 - adapt_probability(0.2, learner.scale)
    )


def test_match_outcome_is_learned_only_after_its_own_prediction(
    final_simulation_inputs,
) -> None:
    tables, teams, schedule, artifact, original, _, _ = final_simulation_inputs
    feature_config = load_feature_config(PROJECT_ROOT / "config" / "feature_config.json")
    completed = schedule.loc[schedule["status"].eq("completed")].sort_values(
        ["scheduled_at", "official_match_id"]
    )
    changed_match = completed.iloc[-1]
    changed_schedule = schedule.copy()
    changed_index = changed_match.name
    team_a_won = changed_match["winner_team_id"] == changed_match["team_a_id"]
    changed_schedule.loc[changed_index, "winner_team_id"] = (
        changed_match["team_b_id"] if team_a_won else changed_match["team_a_id"]
    )
    changed_schedule.loc[changed_index, "winner_side"] = "team_b" if team_a_won else "team_a"
    changed_schedule.loc[changed_index, ["team_a_score", "team_b_score"]] = (
        [0, 2] if team_a_won else [2, 0]
    )

    changed, _, _ = build_season18_match_probabilities(
        tables, changed_schedule, teams, feature_config, artifact
    )
    match_id = changed_match["official_match_id"]
    original_row = original.loc[original["official_match_id"].eq(match_id)].iloc[0]
    changed_row = changed.loc[changed["official_match_id"].eq(match_id)].iloc[0]

    assert changed_row["base_team_a_win_probability"] == pytest.approx(
        original_row["base_team_a_win_probability"]
    )
    assert changed_row["team_a_win_probability"] == pytest.approx(
        original_row["team_a_win_probability"]
    )
    assert changed_row["online_learning_scale_after_update"] != pytest.approx(
        original_row["online_learning_scale_after_update"]
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
    assert set(dashboard_data["model_comparison"]["model_family"]) == {
        "logistic",
        "random_forest",
        "xgboost",
    }
    assert {
        "base_team_a_win_probability",
        "base_prediction_correct",
        "prediction_correct",
        "accuracy_status",
        "result_update_status",
        "online_learning_observation_count",
        "online_learning_error_signal",
        "online_learning_scale",
        "online_learning_scale_after_update",
        "online_learning_update_applied",
    }.issubset(dashboard_data["matches"].columns)
    assert set(dashboard_data["predictions"]["snapshot_id"]) == {
        "S18_PRE",
        "S18_W01",
        "S18_W02",
        "S18_W03",
    }
    sums = dashboard_data["predictions"].groupby("snapshot_id")["champion_probability"].sum()
    assert sums.sub(1.0).abs().lt(1e-9).all()
