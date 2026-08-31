from pathlib import Path

import pandas as pd
import pytest

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.analysis.prediction_policy import (
    build_prediction_windows,
    load_prediction_policy,
)
from mpl_predictor.features.elo import EloTracker
from mpl_predictor.features.roster import add_roster_features
from mpl_predictor.features.snapshots import (
    build_feature_report,
    build_snapshot_features,
    load_feature_config,
    model_feature_columns,
    write_snapshot_outputs,
)
from mpl_predictor.models.baseline import (
    build_baseline_predictions,
    build_baseline_report,
    write_baseline_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"
POLICY_PATH = PROJECT_ROOT / "config" / "prediction_policy.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "config" / "feature_config.json"


@pytest.fixture(scope="module")
def modeling_outputs():
    tables = load_canonical_tables(CANONICAL_DIR)
    policy = load_prediction_policy(POLICY_PATH)
    config = load_feature_config(FEATURE_CONFIG_PATH)
    windows = build_prediction_windows(tables, policy)
    features, roster_metadata = build_snapshot_features(tables, windows, config)
    return tables, windows, config, features, roster_metadata


def test_elo_update_and_season_carryover_are_deterministic() -> None:
    tracker = EloTracker(
        initial_rating=1500.0,
        k_factor=24.0,
        scale=400.0,
        season_carryover=0.75,
    )
    rating_a, rating_b = tracker.update("A", "B", 1.0)

    assert rating_a == rating_b == 1500.0
    assert tracker.ratings["A"] == 1512.0
    assert tracker.ratings["B"] == 1488.0
    assert sum(tracker.ratings.values()) == 3000.0

    tracker.regress_for_new_season(["A", "B"])
    assert tracker.ratings["A"] == 1509.0
    assert tracker.ratings["B"] == 1491.0


def test_snapshot_features_reconcile_and_respect_cutoffs(modeling_outputs, tmp_path: Path) -> None:
    _, windows, _, features, roster_metadata = modeling_outputs
    report = build_feature_report(features, windows, roster_metadata)
    feature_path = tmp_path / "features.parquet"
    report_path = tmp_path / "features.json"
    write_snapshot_outputs(features, report, feature_path, report_path)

    assert len(features) == 1091
    assert features["snapshot_id"].nunique() == 129
    assert report["blocking_issue_count"] == 0
    assert report["feature_column_count"] == 42
    assert report["enabled_feature_column_count"] == 36
    assert report["disabled_feature_groups"] == ["current_roster_temporal"]
    assert (
        features.loc[features["prediction_type"].eq("preseason"), "current_regular_matches"]
        .eq(0)
        .all()
    )
    assert (
        features["latest_match_completion_date_used"].isna()
        | features["latest_match_completion_date_used"].le(features["feature_cutoff_date"])
    ).all()
    assert features.loc[features["snapshot_id"].eq("S04_PRE"), "elo_rating"].eq(1500).all()
    assert not set(model_feature_columns()) & {"champion", "final_rank_min", "final_rank_max"}
    assert feature_path.exists()
    assert report_path.exists()


def test_roster_features_activate_only_with_temporal_data() -> None:
    features = pd.DataFrame(
        {
            "snapshot_id": ["S05_PRE"],
            "season": [5],
            "franchise_slot_id": ["SLOT_A"],
            "feature_cutoff_date": [pd.Timestamp("2020-01-01")],
        }
    )
    players = pd.DataFrame(
        {
            "season": [4, 4, 5, 5],
            "franchise_slot_id": ["SLOT_A"] * 4,
            "player_id": ["P1", "P2", "P1", "P3"],
            "role": ["mid", "jungle", "mid", "gold"],
            "player_identity_review_required": [False] * 4,
            "valid_from": [None, None, "2019-12-15", "2020-02-01"],
            "valid_to": [None, None, None, None],
        }
    )
    roster_config = {
        "start_date_columns": ["valid_from", "announced_at"],
        "end_date_columns": ["valid_to"],
        "current_roster_requires_temporal_column": True,
        "exclude_identity_review_required": True,
    }
    result, metadata = add_roster_features(features, players, roster_config)

    assert metadata["current_roster_features_enabled"] is True
    assert result.loc[0, "lagged_roster_size"] == 2
    assert result.loc[0, "current_roster_size_asof"] == 1
    assert result.loc[0, "current_roster_retained_count_asof"] == 1
    assert result.loc[0, "current_roster_retained_share_asof"] == 1.0


def test_uniform_and_elo_baselines_are_valid(modeling_outputs, tmp_path: Path) -> None:
    _, _, config, features, _ = modeling_outputs
    predictions = build_baseline_predictions(features, config)
    report = build_baseline_report(predictions, config)
    prediction_path = tmp_path / "predictions.parquet"
    report_path = tmp_path / "baseline.json"
    write_baseline_outputs(predictions, report, prediction_path, report_path)

    sums = predictions.groupby(["snapshot_id", "baseline_model"])["champion_probability"].sum()
    assert len(predictions) == 2182
    assert set(predictions["baseline_model"]) == {"uniform", "elo_strength"}
    assert sums.sub(1.0).abs().lt(1e-9).all()
    assert report["evaluated_snapshot_count"] == 93
    assert report["probability_validation"]["invalid_sum_group_count"] == 0
    assert report["elo_vs_uniform"]["log_loss_improvement_pct"] > 0
    assert report["elo_vs_uniform"]["brier_improvement_pct"] > 0
    assert prediction_path.exists()
    assert report_path.exists()
