from pathlib import Path

import pandas as pd
import pytest

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.features.matches import build_match_feature_report, build_match_features
from mpl_predictor.features.snapshots import load_feature_config
from mpl_predictor.models.baseline import build_baseline_predictions
from mpl_predictor.models.evaluation import build_model_evaluation_report, write_model_outputs
from mpl_predictor.models.online_learning import backtest_online_learning
from mpl_predictor.models.walk_forward import (
    load_model_config,
    walk_forward_champion_predictions,
    walk_forward_match_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "config" / "feature_config.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
SNAPSHOT_FEATURE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features" / "team_snapshot_features.parquet"
)


@pytest.fixture(scope="module")
def walk_forward_outputs():
    tables = load_canonical_tables(CANONICAL_DIR)
    feature_config = load_feature_config(FEATURE_CONFIG_PATH)
    model_config = load_model_config(MODEL_CONFIG_PATH)
    match_features = build_match_features(tables, feature_config)
    snapshot_features = pd.read_parquet(SNAPSHOT_FEATURE_PATH)
    baselines = build_baseline_predictions(snapshot_features, feature_config)
    match_predictions, match_folds = walk_forward_match_predictions(match_features, model_config)
    champion_predictions, champion_folds = walk_forward_champion_predictions(
        snapshot_features, baselines, model_config
    )
    report, frames = build_model_evaluation_report(
        match_predictions,
        champion_predictions,
        match_folds,
        champion_folds,
        model_config,
    )
    return {
        "match_features": match_features,
        "match_predictions": match_predictions,
        "champion_predictions": champion_predictions,
        "match_folds": match_folds,
        "champion_folds": champion_folds,
        "report": report,
        "frames": frames,
    }


def test_match_features_are_complete_and_time_safe(walk_forward_outputs) -> None:
    features = walk_forward_outputs["match_features"]
    report = build_match_feature_report(features)

    assert len(features) == 992
    assert report["blocking_issue_count"] == 0
    assert len(report["feature_columns"]) == 15
    assert features["team_a_win"].notna().all()
    assert features["elo_expected_team_a"].between(0, 1).all()


def test_match_walk_forward_uses_only_past_seasons(walk_forward_outputs) -> None:
    predictions = walk_forward_outputs["match_predictions"]
    folds = walk_forward_outputs["match_folds"]

    assert len(predictions) == 1472
    assert predictions["match_id"].nunique() == 736
    assert predictions["team_a_win_probability"].between(0, 1).all()
    assert set(predictions["model_name"]) == {
        "match_logistic_raw",
        "match_logistic_calibrated",
    }
    assert all(fold["training_season_max"] < fold["target_season"] for fold in folds)
    assert all(
        all(season < fold["target_season"] for season in fold["calibration_seasons"])
        for fold in folds
    )
    assert max(fold["max_symmetry_error"] for fold in folds) < 1e-3

    feature_config = load_feature_config(FEATURE_CONFIG_PATH)
    online_report = backtest_online_learning(predictions, feature_config["online_learning"])
    assert online_report["protocol"] == "season_reset_prequential_predict_then_update"
    assert online_report["match_count"] == 736
    assert online_report["historical_validation_passed"] is True
    assert online_report["adaptive_metrics"]["log_loss"] < online_report["base_metrics"]["log_loss"]


def test_champion_walk_forward_probabilities_and_calibration(walk_forward_outputs) -> None:
    predictions = walk_forward_outputs["champion_predictions"]
    folds = walk_forward_outputs["champion_folds"]
    report = walk_forward_outputs["report"]
    probability_sums = predictions.groupby(["snapshot_id", "model_name"])[
        "champion_probability"
    ].sum()

    assert len(predictions) == 3212
    assert predictions["snapshot_id"].nunique() == 93
    assert probability_sums.sub(1).abs().lt(1e-9).all()
    assert report["probability_validation"]["invalid_probability_sum_count"] == 0
    assert all(fold["training_season_max"] < fold["target_season"] for fold in folds)
    assert report["match_model"]["calibration_log_loss_improvement_pct"] > 0
    assert report["champion_snapshot_model"]["calibration_log_loss_improvement_pct"] > 0

    overall = {
        row["model_name"]: row for row in report["champion_snapshot_model"]["overall_metrics"]
    }
    assert (
        overall["snapshot_logistic_calibrated"]["multiclass_log_loss"]
        < overall["snapshot_logistic_raw"]["multiclass_log_loss"]
    )
    assert (
        overall["elo_strength"]["multiclass_log_loss"]
        < overall["snapshot_logistic_calibrated"]["multiclass_log_loss"]
    )


def test_walk_forward_outputs_can_be_written(walk_forward_outputs, tmp_path: Path) -> None:
    match_path = tmp_path / "match_predictions.parquet"
    champion_path = tmp_path / "champion_predictions.parquet"
    report_path = tmp_path / "evaluation.json"
    write_model_outputs(
        walk_forward_outputs["match_predictions"],
        walk_forward_outputs["champion_predictions"],
        walk_forward_outputs["report"],
        match_path,
        champion_path,
        report_path,
    )

    assert match_path.exists()
    assert champion_path.exists()
    assert report_path.exists()
