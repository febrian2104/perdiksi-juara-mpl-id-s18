from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mpl_predictor.analysis.common import write_json
from mpl_predictor.models.walk_forward import (
    _apply_platt,
    _augment_match_training,
    _fit_platt,
    _numeric,
    _pipeline,
)


def train_final_match_model(
    match_features: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit the selected calibrated logistic model on all completed S4-S17 matches."""
    columns = list(config["match_model"]["feature_columns"])
    first_season = int(match_features["season"].min())
    last_season = int(match_features["season"].max())
    minimum_base = int(config["calibration"]["minimum_base_seasons"])
    calibration_probabilities: list[float] = []
    calibration_labels: list[int] = []
    calibration_seasons = []

    for validation_season in range(first_season + minimum_base, last_season + 1):
        inner_train = match_features.loc[match_features["season"].lt(validation_season)]
        validation = match_features.loc[match_features["season"].eq(validation_season)]
        if inner_train.empty or validation.empty:
            continue
        inner_model = _pipeline(config)
        train_x, train_y = _augment_match_training(inner_train, columns)
        inner_model.fit(train_x, train_y)
        probabilities = inner_model.predict_proba(_numeric(validation, columns))[:, 1]
        labels = validation["team_a_win"].astype(int).to_numpy()
        calibration_probabilities.extend(probabilities.tolist())
        calibration_probabilities.extend((1.0 - probabilities).tolist())
        calibration_labels.extend(labels.tolist())
        calibration_labels.extend((1 - labels).tolist())
        calibration_seasons.append(validation_season)

    calibrator = _fit_platt(np.asarray(calibration_probabilities), np.asarray(calibration_labels))
    model = _pipeline(config)
    training_x, training_y = _augment_match_training(match_features, columns)
    model.fit(training_x, training_y)
    return {
        "artifact_version": "1.0",
        "model_version": config["model_version"],
        "model_name": "match_logistic_calibrated",
        "pipeline": model,
        "calibrator": calibrator,
        "feature_columns": columns,
        "training_season_min": first_season,
        "training_season_max": last_season,
        "training_match_count": len(match_features),
        "augmented_training_row_count": len(training_x),
        "calibration_seasons": calibration_seasons,
        "calibration_observation_count": len(calibration_labels),
        "symmetric_training_augmentation": True,
    }


def predict_match_probability(artifact: dict[str, Any], features: dict[str, Any]) -> float:
    """Predict team-A series win probability and enforce side symmetry."""
    columns = list(artifact["feature_columns"])
    frame = pd.DataFrame([{column: features.get(column) for column in columns}])
    numeric = _numeric(frame, columns)
    raw_forward = artifact["pipeline"].predict_proba(numeric)[:, 1]
    raw_reverse = artifact["pipeline"].predict_proba(-numeric)[:, 1]
    raw_symmetric = 0.5 * (raw_forward + (1.0 - raw_reverse))
    calibrated_forward = _apply_platt(artifact["calibrator"], raw_symmetric)
    calibrated_reverse = _apply_platt(artifact["calibrator"], 1.0 - raw_symmetric)
    probability = 0.5 * (calibrated_forward + (1.0 - calibrated_reverse))
    return float(np.clip(probability[0], 1e-6, 1 - 1e-6))


def _metric_lookup(report: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    return {str(row["model_name"]): row for row in report[section]["overall_metrics"]}


def build_final_model_report(
    artifact: dict[str, Any], evaluation_report: dict[str, Any]
) -> dict[str, Any]:
    match_metrics = _metric_lookup(evaluation_report, "match_model")
    champion_metrics = _metric_lookup(evaluation_report, "champion_snapshot_model")
    selected = match_metrics["match_logistic_calibrated"]
    return {
        "report_version": "1.0",
        "model_version": artifact["model_version"],
        "final_system": {
            "match_probability_model": "match_logistic_calibrated",
            "champion_probability_method": "monte_carlo_regular_season_and_playoffs",
            "direct_champion_ranking_benchmark": "elo_strength",
        },
        "selection": {
            "match_model_reason": (
                "Kalibrasi Platt dipilih karena log loss dan ECE walk-forward lebih baik "
                "daripada probabilitas logistic mentah."
            ),
            "selected_match_metrics": selected,
            "match_candidates": list(match_metrics.values()),
            "direct_champion_reason": (
                "Elo strength tetap menjadi benchmark ranking langsung karena memiliki "
                "multiclass log loss terbaik pada backtest champion."
            ),
            "direct_champion_candidates": list(champion_metrics.values()),
        },
        "training": {
            key: artifact[key]
            for key in (
                "training_season_min",
                "training_season_max",
                "training_match_count",
                "augmented_training_row_count",
                "calibration_seasons",
                "calibration_observation_count",
                "symmetric_training_augmentation",
            )
        },
        "feature_columns": artifact["feature_columns"],
        "data_guard": (
            "Hasil S18 tidak digunakan untuk fitting; hasil tersebut hanya memperbarui "
            "fitur live sebelum simulasi pertandingan tersisa."
        ),
    }


def write_final_model_outputs(
    artifact: dict[str, Any],
    report: dict[str, Any],
    artifact_path: Path,
    report_path: Path,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    write_json(report, report_path)


def load_final_match_model(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"pipeline", "calibrator", "feature_columns", "model_name"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Final model artifact is missing: {', '.join(sorted(missing))}")
    return artifact
