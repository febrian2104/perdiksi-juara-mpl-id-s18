import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mpl_predictor.features.snapshots import model_feature_columns


def load_model_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "model_version",
        "evaluation",
        "logistic_regression",
        "calibration",
        "match_model",
        "champion_model",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Model config is missing sections: {', '.join(missing)}")
    return config


def _pipeline(config: dict[str, Any], class_weight: str | None = None) -> Pipeline:
    logistic = config["logistic_regression"]
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(logistic["C"]),
                    max_iter=int(logistic["max_iter"]),
                    solver=str(logistic["solver"]),
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return np.log(clipped / (1 - clipped))


def _fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(labels)) < 2:
        return None
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=42)
    calibrator.fit(_logit(probabilities).reshape(-1, 1), labels)
    return calibrator


def _apply_platt(calibrator: LogisticRegression | None, probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probabilities
    return calibrator.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


def _augment_match_training(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    original = _numeric(frame, columns)
    inverse = -original
    labels = frame["team_a_win"].astype(int)
    return (
        pd.concat([original, inverse], ignore_index=True),
        pd.concat([labels, 1 - labels], ignore_index=True),
    )


def walk_forward_match_predictions(
    match_features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Backtest raw and past-only Platt-calibrated match probabilities by target season."""
    columns = list(config["match_model"]["feature_columns"])
    start_season = int(config["evaluation"]["start_season"])
    minimum_base = int(config["calibration"]["minimum_base_seasons"])
    first_season = int(match_features["season"].min())
    outputs = []
    folds = []

    for target_season in range(start_season, int(match_features["season"].max()) + 1):
        train = match_features.loc[match_features["season"].lt(target_season)]
        test = match_features.loc[match_features["season"].eq(target_season)]
        if train.empty or test.empty:
            continue

        calibration_probabilities = []
        calibration_labels = []
        calibration_seasons = []
        for validation_season in range(first_season + minimum_base, target_season):
            inner_train = match_features.loc[match_features["season"].lt(validation_season)]
            inner_validation = match_features.loc[match_features["season"].eq(validation_season)]
            if inner_train.empty or inner_validation.empty:
                continue
            model = _pipeline(config)
            train_x, train_y = _augment_match_training(inner_train, columns)
            model.fit(train_x, train_y)
            probabilities = model.predict_proba(_numeric(inner_validation, columns))[:, 1]
            calibration_probabilities.extend(probabilities)
            calibration_probabilities.extend(1.0 - probabilities)
            calibration_labels.extend(inner_validation["team_a_win"].astype(int))
            calibration_labels.extend(1 - inner_validation["team_a_win"].astype(int))
            calibration_seasons.append(validation_season)

        calibrator = _fit_platt(
            np.asarray(calibration_probabilities), np.asarray(calibration_labels)
        )
        final_model = _pipeline(config)
        train_x, train_y = _augment_match_training(train, columns)
        final_model.fit(train_x, train_y)
        test_x = _numeric(test, columns)
        raw_probabilities = final_model.predict_proba(test_x)[:, 1]
        calibrated_probabilities = _apply_platt(calibrator, raw_probabilities)
        inverse_probabilities = final_model.predict_proba(-test_x)[:, 1]
        symmetry_error = np.abs(raw_probabilities + inverse_probabilities - 1.0)

        base_columns = [
            "season",
            "stage",
            "week",
            "match_id",
            "completion_date",
            "team_a_id",
            "team_b_id",
            "team_a_franchise_slot_id",
            "team_b_franchise_slot_id",
            "team_a_win",
        ]
        for model_name, probabilities in (
            ("match_logistic_raw", raw_probabilities),
            ("match_logistic_calibrated", calibrated_probabilities),
        ):
            prediction = test[base_columns].copy()
            prediction["model_name"] = model_name
            prediction["team_a_win_probability"] = probabilities
            prediction["predicted_team_a_win"] = probabilities >= 0.5
            prediction["training_season_min"] = int(train["season"].min())
            prediction["training_season_max"] = int(train["season"].max())
            outputs.append(prediction)

        folds.append(
            {
                "target_season": target_season,
                "training_season_min": int(train["season"].min()),
                "training_season_max": int(train["season"].max()),
                "training_match_count": len(train),
                "test_match_count": len(test),
                "calibration_seasons": calibration_seasons,
                "calibration_observation_count": len(calibration_labels),
                "mean_symmetry_error": round(float(symmetry_error.mean()), 10),
                "max_symmetry_error": round(float(symmetry_error.max()), 10),
            }
        )

    result = pd.concat(outputs, ignore_index=True)
    result["season"] = result["season"].astype("Int64")
    result["week"] = result["week"].astype("Int64")
    result["team_a_win"] = result["team_a_win"].astype("Int64")
    result["predicted_team_a_win"] = result["predicted_team_a_win"].astype("boolean")
    return result.sort_values(["season", "match_id", "model_name"]).reset_index(drop=True), folds


def _season_weights(frame: pd.DataFrame) -> np.ndarray:
    snapshot_counts = frame.groupby("season")["snapshot_id"].transform("nunique")
    return 1.0 / snapshot_counts.to_numpy(dtype=float)


def _fit_champion_model(
    frame: pd.DataFrame,
    columns: list[str],
    config: dict[str, Any],
) -> Pipeline:
    class_weight = config["champion_model"]["class_weight"]
    model = _pipeline(config, class_weight=class_weight)
    model.fit(
        _numeric(frame, columns),
        frame["champion"].astype(int),
        model__sample_weight=_season_weights(frame),
    )
    return model


def _softmax_by_snapshot(frame: pd.DataFrame, scores: np.ndarray, temperature: float) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    snapshot_values = frame["snapshot_id"].to_numpy()
    for snapshot_id in pd.unique(snapshot_values):
        positions = np.flatnonzero(snapshot_values == snapshot_id)
        scaled = scores[positions] / temperature
        scaled -= scaled.max()
        strengths = np.exp(scaled)
        result[positions] = strengths / strengths.sum()
    return result


def _temperature_loss(
    temperature: float,
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> float:
    probabilities = _softmax_by_snapshot(frame, scores, temperature)
    champion_mask = frame["champion"].astype(bool).to_numpy()
    return -float(np.log(np.clip(probabilities[champion_mask], 1e-15, 1)).mean())


def _fit_temperature(
    frame: pd.DataFrame,
    scores: np.ndarray,
    config: dict[str, Any],
) -> float:
    if frame.empty:
        return 1.0
    lower = float(config["calibration"]["temperature_min"])
    upper = float(config["calibration"]["temperature_max"])
    result = minimize_scalar(
        _temperature_loss,
        bounds=(lower, upper),
        method="bounded",
        args=(frame, scores),
    )
    return float(result.x) if result.success else 1.0


def _champion_model_columns(features: pd.DataFrame, prediction_type: str) -> list[str]:
    columns = [
        column
        for column in model_feature_columns()
        if column in features and not features[column].isna().all()
    ]
    if prediction_type == "weekly":
        columns.append("completed_week")
    return columns


def walk_forward_champion_predictions(
    features: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Backtest group-normalized champion probabilities with past-only temperature fitting."""
    start_season = int(config["evaluation"]["start_season"])
    minimum_base = int(config["calibration"]["minimum_base_seasons"])
    first_season = int(features["season"].min())
    outputs = []
    folds = []

    for target_season in range(start_season, int(features["season"].max()) + 1):
        for prediction_type in ("preseason", "weekly"):
            typed = features.loc[features["prediction_type"].eq(prediction_type)]
            train = typed.loc[typed["season"].lt(target_season)]
            test = typed.loc[typed["season"].eq(target_season)]
            if train.empty or test.empty:
                continue
            columns = _champion_model_columns(typed, prediction_type)
            oof_frames = []
            oof_scores = []
            calibration_seasons = []
            for validation_season in range(first_season + minimum_base, target_season):
                inner_train = typed.loc[typed["season"].lt(validation_season)]
                inner_validation = typed.loc[typed["season"].eq(validation_season)]
                if inner_train.empty or inner_validation.empty:
                    continue
                inner_model = _fit_champion_model(inner_train, columns, config)
                scores = inner_model.decision_function(_numeric(inner_validation, columns))
                oof_frames.append(inner_validation)
                oof_scores.extend(scores)
                calibration_seasons.append(validation_season)

            calibration_frame = (
                pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
            )
            temperature = _fit_temperature(
                calibration_frame,
                np.asarray(oof_scores, dtype=float),
                config,
            )
            final_model = _fit_champion_model(train, columns, config)
            scores = final_model.decision_function(_numeric(test, columns))
            raw_probabilities = _softmax_by_snapshot(test, scores, 1.0)
            calibrated_probabilities = _softmax_by_snapshot(test, scores, temperature)

            base_columns = [
                "snapshot_id",
                "season",
                "prediction_type",
                "completed_week",
                "feature_cutoff_date",
                "team_id",
                "team_name",
                "organization_id",
                "franchise_slot_id",
                "target_available",
                "champion",
            ]
            for model_name, probabilities in (
                ("snapshot_logistic_raw", raw_probabilities),
                ("snapshot_logistic_calibrated", calibrated_probabilities),
            ):
                prediction = test[base_columns].copy()
                prediction["model_name"] = model_name
                prediction["champion_probability"] = probabilities
                prediction["model_score"] = scores
                prediction["temperature"] = temperature if "calibrated" in model_name else 1.0
                prediction["training_season_min"] = int(train["season"].min())
                prediction["training_season_max"] = int(train["season"].max())
                outputs.append(prediction)

            folds.append(
                {
                    "target_season": target_season,
                    "prediction_type": prediction_type,
                    "training_season_min": int(train["season"].min()),
                    "training_season_max": int(train["season"].max()),
                    "training_row_count": len(train),
                    "test_snapshot_count": int(test["snapshot_id"].nunique()),
                    "test_row_count": len(test),
                    "feature_count": len(columns),
                    "calibration_seasons": calibration_seasons,
                    "calibration_snapshot_count": (
                        int(calibration_frame["snapshot_id"].nunique())
                        if not calibration_frame.empty
                        else 0
                    ),
                    "temperature": round(temperature, 6),
                }
            )

    predictions = pd.concat(outputs, ignore_index=True)
    baseline = baseline_predictions.loc[baseline_predictions["season"].ge(start_season)].rename(
        columns={"baseline_model": "model_name"}
    )
    baseline = baseline[
        [
            "snapshot_id",
            "season",
            "prediction_type",
            "completed_week",
            "feature_cutoff_date",
            "team_id",
            "team_name",
            "organization_id",
            "franchise_slot_id",
            "target_available",
            "champion",
            "model_name",
            "champion_probability",
        ]
    ].copy()
    baseline["model_score"] = pd.NA
    baseline["temperature"] = pd.NA
    baseline["training_season_min"] = pd.NA
    baseline["training_season_max"] = pd.NA
    predictions = pd.concat([predictions, baseline], ignore_index=True)
    predictions["probability_rank"] = predictions.groupby(["snapshot_id", "model_name"])[
        "champion_probability"
    ].rank(method="average", ascending=False)
    predictions["season"] = predictions["season"].astype("Int64")
    predictions["completed_week"] = predictions["completed_week"].astype("Int64")
    predictions["champion"] = predictions["champion"].astype("boolean")
    predictions["target_available"] = predictions["target_available"].astype("boolean")
    return predictions.sort_values(
        ["season", "snapshot_id", "model_name", "probability_rank", "team_id"]
    ).reset_index(drop=True), folds
