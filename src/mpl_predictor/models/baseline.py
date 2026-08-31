from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json


def build_baseline_predictions(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create uniform and Elo-strength champion probabilities for every snapshot."""
    scale = float(config["baseline"]["elo_probability_scale"])
    prediction_frames = []
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
        "elo_rating",
    ]

    for _, snapshot in features.groupby("snapshot_id", sort=False):
        uniform = snapshot[base_columns].copy()
        uniform["baseline_model"] = "uniform"
        uniform["champion_probability"] = 1.0 / len(uniform)
        prediction_frames.append(uniform)

        elo = snapshot[base_columns].copy()
        centered_rating = elo["elo_rating"] - elo["elo_rating"].max()
        strength = np.power(10.0, centered_rating / scale)
        elo["baseline_model"] = "elo_strength"
        elo["champion_probability"] = strength / strength.sum()
        prediction_frames.append(elo)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["probability_rank"] = predictions.groupby(["snapshot_id", "baseline_model"])[
        "champion_probability"
    ].rank(method="average", ascending=False)
    predictions["season"] = predictions["season"].astype("Int64")
    predictions["completed_week"] = predictions["completed_week"].astype("Int64")
    predictions["target_available"] = predictions["target_available"].astype("boolean")
    predictions["champion"] = predictions["champion"].astype("boolean")
    return predictions.sort_values(
        ["season", "snapshot_id", "baseline_model", "probability_rank", "team_id"]
    ).reset_index(drop=True)


def _evaluate_snapshots(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    available = predictions.loc[predictions["target_available"]].copy()
    for (snapshot_id, model), group in available.groupby(
        ["snapshot_id", "baseline_model"], sort=False
    ):
        champion = group.loc[group["champion"]]
        if len(champion) != 1:
            raise ValueError(
                f"Expected one champion in {snapshot_id}/{model}, found {len(champion)}"
            )
        champion_row = champion.iloc[0]
        probability = float(champion_row["champion_probability"])
        labels = group["champion"].astype(float).to_numpy()
        probabilities = group["champion_probability"].to_numpy()
        maximum = float(group["champion_probability"].max())
        unique_top = int(np.isclose(group["champion_probability"], maximum).sum()) == 1
        records.append(
            {
                "snapshot_id": snapshot_id,
                "season": int(champion_row["season"]),
                "prediction_type": champion_row["prediction_type"],
                "completed_week": champion_row["completed_week"],
                "baseline_model": model,
                "champion_probability": probability,
                "champion_rank": float(champion_row["probability_rank"]),
                "multiclass_log_loss": -float(np.log(max(probability, 1e-15))),
                "multiclass_brier_score": float(np.square(probabilities - labels).sum()),
                "top_1_correct": int(unique_top and bool(np.isclose(probability, maximum))),
                "top_3_correct": int(float(champion_row["probability_rank"]) <= 3.0),
                "reciprocal_rank": 1.0 / float(champion_row["probability_rank"]),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["completed_week"] = result["completed_week"].astype("Int64")
    return result


def _aggregate_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            snapshot_count=("snapshot_id", "nunique"),
            mean_champion_probability=("champion_probability", "mean"),
            multiclass_log_loss=("multiclass_log_loss", "mean"),
            multiclass_brier_score=("multiclass_brier_score", "mean"),
            mean_champion_rank=("champion_rank", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            top_1_accuracy=("top_1_correct", "mean"),
            top_3_accuracy=("top_3_correct", "mean"),
        )
        .round(6)
    )


def build_baseline_report(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    tolerance = float(config["baseline"]["probability_tolerance"])
    evaluation_start = int(config["baseline"]["evaluation_start_season"])
    probability_sums = predictions.groupby(["snapshot_id", "baseline_model"])[
        "champion_probability"
    ].sum()
    invalid_probability_groups = int((probability_sums.sub(1.0).abs() > tolerance).sum())
    all_evaluations = _evaluate_snapshots(predictions)
    evaluation = all_evaluations.loc[all_evaluations["season"].ge(evaluation_start)].copy()
    overall = _aggregate_metrics(evaluation, ["baseline_model"])
    by_type = _aggregate_metrics(evaluation, ["baseline_model", "prediction_type"])
    weekly = evaluation.loc[evaluation["prediction_type"].eq("weekly")]
    by_week = _aggregate_metrics(weekly, ["baseline_model", "completed_week"])

    overall_lookup = overall.set_index("baseline_model")
    uniform_loss = float(overall_lookup.loc["uniform", "multiclass_log_loss"])
    elo_loss = float(overall_lookup.loc["elo_strength", "multiclass_log_loss"])
    uniform_brier = float(overall_lookup.loc["uniform", "multiclass_brier_score"])
    elo_brier = float(overall_lookup.loc["elo_strength", "multiclass_brier_score"])
    return {
        "report_version": "1.0",
        "baseline_models": {
            "uniform": "Probabilitas sama besar untuk semua tim aktif.",
            "elo_strength": (
                "Probabilitas dinormalisasi dari strength 10^(Elo / scale); belum "
                "menggunakan model playoff atau kalibrasi."
            ),
        },
        "configuration": {
            "evaluation_start_season": evaluation_start,
            "evaluation_end_season": int(evaluation["season"].max()),
            "elo_probability_scale": float(config["baseline"]["elo_probability_scale"]),
        },
        "prediction_row_count": len(predictions),
        "evaluated_snapshot_count": int(evaluation["snapshot_id"].nunique()),
        "probability_validation": {
            "group_count": len(probability_sums),
            "invalid_sum_group_count": invalid_probability_groups,
            "tolerance": tolerance,
        },
        "overall_metrics": dataframe_records(overall),
        "metrics_by_prediction_type": dataframe_records(by_type),
        "weekly_metrics_by_completed_week": dataframe_records(by_week),
        "elo_vs_uniform": {
            "log_loss_improvement_pct": round(100 * (uniform_loss - elo_loss) / uniform_loss, 4),
            "brier_improvement_pct": round(100 * (uniform_brier - elo_brier) / uniform_brier, 4),
        },
        "interpretation_guard": (
            "Baseline hanya menjadi pembanding. Hasil ini belum merupakan prediksi final "
            "Season 18 dan belum melalui kalibrasi atau simulasi bracket."
        ),
    }


def write_baseline_outputs(
    predictions: pd.DataFrame,
    report: dict[str, Any],
    prediction_path: Path,
    report_path: Path,
) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)
    write_json(report, report_path)
