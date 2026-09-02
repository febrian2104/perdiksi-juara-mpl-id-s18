from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from mpl_predictor.analysis.common import dataframe_records, write_json


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        value += mask.sum() / total * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value)


def _calibration_coefficients(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(logits, labels)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def _reliability_table(
    frame: pd.DataFrame,
    label_column: str,
    probability_column: str,
    bins: int,
    scope: str,
) -> pd.DataFrame:
    records = []
    edges = np.linspace(0, 1, bins + 1)
    for model_name, group in frame.groupby("model_name"):
        probabilities = group[probability_column].to_numpy(dtype=float)
        labels = group[label_column].astype(int).to_numpy()
        assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
        for index in range(bins):
            mask = assignments == index
            if not mask.any():
                continue
            records.append(
                {
                    "scope": scope,
                    "model_name": model_name,
                    "bin_index": index + 1,
                    "bin_lower": edges[index],
                    "bin_upper": edges[index + 1],
                    "observation_count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "observed_rate": float(labels[mask].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def evaluate_match_predictions(
    predictions: pd.DataFrame, bins: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    for model_name, group in predictions.groupby("model_name"):
        labels = group["team_a_win"].astype(int).to_numpy()
        probabilities = group["team_a_win_probability"].to_numpy(dtype=float)
        intercept, slope = _calibration_coefficients(labels, probabilities)
        records.append(
            {
                "model_name": model_name,
                "match_count": len(group),
                "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
                "brier_score": brier_score_loss(labels, probabilities),
                "roc_auc": roc_auc_score(labels, probabilities),
                "accuracy": accuracy_score(labels, probabilities >= 0.5),
                "ece": _ece(labels, probabilities, bins),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
    overall = pd.DataFrame.from_records(records).round(6)

    by_stage_records = []
    for (model_name, stage), group in predictions.groupby(["model_name", "stage"]):
        labels = group["team_a_win"].astype(int).to_numpy()
        probabilities = group["team_a_win_probability"].to_numpy(dtype=float)
        by_stage_records.append(
            {
                "model_name": model_name,
                "stage": stage,
                "match_count": len(group),
                "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
                "brier_score": brier_score_loss(labels, probabilities),
                "roc_auc": roc_auc_score(labels, probabilities),
                "accuracy": accuracy_score(labels, probabilities >= 0.5),
                "ece": _ece(labels, probabilities, bins),
            }
        )
    by_stage = pd.DataFrame.from_records(by_stage_records).round(6)
    reliability = _reliability_table(
        predictions,
        "team_a_win",
        "team_a_win_probability",
        bins,
        "match",
    )
    return overall, by_stage, reliability


def _snapshot_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (snapshot_id, model_name), group in predictions.groupby(
        ["snapshot_id", "model_name"], sort=False
    ):
        champion = group.loc[group["champion"]]
        if len(champion) != 1:
            raise ValueError(
                f"Expected one champion for {snapshot_id}/{model_name}, found {len(champion)}"
            )
        row = champion.iloc[0]
        probability = float(row["champion_probability"])
        probabilities = group["champion_probability"].to_numpy(dtype=float)
        labels = group["champion"].astype(int).to_numpy()
        maximum = float(probabilities.max())
        unique_top = int(np.isclose(probabilities, maximum).sum()) == 1
        rank = float(row["probability_rank"])
        records.append(
            {
                "snapshot_id": snapshot_id,
                "season": int(row["season"]),
                "prediction_type": row["prediction_type"],
                "completed_week": row["completed_week"],
                "model_name": model_name,
                "champion_probability": probability,
                "champion_rank": rank,
                "multiclass_log_loss": -np.log(max(probability, 1e-15)),
                "multiclass_brier_score": float(np.square(probabilities - labels).sum()),
                "top_1_correct": int(unique_top and np.isclose(probability, maximum)),
                "top_3_correct": int(rank <= 3),
                "reciprocal_rank": 1.0 / rank,
            }
        )
    result = pd.DataFrame.from_records(records)
    result["completed_week"] = result["completed_week"].astype("Int64")
    return result


def _aggregate_snapshot_metrics(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        metrics.groupby(group_columns, dropna=False, as_index=False)
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


def evaluate_champion_predictions(
    predictions: pd.DataFrame, bins: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = _snapshot_metrics(predictions)
    overall = _aggregate_snapshot_metrics(metrics, ["model_name"])
    by_type = _aggregate_snapshot_metrics(metrics, ["model_name", "prediction_type"])
    by_week = _aggregate_snapshot_metrics(
        metrics.loc[metrics["prediction_type"].eq("weekly")],
        ["model_name", "completed_week"],
    )

    calibration_records = []
    for (model_name, prediction_type), group in predictions.groupby(
        ["model_name", "prediction_type"]
    ):
        labels = group["champion"].astype(int).to_numpy()
        probabilities = group["champion_probability"].to_numpy(dtype=float)
        intercept, slope = _calibration_coefficients(labels, probabilities)
        calibration_records.append(
            {
                "model_name": model_name,
                "prediction_type": prediction_type,
                "team_snapshot_count": len(group),
                "ece": _ece(labels, probabilities, bins),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
    calibration = pd.DataFrame.from_records(calibration_records).round(6)
    reliability = _reliability_table(
        predictions,
        "champion",
        "champion_probability",
        bins,
        "champion",
    )
    return overall, by_type, by_week, calibration, reliability


def _metric_improvement(
    frame: pd.DataFrame,
    raw_model: str,
    calibrated_model: str,
    metric: str,
) -> float:
    lookup = frame.set_index("model_name")
    raw = float(lookup.loc[raw_model, metric])
    calibrated = float(lookup.loc[calibrated_model, metric])
    return round(100 * (raw - calibrated) / raw, 4)


def _match_challenger_comparison(overall: pd.DataFrame) -> list[dict[str, Any]]:
    calibrated = overall.loc[overall["model_name"].str.endswith("_calibrated")].copy()
    calibrated = calibrated.sort_values(
        ["log_loss", "brier_score", "ece", "model_name"], ignore_index=True
    )
    calibrated["log_loss_rank"] = np.arange(1, len(calibrated) + 1)
    baseline = calibrated.loc[calibrated["model_name"].eq("match_logistic_calibrated")]
    if baseline.empty:
        raise ValueError("Calibrated logistic baseline is missing from match predictions.")
    baseline_row = baseline.iloc[0]
    calibrated["log_loss_delta_vs_logistic"] = calibrated["log_loss"] - float(
        baseline_row["log_loss"]
    )
    calibrated["brier_delta_vs_logistic"] = calibrated["brier_score"] - float(
        baseline_row["brier_score"]
    )
    return dataframe_records(calibrated)


def _best_match_variant_by_family(overall: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = overall.copy()
    candidates["model_family"] = (
        candidates["model_name"]
        .str.removeprefix("match_")
        .str.removesuffix("_calibrated")
        .str.removesuffix("_raw")
    )
    best_rows = candidates.loc[candidates.groupby("model_family")["log_loss"].idxmin()].copy()
    best_rows = best_rows.sort_values(
        ["log_loss", "brier_score", "ece", "model_name"], ignore_index=True
    )
    best_rows["log_loss_rank"] = np.arange(1, len(best_rows) + 1)
    logistic = best_rows.loc[best_rows["model_family"].eq("logistic")]
    if logistic.empty:
        raise ValueError("Logistic model family is missing from match predictions.")
    baseline = logistic.iloc[0]
    best_rows["log_loss_delta_vs_logistic"] = best_rows["log_loss"] - float(baseline["log_loss"])
    best_rows["brier_delta_vs_logistic"] = best_rows["brier_score"] - float(baseline["brier_score"])
    return dataframe_records(best_rows)


def build_model_evaluation_report(
    match_predictions: pd.DataFrame,
    champion_predictions: pd.DataFrame,
    match_folds: list[dict[str, Any]],
    champion_folds: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    bins = int(config["evaluation"]["ece_bins"])
    tolerance = float(config["evaluation"]["probability_tolerance"])
    match_overall, match_by_stage, match_reliability = evaluate_match_predictions(
        match_predictions, bins
    )
    (
        champion_overall,
        champion_by_type,
        champion_by_week,
        champion_calibration,
        champion_reliability,
    ) = evaluate_champion_predictions(champion_predictions, bins)

    sums = champion_predictions.groupby(["snapshot_id", "model_name"])["champion_probability"].sum()
    invalid_sums = int(sums.sub(1.0).abs().gt(tolerance).sum())
    report = {
        "report_version": "1.0",
        "model_version": config["model_version"],
        "evaluation_scope": {
            "start_season": int(config["evaluation"]["start_season"]),
            "end_season": int(match_predictions["season"].max()),
            "match_count": int(match_predictions["match_id"].nunique()),
            "snapshot_count": int(champion_predictions["snapshot_id"].nunique()),
            "team_snapshot_count": int(
                champion_predictions[["snapshot_id", "franchise_slot_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "leakage_policy": {
            "outer_fold": "Target season S hanya dilatih dari season < S.",
            "calibration": (
                "Calibrator hanya memakai prediksi out-of-fold dari season sebelum target."
            ),
            "snapshot_alignment": (
                "Model pramusim dan mingguan dipisah; model mingguan memakai completed_week."
            ),
        },
        "probability_validation": {
            "snapshot_model_group_count": len(sums),
            "invalid_probability_sum_count": invalid_sums,
            "tolerance": tolerance,
        },
        "match_model": {
            "features": config["match_model"]["feature_columns"],
            "candidate_families": config["match_model"]["candidate_families"],
            "overall_metrics": dataframe_records(match_overall),
            "metrics_by_stage": dataframe_records(match_by_stage),
            "calibrated_challenger_comparison": _match_challenger_comparison(match_overall),
            "best_variant_by_family": _best_match_variant_by_family(match_overall),
            "calibration_log_loss_improvement_pct": _metric_improvement(
                match_overall,
                "match_logistic_raw",
                "match_logistic_calibrated",
                "log_loss",
            ),
            "folds": match_folds,
        },
        "champion_snapshot_model": {
            "overall_metrics": dataframe_records(champion_overall),
            "metrics_by_prediction_type": dataframe_records(champion_by_type),
            "weekly_metrics_by_completed_week": dataframe_records(champion_by_week),
            "calibration_metrics": dataframe_records(champion_calibration),
            "calibration_log_loss_improvement_pct": _metric_improvement(
                champion_overall,
                "snapshot_logistic_raw",
                "snapshot_logistic_calibrated",
                "multiclass_log_loss",
            ),
            "folds": champion_folds,
        },
        "selection_guard": (
            "Hasil ini adalah backtest. Pemilihan model final dan simulasi playoff tetap "
            "dilakukan pada tahap berikutnya."
        ),
    }
    frames = {
        "match_overall": match_overall,
        "match_reliability": match_reliability,
        "champion_overall": champion_overall,
        "champion_by_week": champion_by_week,
        "champion_reliability": champion_reliability,
    }
    return report, frames


def write_model_outputs(
    match_predictions: pd.DataFrame,
    champion_predictions: pd.DataFrame,
    report: dict[str, Any],
    match_path: Path,
    champion_path: Path,
    report_path: Path,
) -> None:
    match_path.parent.mkdir(parents=True, exist_ok=True)
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    match_predictions.to_parquet(match_path, index=False)
    champion_predictions.to_parquet(champion_path, index=False)
    write_json(report, report_path)


def write_evaluation_figures(frames: dict[str, pd.DataFrame], directory: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs = []

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, key, title in (
        (axes[0], "match_reliability", "Kalibrasi probabilitas match"),
        (axes[1], "champion_reliability", "Kalibrasi probabilitas juara"),
    ):
        table = frames[key]
        for model_name, group in table.groupby("model_name"):
            axis.plot(
                group["mean_probability"],
                group["observed_rate"],
                marker="o",
                label=model_name,
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axis.set(title=title, xlabel="Rata-rata probabilitas", ylabel="Frekuensi aktual")
        axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    path = directory / "model_calibration_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    weekly = frames["champion_by_week"]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(
        data=weekly,
        x="completed_week",
        y="multiclass_log_loss",
        hue="model_name",
        marker="o",
        ax=axis,
    )
    axis.set(
        title="Walk-forward champion log loss per minggu",
        xlabel="Completed week",
        ylabel="Multiclass log loss",
    )
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    path = directory / "walk_forward_log_loss_by_week.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)
    return outputs
