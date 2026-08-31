from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def adapt_probability(base_probability: float, scale: float) -> float:
    """Apply a symmetric confidence scale to a base probability."""
    probability = float(np.clip(base_probability, 1e-6, 1.0 - 1e-6))
    logit = np.log(probability / (1.0 - probability))
    adapted = 1.0 / (1.0 + np.exp(-float(scale) * logit))
    return float(np.clip(adapted, 1e-6, 1.0 - 1e-6))


@dataclass
class OnlineTemperatureLearner:
    """Learn a small confidence correction from prior live-season residuals."""

    enabled: bool
    learning_rate: float
    l2_regularization: float
    minimum_scale: float
    maximum_scale: float
    log_scale: float = 0.0
    observation_count: int = 0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "OnlineTemperatureLearner":
        values = config or {}
        learner = cls(
            enabled=bool(values.get("enabled", False)),
            learning_rate=float(values.get("learning_rate", 0.05)),
            l2_regularization=float(values.get("l2_regularization", 0.05)),
            minimum_scale=float(values.get("minimum_scale", 0.5)),
            maximum_scale=float(values.get("maximum_scale", 2.0)),
        )
        learner.validate()
        return learner

    @property
    def scale(self) -> float:
        return float(np.exp(self.log_scale))

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("Online learning_rate must be greater than zero.")
        if self.l2_regularization < 0:
            raise ValueError("Online l2_regularization cannot be negative.")
        if not 0 < self.minimum_scale <= 1.0 <= self.maximum_scale:
            raise ValueError(
                "Online learning scale bounds must satisfy 0 < minimum <= 1 <= maximum."
            )

    def predict(self, base_probability: float) -> float:
        if not self.enabled:
            return float(base_probability)
        return adapt_probability(base_probability, self.scale)

    def update(self, base_probability: float, actual_team_a_win: float) -> float:
        """Update only after a prediction; return the post-result confidence scale."""
        if actual_team_a_win not in {0.0, 1.0}:
            raise ValueError("Online learning target must be 0.0 or 1.0.")
        adapted_probability = self.predict(base_probability)
        if self.enabled:
            probability = float(np.clip(base_probability, 1e-6, 1.0 - 1e-6))
            base_logit = float(np.log(probability / (1.0 - probability)))
            gradient = (
                adapted_probability - actual_team_a_win
            ) * self.scale * base_logit + self.l2_regularization * self.log_scale
            self.log_scale -= self.learning_rate * gradient
            self.log_scale = float(
                np.clip(
                    self.log_scale,
                    np.log(self.minimum_scale),
                    np.log(self.maximum_scale),
                )
            )
        self.observation_count += 1
        return self.scale


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
    targets = labels.astype(float)
    return {
        "accuracy": float(np.mean((clipped >= 0.5) == targets)),
        "brier_score": float(np.mean((clipped - targets) ** 2)),
        "log_loss": float(
            -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        ),
    }


def backtest_online_learning(
    walk_forward_predictions: pd.DataFrame,
    config: dict[str, Any],
    model_name: str = "match_logistic_calibrated",
) -> dict[str, Any]:
    """Prequentially validate the online learner on leakage-safe historical folds."""
    frame = walk_forward_predictions.loc[
        walk_forward_predictions["model_name"].eq(model_name)
    ].copy()
    if frame.empty:
        raise ValueError(f"No walk-forward rows found for model {model_name!r}.")
    required = {
        "season",
        "completion_date",
        "match_id",
        "team_a_win",
        "team_a_win_probability",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(
            "Online learning backtest is missing columns: " + ", ".join(sorted(missing))
        )

    records: list[dict[str, Any]] = []
    season_metrics = []
    for season, group in frame.groupby("season", sort=True):
        learner = OnlineTemperatureLearner.from_config(config)
        group = group.sort_values(["completion_date", "match_id"])
        season_records = []
        for row in group.itertuples(index=False):
            base_probability = float(row.team_a_win_probability)
            target = float(row.team_a_win)
            adapted_probability = learner.predict(base_probability)
            record = {
                "season": int(season),
                "target": target,
                "base_probability": base_probability,
                "adapted_probability": adapted_probability,
            }
            records.append(record)
            season_records.append(record)
            learner.update(base_probability, target)
        season_frame = pd.DataFrame.from_records(season_records)
        labels = season_frame["target"].to_numpy(dtype=float)
        base = _probability_metrics(labels, season_frame["base_probability"].to_numpy())
        adaptive = _probability_metrics(labels, season_frame["adapted_probability"].to_numpy())
        season_metrics.append(
            {
                "season": int(season),
                "match_count": len(season_frame),
                "base": base,
                "adaptive": adaptive,
                "final_confidence_scale": learner.scale,
            }
        )

    result = pd.DataFrame.from_records(records)
    labels = result["target"].to_numpy(dtype=float)
    base = _probability_metrics(labels, result["base_probability"].to_numpy())
    adaptive = _probability_metrics(labels, result["adapted_probability"].to_numpy())
    return {
        "protocol": "season_reset_prequential_predict_then_update",
        "base_model": model_name,
        "season_min": int(result["season"].min()),
        "season_max": int(result["season"].max()),
        "match_count": len(result),
        "config": dict(config),
        "base_metrics": base,
        "adaptive_metrics": adaptive,
        "adaptive_minus_base": {
            metric: adaptive[metric] - base[metric]
            for metric in ("accuracy", "brier_score", "log_loss")
        },
        "historical_validation_passed": (
            adaptive["brier_score"] < base["brier_score"]
            and adaptive["log_loss"] < base["log_loss"]
        ),
        "metrics_by_season": season_metrics,
    }
