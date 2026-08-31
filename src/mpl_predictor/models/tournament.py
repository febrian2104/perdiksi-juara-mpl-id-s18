import math
from functools import lru_cache
from typing import Any

import numpy as np

CONFIRMED_FORMAT_STATUSES = {"official", "historical_assumption"}
SUPPORTED_RANKING_RULES = {
    "match_wins_desc",
    "match_losses_asc",
    "game_differential_desc",
    "game_wins_desc",
    "random_tiebreak",
}


def _participant_error(
    participant: Any,
    playoff_team_count: int,
    available_matches: set[str],
) -> str | None:
    if not isinstance(participant, dict):
        return "participant must be an object"
    source = participant.get("source")
    if source == "seed":
        seed = participant.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            return "seed participant must contain an integer seed"
        if not 1 <= seed <= playoff_team_count:
            return f"seed {seed} is outside 1..{playoff_team_count}"
        return None
    if source in {"winner", "loser"}:
        match_id = participant.get("match_id")
        if not isinstance(match_id, str) or not match_id:
            return f"{source} participant must contain match_id"
        if match_id not in available_matches:
            return f"{source} references unavailable match {match_id!r}"
        return None
    return f"unknown participant source {source!r}"


def validate_simulation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a season format before it can be used for simulation."""
    required = {
        "simulation_version",
        "season",
        "iterations",
        "random_seed",
        "format_confirmation",
        "regular_season",
        "playoffs",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Simulation config is missing: {', '.join(sorted(missing))}")

    season = config["season"]
    if not isinstance(season, int) or isinstance(season, bool) or season <= 0:
        raise ValueError("Simulation season must be a positive integer.")
    if not isinstance(config["iterations"], int) or config["iterations"] <= 0:
        raise ValueError("Simulation iterations must be a positive integer.")

    confirmation = config["format_confirmation"]
    if not isinstance(confirmation, dict):
        raise ValueError("format_confirmation must be an object.")
    confirmed_season = confirmation.get("season")
    if confirmed_season != season:
        raise ValueError(
            "format_confirmation.season must match simulation season: "
            f"{confirmed_season!r} != {season}."
        )
    status = confirmation.get("status")
    if status not in CONFIRMED_FORMAT_STATUSES:
        allowed = ", ".join(sorted(CONFIRMED_FORMAT_STATUSES))
        raise ValueError(
            f"Season {season} format is not confirmed. Set status to one of: {allowed}, "
            "and document the source basis before running predictions."
        )
    if not str(confirmation.get("basis", "")).strip():
        raise ValueError("format_confirmation.basis must document the format source.")

    regular = config["regular_season"]
    playoffs = config["playoffs"]
    if not isinstance(regular, dict) or not isinstance(playoffs, dict):
        raise ValueError("regular_season and playoffs must be objects.")
    playoff_team_count = regular.get("playoff_team_count")
    if (
        not isinstance(playoff_team_count, int)
        or isinstance(playoff_team_count, bool)
        or playoff_team_count < 2
    ):
        raise ValueError("regular_season.playoff_team_count must be at least two.")
    ranking_order = regular.get("ranking_order")
    if not isinstance(ranking_order, list) or not ranking_order:
        raise ValueError("regular_season.ranking_order must be a non-empty list.")
    unknown_ranking_rules = set(ranking_order) - SUPPORTED_RANKING_RULES
    if unknown_ranking_rules:
        raise ValueError(
            "Unsupported regular-season ranking rules: "
            f"{', '.join(sorted(unknown_ranking_rules))}."
        )
    if len(ranking_order) != len(set(ranking_order)):
        raise ValueError("regular_season.ranking_order cannot contain duplicates.")

    reference_best_of = playoffs.get("probability_reference_best_of")
    if (
        not isinstance(reference_best_of, int)
        or isinstance(reference_best_of, bool)
        or reference_best_of <= 0
        or reference_best_of % 2 == 0
    ):
        raise ValueError("playoffs.probability_reference_best_of must be a positive odd integer.")

    bracket = playoffs.get("bracket")
    if not isinstance(bracket, list) or not bracket:
        raise ValueError("playoffs.bracket must contain at least one match.")
    available_matches: set[str] = set()
    seed_reference_counts = {seed: 0 for seed in range(1, playoff_team_count + 1)}
    round_counts: dict[str, int] = {}
    for index, match in enumerate(bracket, start=1):
        if not isinstance(match, dict):
            raise ValueError(f"Bracket match {index} must be an object.")
        match_id = match.get("match_id")
        if not isinstance(match_id, str) or not match_id:
            raise ValueError(f"Bracket match {index} must have a non-empty match_id.")
        if match_id in available_matches:
            raise ValueError(f"Duplicate bracket match_id: {match_id!r}.")
        best_of = match.get("best_of")
        if (
            not isinstance(best_of, int)
            or isinstance(best_of, bool)
            or best_of <= 0
            or best_of % 2 == 0
        ):
            raise ValueError(f"Bracket match {match_id!r} best_of must be a positive odd integer.")
        round_name = match.get("round")
        if not isinstance(round_name, str) or not round_name:
            raise ValueError(f"Bracket match {match_id!r} must have a round name.")
        round_counts[round_name] = round_counts.get(round_name, 0) + 1
        for side in ("team_a", "team_b"):
            participant = match.get(side)
            error = _participant_error(participant, playoff_team_count, available_matches)
            if error:
                raise ValueError(f"Bracket match {match_id!r} {side}: {error}.")
            if participant["source"] == "seed":
                seed_reference_counts[int(participant["seed"])] += 1
        available_matches.add(match_id)

    championship_match_id = playoffs.get("championship_match_id")
    if championship_match_id not in available_matches:
        raise ValueError("playoffs.championship_match_id must reference a bracket match.")
    if bracket[-1]["match_id"] != championship_match_id:
        raise ValueError("The championship match must be the final configured bracket match.")
    invalid_seed_entries = {
        seed: count for seed, count in seed_reference_counts.items() if count != 1
    }
    if invalid_seed_entries:
        raise ValueError(
            "Every playoff seed must enter the bracket exactly through configuration; "
            f"invalid entry counts={invalid_seed_entries}."
        )

    return {
        "season": season,
        "format_status": status,
        "playoff_team_count": playoff_team_count,
        "bracket_match_count": len(bracket),
        "round_counts": round_counts,
        "championship_match_id": championship_match_id,
    }


def _best_of_win_probability(game_probability: float, best_of: int) -> float:
    wins_needed = best_of // 2 + 1
    return float(
        sum(
            math.comb(best_of, wins)
            * game_probability**wins
            * (1.0 - game_probability) ** (best_of - wins)
            for wins in range(wins_needed, best_of + 1)
        )
    )


@lru_cache(maxsize=1024)
def reference_series_to_game_probability(
    series_probability: float, reference_best_of: int
) -> float:
    """Infer a per-game probability from a calibrated reference series probability."""
    probability = float(np.clip(series_probability, 0.0, 1.0))
    if probability in {0.0, 1.0} or reference_best_of == 1:
        return probability
    low, high = 0.0, 1.0
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if _best_of_win_probability(midpoint, reference_best_of) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


@lru_cache(maxsize=2048)
def adjusted_series_probability(
    reference_probability: float,
    best_of: int,
    reference_best_of: int,
) -> float:
    """Convert a reference-series probability to the configured series length."""
    game_probability = reference_series_to_game_probability(
        reference_probability, reference_best_of
    )
    return _best_of_win_probability(game_probability, best_of)


def _resolve_participant(
    participant: dict[str, Any],
    seeds: np.ndarray,
    results: dict[str, tuple[int, int]],
) -> int:
    source = participant["source"]
    if source == "seed":
        return int(seeds[int(participant["seed"]) - 1])
    match_result = results[str(participant["match_id"])]
    return match_result[0] if source == "winner" else match_result[1]


def simulate_playoff_bracket(
    seeds: np.ndarray,
    probabilities: np.ndarray,
    rng: np.random.Generator,
    playoffs: dict[str, Any],
    *,
    include_trace: bool = False,
) -> tuple[int, tuple[int, int], list[dict[str, Any]]]:
    """Resolve a validated declarative bracket in configuration order."""
    results: dict[str, tuple[int, int]] = {}
    trace: list[dict[str, Any]] = []
    reference_best_of = int(playoffs["probability_reference_best_of"])
    championship_match_id = str(playoffs["championship_match_id"])
    finalists: tuple[int, int] | None = None
    for match in playoffs["bracket"]:
        team_a = _resolve_participant(match["team_a"], seeds, results)
        team_b = _resolve_participant(match["team_b"], seeds, results)
        if team_a == team_b:
            raise ValueError(f"Bracket match {match['match_id']!r} resolved to the same team.")
        best_of = int(match["best_of"])
        probability = adjusted_series_probability(
            float(probabilities[team_a, team_b]), best_of, reference_best_of
        )
        winner, loser = (team_a, team_b) if rng.random() < probability else (team_b, team_a)
        match_id = str(match["match_id"])
        results[match_id] = (winner, loser)
        if match_id == championship_match_id:
            finalists = (team_a, team_b)
        if include_trace:
            trace.append(
                {
                    "match_id": match_id,
                    "round": str(match["round"]),
                    "best_of": best_of,
                    "team_a": team_a,
                    "team_b": team_b,
                    "team_a_win_probability": probability,
                    "winner": winner,
                    "loser": loser,
                }
            )
    if finalists is None:
        raise ValueError("Configured championship match was not simulated.")
    return results[championship_match_id][0], finalists, trace
