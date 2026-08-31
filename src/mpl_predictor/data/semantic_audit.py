import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.data.normalization import _winner_side_from_raw


@dataclass(frozen=True, slots=True)
class SemanticCheck:
    status: str
    code: str
    count: int
    message: str
    examples: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticAuditReport:
    row_counts: dict[str, int]
    checks: tuple[SemanticCheck, ...]
    coverage_by_season: dict[str, dict[str, int | float]]

    @property
    def errors(self) -> tuple[SemanticCheck, ...]:
        return tuple(check for check in self.checks if check.status == "error")

    @property
    def warnings(self) -> tuple[SemanticCheck, ...]:
        return tuple(check for check in self.checks if check.status == "warning")

    @property
    def information(self) -> tuple[SemanticCheck, ...]:
        return tuple(check for check in self.checks if check.status == "info")

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_counts": self.row_counts,
            "summary": {
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
                "info_count": len(self.information),
                "passed_check_count": sum(check.status == "pass" for check in self.checks),
            },
            "checks": [asdict(check) for check in self.checks],
            "coverage_by_season": self.coverage_by_season,
        }


def _check(
    status: str, code: str, count: int, message: str, examples: list[dict[str, Any]] | None = None
) -> SemanticCheck:
    return SemanticCheck(
        status=status,
        code=code,
        count=int(count),
        message=message,
        examples=tuple(examples or []),
    )


def _error_or_pass(
    code: str, count: int, message: str, examples: list[dict[str, Any]] | None = None
) -> SemanticCheck:
    return _check("error" if count else "pass", code, count, message, examples)


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in record.items()
    }


def audit_semantics(tables: dict[str, pd.DataFrame]) -> SemanticAuditReport:
    """Validate cross-table meaning and summarize known coverage limitations."""
    teams = tables["teams"]
    matches = tables["matches"]
    games = tables["games"]
    players = tables["players"]
    championships = tables["championships"]
    drafts = tables["drafts"]
    player_stats = tables["player_season_stats"]
    checks: list[SemanticCheck] = []

    duplicate_team_ids = int(teams.duplicated(["season", "team_id"]).sum())
    duplicate_team_names = int(teams.duplicated(["season", "team_name"]).sum())
    checks.append(
        _error_or_pass(
            "team_identity_unique",
            duplicate_team_ids + duplicate_team_names,
            "Team IDs and team names must be unique within each season.",
        )
    )

    duplicate_matches = int(matches.duplicated("match_id").sum())
    checks.append(
        _error_or_pass("match_id_unique", duplicate_matches, "Match IDs must be globally unique.")
    )

    invalid_match_team_refs = int(
        matches["team_a_id"].isna().sum() + matches["team_b_id"].isna().sum()
    )
    checks.append(
        _error_or_pass(
            "match_team_references",
            invalid_match_team_refs,
            "Every match team must exist in the same season's team table.",
        )
    )

    invalid_match_dates = int(matches["date"].isna().sum())
    checks.append(
        _error_or_pass(
            "match_dates_valid", invalid_match_dates, "Every match date must be parseable."
        )
    )

    raw_winner_sides = pd.Series(
        [
            _winner_side_from_raw(raw, team_a, team_b)
            for raw, team_a, team_b in zip(
                matches["winner_raw"], matches["team_a"], matches["team_b"], strict=True
            )
        ],
        index=matches.index,
        dtype="string",
    )
    winner_mismatch_mask = raw_winner_sides.ne(matches["winner_side"]).fillna(True)
    winner_mismatches = int(winner_mismatch_mask.sum())
    winner_examples = [
        _safe_record(record)
        for record in matches.loc[
            winner_mismatch_mask,
            ["season", "match_id", "winner_raw", "winner_side"],
        ]
        .head(5)
        .to_dict("records")
    ]
    checks.append(
        _error_or_pass(
            "match_winner_matches_score",
            winner_mismatches,
            "Raw match winners must agree with the normalized winner derived from scores.",
            winner_examples,
        )
    )

    duplicate_game_keys = int(games.duplicated(["match_id", "game_number"]).sum())
    checks.append(
        _error_or_pass(
            "game_key_unique",
            duplicate_game_keys,
            "The match ID and game number pair must be unique.",
        )
    )

    match_ids = set(matches["match_id"].dropna())
    missing_game_match_refs = int((~games["match_id"].isin(match_ids)).sum())
    checks.append(
        _error_or_pass(
            "game_match_references",
            missing_game_match_refs,
            "Every game must reference an existing match.",
        )
    )

    match_lookup = matches.set_index("match_id")
    game_team_mismatches: list[dict[str, Any]] = []
    date_differences: list[dict[str, Any]] = []
    for row in games.itertuples():
        if row.match_id not in match_lookup.index:
            continue
        match = match_lookup.loc[row.match_id]
        if {row.team_a_id, row.team_b_id} != {match.team_a_id, match.team_b_id}:
            game_team_mismatches.append(
                {"match_id": row.match_id, "game_number": int(row.game_number)}
            )
        if row.date != match.date:
            date_differences.append(
                {
                    "match_id": row.match_id,
                    "game_number": int(row.game_number),
                    "game_date": row.date.isoformat(),
                    "match_date": match.date.isoformat(),
                }
            )
    checks.append(
        _error_or_pass(
            "game_teams_match_series",
            len(game_team_mismatches),
            "Game teams must match the teams in the referenced match.",
            game_team_mismatches[:5],
        )
    )
    if date_differences:
        checks.append(
            _check(
                "info",
                "split_date_matches",
                len(date_differences),
                "Some series span multiple dates; game dates are preserved instead of overwritten.",
                date_differences[:5],
            )
        )

    game_counts = games.groupby("match_id").size()
    expected_game_counts = matches.set_index("match_id")[["team_a_score", "team_b_score"]].sum(
        axis=1
    )
    aligned_game_counts = game_counts.reindex(expected_game_counts.index, fill_value=0)
    game_count_mismatch = int((aligned_game_counts != expected_game_counts).sum())
    checks.append(
        _error_or_pass(
            "game_rows_match_series_score",
            game_count_mismatch,
            "The number of game rows must equal the sum of each match score.",
        )
    )

    known_tally_mismatches = 0
    for match_id, group in games.groupby("match_id"):
        if group["winner_side"].isna().any():
            continue
        team_a_wins = int(group["winner_side"].eq("team_a").sum())
        team_b_wins = int(group["winner_side"].eq("team_b").sum())
        match = match_lookup.loc[match_id]
        known_tally_mismatches += (team_a_wins, team_b_wins) != (
            int(match.team_a_score),
            int(match.team_b_score),
        )
    checks.append(
        _error_or_pass(
            "known_game_winners_match_score",
            known_tally_mismatches,
            "When every game winner is known, their totals must equal the match score.",
        )
    )

    championship_problems = 0
    for _, group in championships.groupby("season"):
        champion_teams = set(group.loc[group["champion"].fillna(False), "team_id"])
        runner_up_teams = set(group.loc[group["runner_up"].fillna(False), "team_id"])
        rank_one = set(group.loc[group["final_rank"].eq("1"), "team_id"])
        rank_two = set(group.loc[group["final_rank"].eq("2"), "team_id"])
        championship_problems += not (
            len(champion_teams) == 1
            and len(runner_up_teams) == 1
            and champion_teams == rank_one
            and runner_up_teams == rank_two
        )
    checks.append(
        _error_or_pass(
            "championship_labels_consistent",
            championship_problems,
            "Champion and runner-up flags must agree with final ranks one and two.",
        )
    )

    duplicate_players = int(players.duplicated(["season", "team_id", "player"]).sum())
    missing_player_team_refs = int(players["team_id"].isna().sum())
    checks.append(
        _error_or_pass(
            "player_roster_identity",
            duplicate_players + missing_player_team_refs,
            "Roster rows must have unique player/team pairs and valid team references.",
        )
    )

    non_aggregate_drafts = drafts.loc[~drafts["is_aggregate"].fillna(False)]
    game_keys = set(zip(games["match_id"], games["game_number"], strict=True))
    invalid_draft_refs = sum(
        (match_id, game_number) not in game_keys
        for match_id, game_number in zip(
            non_aggregate_drafts["match_id"],
            non_aggregate_drafts["game_number"],
            strict=True,
        )
    )
    checks.append(
        _error_or_pass(
            "draft_game_references",
            invalid_draft_refs,
            "Every non-aggregate draft row must reference an existing game.",
        )
    )

    missing_game_winners = int(games["winner_side"].isna().sum())
    if missing_game_winners:
        checks.append(
            _check(
                "warning",
                "game_winner_partial",
                missing_game_winners,
                "Missing game winners remain null and are not inferred from match-level scores.",
            )
        )

    missing_durations = int(games["duration_minutes"].isna().sum())
    if missing_durations:
        checks.append(
            _check(
                "warning",
                "game_duration_partial",
                missing_durations,
                "Game duration has partial season coverage and remains null when unavailable.",
            )
        )

    coverage_by_season: dict[str, dict[str, int | float]] = {}
    incomplete_draft_seasons = 0
    for season in sorted(int(value) for value in teams["season"].dropna().unique()):
        season_matches = matches.loc[matches["season"].eq(season)]
        season_games = games.loc[games["season"].eq(season)]
        season_drafts = non_aggregate_drafts.loc[non_aggregate_drafts["season"].eq(season)]
        season_players = players.loc[players["season"].eq(season)]
        season_stats = player_stats.loc[player_stats["season"].eq(season)]
        season_game_keys = set(
            zip(season_games["match_id"], season_games["game_number"], strict=True)
        )
        season_draft_keys = set(
            zip(season_drafts["match_id"], season_drafts["game_number"], strict=True)
        )
        covered_draft_games = len(season_game_keys & season_draft_keys)
        incomplete_draft_seasons += covered_draft_games < len(season_game_keys)
        coverage_by_season[str(season)] = {
            "matches": len(season_matches),
            "games": len(season_games),
            "games_with_winner": int(season_games["winner_side"].notna().sum()),
            "games_with_duration": int(season_games["duration_minutes"].notna().sum()),
            "games_with_draft": covered_draft_games,
            "roster_rows": len(season_players),
            "player_stat_rows": len(season_stats),
        }

    if incomplete_draft_seasons:
        checks.append(
            _check(
                "warning",
                "draft_coverage_partial",
                incomplete_draft_seasons,
                "Draft coverage is incomplete in some seasons; aggregate rows stay separate.",
            )
        )

    seasons_with_player_stats = int(player_stats["season"].nunique())
    checks.append(
        _check(
            "warning",
            "player_stats_partial",
            len(player_stats),
            "Player statistics contain selected snapshots in "
            f"{seasons_with_player_stats} seasons, not full rosters.",
        )
    )

    return SemanticAuditReport(
        row_counts={table: len(frame) for table, frame in tables.items()},
        checks=tuple(checks),
        coverage_by_season=coverage_by_season,
    )


def write_semantic_report(report: SemanticAuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
