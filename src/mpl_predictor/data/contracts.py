from collections.abc import Mapping
from typing import Final

BASE_TABLES: Final[frozenset[str]] = frozenset(
    {"championships", "drafts", "games", "matches", "players", "teams"}
)
PLAYER_STATS_TABLE: Final[str] = "player_season_stats"
FRANCHISE_START_SEASON: Final[int] = 4
HISTORICAL_SEASONS: Final[range] = range(1, 18)

REQUIRED_COLUMNS: Final[Mapping[str, frozenset[str]]] = {
    "championships": frozenset({"season", "team", "final_rank", "champion", "runner_up", "source"}),
    "drafts": frozenset(
        {"season", "match_id", "game_number", "team", "action", "order", "hero", "source"}
    ),
    "games": frozenset(
        {
            "season",
            "match_id",
            "game_number",
            "date",
            "team_a",
            "team_b",
            "winner",
            "duration_minutes",
            "source",
        }
    ),
    "matches": frozenset(
        {
            "season",
            "stage",
            "week",
            "date",
            "match_id",
            "team_a",
            "team_b",
            "team_a_score",
            "team_b_score",
            "winner",
            "best_of",
            "round",
            "source",
        }
    ),
    "players": frozenset({"season", "team", "player", "role", "status", "source"}),
    "player_season_stats": frozenset(
        {
            "season",
            "team",
            "player",
            "role",
            "stat_scope",
            "snapshot_label",
            "snapshot_date",
            "coverage_status",
            "source",
        }
    ),
    "teams": frozenset({"season", "team_id", "team_name", "source"}),
}


def expected_tables(season: int) -> frozenset[str]:
    """Return tables expected for a season without treating S1-S3 stats as missing."""
    if season < FRANCHISE_START_SEASON:
        return BASE_TABLES
    return BASE_TABLES | {PLAYER_STATS_TABLE}
