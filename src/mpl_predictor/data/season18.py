import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.data.identity import canonical_player_id, player_key

SCHEDULE_URL = "https://id-mpl.com/id/schedule"
TEAM_URL_TEMPLATE = "https://id-mpl.com/en/team/{slug}"
TEAM_METADATA = {
    "AE": {
        "slug": "ae",
        "team_name": "Alter Ego Esports",
        "organization_id": "AE",
        "franchise_slot_id": "MPLID_SLOT_ALTER_EGO",
    },
    "BTR": {
        "slug": "btr",
        "team_name": "Bigetron by Vitality",
        "organization_id": "BTR",
        "franchise_slot_id": "MPLID_SLOT_BIGETRON",
    },
    "DEWA": {
        "slug": "dewa",
        "team_name": "Dewa United Esports",
        "organization_id": "DEWA",
        "franchise_slot_id": "MPLID_SLOT_DEWA",
    },
    "EVOS": {
        "slug": "evos",
        "team_name": "EVOS",
        "organization_id": "EVOS",
        "franchise_slot_id": "MPLID_SLOT_EVOS",
    },
    "GEEK": {
        "slug": "geek",
        "team_name": "Geek Fam",
        "organization_id": "GEEK",
        "franchise_slot_id": "MPLID_SLOT_GEEK",
    },
    "NAVI": {
        "slug": "navi",
        "team_name": "NAVI",
        "organization_id": "NAVI",
        "franchise_slot_id": "MPLID_SLOT_AEROWOLF_REBELLION_NAVI",
    },
    "ONIC": {
        "slug": "onic",
        "team_name": "ONIC",
        "organization_id": "ONIC",
        "franchise_slot_id": "MPLID_SLOT_ONIC",
    },
    "RRQ": {
        "slug": "rrq",
        "team_name": "RRQ Hoshi",
        "organization_id": "RRQ",
        "franchise_slot_id": "MPLID_SLOT_RRQ",
    },
    "TLID": {
        "slug": "tlid",
        "team_name": "Team Liquid ID",
        "organization_id": "TLID",
        "franchise_slot_id": "MPLID_SLOT_AURA_LIQUID",
    },
}

MONTHS_ID = {
    "Januari": 1,
    "Februari": 2,
    "Maret": 3,
    "April": 4,
    "Mei": 5,
    "Juni": 6,
    "Juli": 7,
    "Agustus": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Desember": 12,
}

ROLE_MAP = {
    "exp lane": "exp_lane",
    "jungle": "jungler",
    "gold lane": "gold_lane",
    "mid lane": "mid_lane",
    "midlane": "mid_lane",
    "roam": "roamer",
    "coach": "coach",
    "asst. coach": "assistant_coach",
    "analyst": "analyst",
}
STAFF_ROLES = {"coach", "assistant_coach", "analyst"}


def fetch_official_html(url: str, timeout: int = 30) -> str:
    """Download an official MPL page with a stable, explicit user agent."""
    request = Request(url, headers={"User-Agent": "mpl-season18-data-integration/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def build_season18_teams() -> pd.DataFrame:
    records = []
    for team_id, metadata in TEAM_METADATA.items():
        records.append(
            {
                "season": 18,
                "team_id": team_id,
                "team_name": metadata["team_name"],
                "organization_id": metadata["organization_id"],
                "franchise_slot_id": metadata["franchise_slot_id"],
                "source_url": TEAM_URL_TEMPLATE.format(slug=metadata["slug"]),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("team_id").reset_index(drop=True)


def _parse_indonesian_date(value: str) -> date:
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", value)
    if match is None or match.group(2) not in MONTHS_ID:
        raise ValueError(f"Cannot parse official schedule date: {value!r}")
    return date(int(match.group(3)), MONTHS_ID[match.group(2)], int(match.group(1)))


def parse_schedule_html(html: str, observed_at: date) -> pd.DataFrame:
    """Parse the official S18 regular-season schedule and currently published scores."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []
    for week in range(1, 10):
        container = soup.select_one(f"#t-week-{week}")
        if container is None:
            raise ValueError(f"Official schedule is missing week {week}.")
        scheduled_date: date | None = None
        for element in container.select(".match"):
            classes = set(element.get("class", []))
            if "date" in classes:
                scheduled_date = _parse_indonesian_date(element.get_text(" ", strip=True))
                continue
            if "position-relative" not in classes:
                continue
            if scheduled_date is None:
                raise ValueError(f"Week {week} contains a match without a date.")

            team_a_element = element.select_one(".team1 .name")
            team_b_element = element.select_one(".team2 .name")
            if team_a_element is None or team_b_element is None:
                raise ValueError(f"Week {week} contains a match without two teams.")
            team_a = team_a_element.get_text(" ", strip=True).upper()
            team_b = team_b_element.get_text(" ", strip=True).upper()
            if team_a not in TEAM_METADATA or team_b not in TEAM_METADATA:
                raise ValueError(f"Unknown S18 team code: {team_a!r} vs {team_b!r}")

            time_match = re.search(r"\b(\d{2}:\d{2})\b", element.get_text(" ", strip=True))
            if time_match is None:
                raise ValueError(f"Week {week} match {team_a}-{team_b} has no start time.")
            scheduled_at = pd.Timestamp(
                f"{scheduled_date.isoformat()} {time_match.group(1)}", tz="Asia/Jakarta"
            )

            detail_link = element.select_one("[onclick*='openMatchDetail']")
            detail_match = (
                re.search(r"openMatchDetail\((\d+)\)", detail_link.get("onclick", ""))
                if detail_link is not None
                else None
            )
            if detail_match is None:
                modal = element.select_one("[id^='modal2-']")
                detail_match = re.search(r"modal2-(\d+)", modal.get("id", "")) if modal else None
            if detail_match is None:
                raise ValueError(f"Cannot identify official match {team_a}-{team_b}.")
            official_match_id = int(detail_match.group(1))

            score_values = [item.get_text(" ", strip=True) for item in element.select(".score")]
            if len(score_values) != 2:
                raise ValueError(f"Official match {official_match_id} has invalid score fields.")
            completed = all(value.isdigit() for value in score_values)
            score_a = int(score_values[0]) if completed else None
            score_b = int(score_values[1]) if completed else None
            winner = team_a if completed and score_a > score_b else team_b if completed else None
            records.append(
                {
                    "season": 18,
                    "stage": "regular_season",
                    "week": week,
                    "match_id": f"S18-RS-{official_match_id}",
                    "official_match_id": official_match_id,
                    "scheduled_at": scheduled_at,
                    "team_a_id": team_a,
                    "team_b_id": team_b,
                    "team_a_franchise_slot_id": TEAM_METADATA[team_a]["franchise_slot_id"],
                    "team_b_franchise_slot_id": TEAM_METADATA[team_b]["franchise_slot_id"],
                    "team_a_score": score_a,
                    "team_b_score": score_b,
                    "winner_team_id": winner,
                    "winner_side": (
                        "team_a" if winner == team_a else "team_b" if winner == team_b else None
                    ),
                    "best_of": 3,
                    "status": "completed" if completed else "scheduled",
                    "observed_at": observed_at,
                    "source_url": SCHEDULE_URL,
                }
            )

    result = pd.DataFrame.from_records(records).sort_values(["scheduled_at", "official_match_id"])
    for column in ("season", "week", "official_match_id", "team_a_score", "team_b_score"):
        result[column] = result[column].astype("Int64")
    return result.reset_index(drop=True)


def parse_roster_html(
    html: str,
    team_id: str,
    observed_at: date,
    player_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Parse one official roster page and assign conservative effective dates."""
    player_aliases = player_aliases or {}
    metadata = TEAM_METADATA[team_id]
    soup = BeautifulSoup(html, "html.parser")
    heading = next(
        (
            item
            for item in soup.find_all(["h4", "h5", "h6"])
            if "ROSTER SEASON 18" in item.get_text(" ", strip=True).upper()
        ),
        None,
    )
    if heading is None:
        raise ValueError(f"Official {team_id} page has no 'Roster Season 18' section.")
    root = heading.find_parent("section") or soup
    records = []
    for card in root.select(".col-md-3.col-6"):
        nickname_element = card.select_one(".player-name")
        role_element = card.select_one(".player-role")
        if nickname_element is None or role_element is None:
            continue
        nickname = nickname_element.get_text(" ", strip=True)
        role_raw = role_element.get_text(" ", strip=True)
        role = ROLE_MAP.get(role_raw.casefold())
        if role is None:
            raise ValueError(f"Unknown S18 roster role for {team_id}: {role_raw!r}")
        raw_key = player_key(nickname)
        if raw_key is None:
            raise ValueError(f"Empty nickname on official {team_id} roster page.")
        canonical_key = player_aliases.get(raw_key, raw_key)
        records.append(
            {
                "season": 18,
                "team_id": team_id,
                "team_name": metadata["team_name"],
                "organization_id": metadata["organization_id"],
                "franchise_slot_id": metadata["franchise_slot_id"],
                "nickname": nickname,
                "player_key_raw": raw_key,
                "player_key": canonical_key,
                "player_id": canonical_player_id(canonical_key),
                "role_raw": role_raw,
                "role": role,
                "member_type": "staff" if role in STAFF_ROLES else "player",
                "observed_at": observed_at,
                "valid_from": observed_at,
                "valid_to": None,
                "temporal_basis": "first_verified_on_official_page",
                "source_url": TEAM_URL_TEMPLATE.format(slug=metadata["slug"]),
            }
        )
    if not records:
        raise ValueError(f"Official {team_id} roster section contains no members.")
    return pd.DataFrame.from_records(records)


def merge_roster_history(
    existing: pd.DataFrame, current: pd.DataFrame, observed_at: date
) -> pd.DataFrame:
    """Preserve first-seen dates and close members removed from the official roster page."""
    if existing.empty:
        return current
    key_columns = ["team_id", "player_id"]
    existing = existing.copy()
    for column in ("valid_from", "valid_to", "observed_at"):
        existing[column] = pd.to_datetime(existing[column], errors="coerce").dt.date
    existing_lookup = {
        tuple(getattr(row, column) for column in key_columns): row._asdict()
        for row in existing.itertuples(index=False)
    }
    current_keys = set()
    records = []
    for row in current.itertuples(index=False):
        record = row._asdict()
        key = tuple(record[column] for column in key_columns)
        current_keys.add(key)
        previous = existing_lookup.get(key)
        if previous is not None and pd.notna(previous["valid_from"]):
            record["valid_from"] = min(previous["valid_from"], record["valid_from"])
        records.append(record)
    for key, previous in existing_lookup.items():
        if key in current_keys:
            continue
        record = previous.copy()
        if pd.isna(record["valid_to"]):
            record["valid_to"] = observed_at
        records.append(record)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["team_id", "member_type", "player_id"])
        .reset_index(drop=True)
    )


def build_season18_asof_snapshot(
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    schedule: pd.DataFrame,
    cutoff_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct an end-of-day S18 snapshot without backdating unavailable roster data."""
    source_observed_at = str(schedule["observed_at"].iloc[0])
    source_observed_date = pd.to_datetime(source_observed_at, errors="raise").date()
    if cutoff_date > source_observed_date:
        raise ValueError(
            "Snapshot cutoff cannot be later than the official source observation date: "
            f"{cutoff_date.isoformat()} > {source_observed_date.isoformat()}."
        )
    retrospective = source_observed_date > cutoff_date
    snapshot_basis = (
        "retrospective_end_of_day_wib" if retrospective else "official_observation_date"
    )
    if retrospective:
        source_limitation = (
            "Snapshot direkonstruksi dari halaman resmi yang diamati pada "
            f"{source_observed_date.isoformat()}, setelah cutoff {cutoff_date.isoformat()}; "
            "bukan arsip halaman yang ditangkap tepat pada cutoff."
        )
    else:
        source_limitation = (
            "Halaman resmi diamati pada tanggal cutoff. Waktu pengamatan intrahari tidak "
            "direkam; snapshot merepresentasikan data yang tersedia saat pengambilan."
        )
    cutoff_end = pd.Timestamp(cutoff_date, tz="Asia/Jakarta") + pd.Timedelta(days=1)
    snapshot_schedule = schedule.copy()
    scheduled_at = pd.to_datetime(snapshot_schedule["scheduled_at"], utc=True).dt.tz_convert(
        "Asia/Jakarta"
    )
    available = snapshot_schedule["status"].eq("completed") & scheduled_at.lt(cutoff_end)
    hidden = ~available
    for column in ("team_a_score", "team_b_score", "winner_team_id", "winner_side"):
        snapshot_schedule.loc[hidden, column] = pd.NA
    snapshot_schedule.loc[hidden, "status"] = "scheduled"
    snapshot_schedule["observed_at"] = cutoff_date.isoformat()
    snapshot_schedule["snapshot_basis"] = snapshot_basis

    snapshot_rosters = rosters.copy()
    for column in ("valid_from", "valid_to"):
        snapshot_rosters[column] = pd.to_datetime(snapshot_rosters[column], errors="coerce")
    roster_cutoff = pd.Timestamp(cutoff_date)
    snapshot_rosters = snapshot_rosters.loc[
        snapshot_rosters["valid_from"].notna()
        & snapshot_rosters["valid_from"].le(roster_cutoff)
        & (snapshot_rosters["valid_to"].isna() | snapshot_rosters["valid_to"].ge(roster_cutoff))
    ].copy()

    completed_weeks = []
    partial_weeks = []
    for week in sorted(int(value) for value in snapshot_schedule["week"].unique()):
        week_rows = snapshot_schedule.loc[snapshot_schedule["week"].eq(week)]
        completed_count = int(week_rows["status"].eq("completed").sum())
        if completed_count == len(week_rows):
            completed_weeks.append(week)
        elif completed_count:
            partial_weeks.append(week)
    report = {
        "report_version": "1.0",
        "season": 18,
        "snapshot_id": f"S18_D{cutoff_date.strftime('%Y%m%d')}",
        "cutoff_date": cutoff_date.isoformat(),
        "cutoff_interpretation": "End of day Asia/Jakarta (WIB).",
        "source_data_observed_at": source_observed_at,
        "retrospective_reconstruction": retrospective,
        "scope": {
            "team_count": len(teams),
            "schedule_match_count": len(snapshot_schedule),
            "completed_match_count": int(available.sum()),
            "scheduled_match_count": int(hidden.sum()),
            "completed_weeks": completed_weeks,
            "partial_weeks": partial_weeks,
            "roster_member_count_available_at_cutoff": len(snapshot_rosters),
        },
        "temporal_guards": {
            "future_results_hidden": True,
            "roster_rule": "valid_from <= cutoff_date <= valid_to (when valid_to exists)",
            "outcome_availability_assumption": (
                "Skor final pada tanggal pertandingan dianggap tersedia pada akhir hari WIB."
            ),
            "source_limitation": source_limitation,
        },
        "completed_results": dataframe_records(
            snapshot_schedule.loc[
                available,
                [
                    "week",
                    "scheduled_at",
                    "match_id",
                    "team_a_id",
                    "team_b_id",
                    "team_a_score",
                    "team_b_score",
                    "winner_team_id",
                ],
            ]
        ),
    }
    return teams.copy(), snapshot_rosters, snapshot_schedule, report


def write_season18_asof_snapshot(
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    schedule: pd.DataFrame,
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    teams.to_csv(output_dir / "teams.csv", index=False)
    rosters.to_csv(output_dir / "rosters.csv", index=False)
    schedule.to_csv(output_dir / "schedule_results.csv", index=False)
    write_json(report, output_dir / "snapshot_metadata.json")


def validate_season18_data(
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    schedule: pd.DataFrame,
    observed_at: date,
) -> list[dict[str, Any]]:
    expected_teams = set(TEAM_METADATA)
    actual_teams = set(teams["team_id"])
    scheduled_teams = set(schedule["team_a_id"]) | set(schedule["team_b_id"])
    roster_teams = set(rosters["team_id"])
    pair_counts = schedule.assign(
        pair=schedule.apply(
            lambda row: "|".join(sorted((str(row.team_a_id), str(row.team_b_id)))), axis=1
        )
    )["pair"].value_counts()
    appearances = pd.concat([schedule["team_a_id"], schedule["team_b_id"]]).value_counts()
    completed = schedule.loc[schedule["status"].eq("completed")]
    valid_completed_scores = completed.apply(
        lambda row: (
            max(int(row.team_a_score), int(row.team_b_score)) == 2
            and int(row.team_a_score) != int(row.team_b_score)
        ),
        axis=1,
    )
    completed_after_observation = completed["scheduled_at"].dt.date.gt(observed_at).sum()
    checks = [
        ("nine_expected_teams", actual_teams == expected_teams, len(actual_teams)),
        ("schedule_team_coverage", scheduled_teams == expected_teams, len(scheduled_teams)),
        ("roster_team_coverage", roster_teams == expected_teams, len(roster_teams)),
        ("double_round_robin_match_count", len(schedule) == 72, len(schedule)),
        (
            "each_pair_plays_twice",
            len(pair_counts) == 36 and pair_counts.eq(2).all(),
            len(pair_counts),
        ),
        ("each_team_plays_sixteen", appearances.eq(16).all(), int(appearances.min())),
        (
            "unique_official_match_ids",
            not schedule["official_match_id"].duplicated().any(),
            int(schedule["official_match_id"].duplicated().sum()),
        ),
        ("completed_scores_valid", bool(valid_completed_scores.all()), len(completed)),
        ("no_future_results", completed_after_observation == 0, int(completed_after_observation)),
        (
            "unique_team_nickname_rows",
            not rosters.duplicated(["team_id", "player_key"]).any(),
            int(rosters.duplicated(["team_id", "player_key"]).sum()),
        ),
    ]
    return [
        {"check_id": check_id, "status": "pass" if passed else "fail", "count": count}
        for check_id, passed, count in checks
    ]


def build_season18_report(
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    schedule: pd.DataFrame,
    observed_at: date,
) -> dict[str, Any]:
    checks = validate_season18_data(teams, rosters, schedule, observed_at)
    completed = schedule["status"].eq("completed")
    active_roster = rosters.loc[rosters["valid_to"].isna()]
    return {
        "report_version": "1.0",
        "season": 18,
        "observed_at": observed_at.isoformat(),
        "sources": {
            "schedule_results": SCHEDULE_URL,
            "rosters": [
                TEAM_URL_TEMPLATE.format(slug=value["slug"]) for value in TEAM_METADATA.values()
            ],
        },
        "scope": {
            "team_count": len(teams),
            "roster_member_count": len(active_roster),
            "roster_history_row_count": len(rosters),
            "player_count": int(active_roster["member_type"].eq("player").sum()),
            "staff_count": int(active_roster["member_type"].eq("staff").sum()),
            "scheduled_match_count": len(schedule),
            "completed_match_count": int(completed.sum()),
            "remaining_match_count": int((~completed).sum()),
            "completed_through_week": int(schedule.loc[completed, "week"].max())
            if completed.any()
            else 0,
        },
        "temporal_policy": {
            "roster_valid_from": observed_at.isoformat(),
            "basis": "Tanggal verifikasi halaman resmi, bukan tanggal pengumuman roster.",
            "historical_use_guard": (
                "Roster ini hanya boleh dipakai pada snapshot dengan cutoff >= valid_from."
            ),
        },
        "blocking_issue_count": sum(item["status"] == "fail" for item in checks),
        "checks": checks,
        "completed_results": dataframe_records(
            schedule.loc[
                completed,
                [
                    "week",
                    "scheduled_at",
                    "match_id",
                    "team_a_id",
                    "team_b_id",
                    "team_a_score",
                    "team_b_score",
                    "winner_team_id",
                ],
            ]
        ),
    }


def write_season18_outputs(
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    schedule: pd.DataFrame,
    report: dict[str, Any],
    output_dir: Path,
    report_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    teams.to_csv(output_dir / "teams.csv", index=False)
    rosters.to_csv(output_dir / "rosters.csv", index=False)
    schedule.to_csv(output_dir / "schedule_results.csv", index=False)
    write_json(report, report_path)
