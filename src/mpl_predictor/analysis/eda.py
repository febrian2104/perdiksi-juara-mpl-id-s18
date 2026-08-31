from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.analysis.quality import _season_coverage


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0))).mul(100).round(2)


def build_team_season_performance(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build descriptive regular-season performance for every team-season."""
    matches = tables["matches"].loc[tables["matches"]["stage"].eq("regular_season")].copy()
    records: list[dict[str, Any]] = []

    for row in matches.itertuples():
        for side, opponent_side in (("team_a", "team_b"), ("team_b", "team_a")):
            team_id = getattr(row, f"{side}_id")
            records.append(
                {
                    "season": int(row.season),
                    "team_id": team_id,
                    "organization_id": getattr(row, f"{side}_organization_id"),
                    "franchise_slot_id": getattr(row, f"{side}_franchise_slot_id"),
                    "match_played": 1,
                    "match_won": int(row.winner_side == side),
                    "match_lost": int(row.winner_side == opponent_side),
                    "match_drawn": int(row.winner_side == "draw"),
                    "games_won": int(getattr(row, f"{side}_score")),
                    "games_lost": int(getattr(row, f"{opponent_side}_score")),
                }
            )

    observations = pd.DataFrame.from_records(records)
    metrics = (
        observations.groupby(
            ["season", "team_id", "organization_id", "franchise_slot_id"],
            dropna=False,
            as_index=False,
        )[
            [
                "match_played",
                "match_won",
                "match_lost",
                "match_drawn",
                "games_won",
                "games_lost",
            ]
        ]
        .sum()
        .rename(columns={"match_played": "matches_played"})
    )
    decided = metrics["match_won"] + metrics["match_lost"]
    total_games = metrics["games_won"] + metrics["games_lost"]
    metrics["match_win_rate_pct"] = _safe_pct(metrics["match_won"], decided)
    metrics["game_win_rate_pct"] = _safe_pct(metrics["games_won"], total_games)
    metrics["game_differential"] = metrics["games_won"] - metrics["games_lost"]

    targets = tables["championships"][
        ["season", "team_id", "team", "champion", "runner_up", "final_rank_min", "final_rank_max"]
    ].rename(columns={"team": "team_name"})
    metrics = metrics.merge(targets, on=["season", "team_id"], how="left", validate="one_to_one")
    metrics["champion"] = metrics["champion"].fillna(False).astype(bool)
    metrics["runner_up"] = metrics["runner_up"].fillna(False).astype(bool)
    metrics["regular_performance_rank_proxy"] = (
        metrics.sort_values(
            ["season", "match_win_rate_pct", "game_differential", "games_won"],
            ascending=[True, False, False, False],
        )
        .groupby("season")
        .cumcount()
        .add(1)
        .astype(int)
    )
    return metrics.sort_values(["season", "regular_performance_rank_proxy"]).reset_index(drop=True)


def build_roster_continuity(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Describe full-season roster overlap for consecutive franchise seasons."""
    players = tables["players"].loc[
        tables["players"]["season"].ge(4)
        & tables["players"]["franchise_slot_id"].notna()
        & tables["players"]["player_id"].notna()
    ]
    rosters = {
        (int(season), str(slot)): set(group["player_id"].astype(str))
        for (season, slot), group in players.groupby(["season", "franchise_slot_id"])
    }
    records = []
    for (season, slot), current in sorted(rosters.items()):
        previous = rosters.get((season - 1, slot))
        if previous is None:
            continue
        shared = current & previous
        union = current | previous
        records.append(
            {
                "season": season,
                "franchise_slot_id": slot,
                "previous_roster_size": len(previous),
                "current_roster_size": len(current),
                "retained_player_count": len(shared),
                "new_player_count": len(current - previous),
                "departed_player_count": len(previous - current),
                "roster_jaccard_pct": round(100 * len(shared) / len(union), 2) if union else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _season_summary(tables: dict[str, pd.DataFrame], performance: pd.DataFrame) -> pd.DataFrame:
    coverage = _season_coverage(tables)
    matches = tables["matches"]
    championships = tables["championships"].loc[tables["championships"]["champion"]]
    stage_counts = matches.groupby(["season", "stage"]).size().unstack(fill_value=0).reset_index()
    stage_counts = stage_counts.rename(
        columns={"regular_season": "regular_season_match_count", "playoffs": "playoff_match_count"}
    )
    champion_metrics = performance.loc[
        performance["champion"],
        [
            "season",
            "match_win_rate_pct",
            "game_win_rate_pct",
            "game_differential",
            "regular_performance_rank_proxy",
        ],
    ].rename(
        columns={
            "match_win_rate_pct": "champion_regular_match_win_rate_pct",
            "game_win_rate_pct": "champion_regular_game_win_rate_pct",
            "game_differential": "champion_regular_game_differential",
            "regular_performance_rank_proxy": "champion_regular_performance_rank_proxy",
        }
    )
    champion_names = championships[["season", "team", "organization_id"]].rename(
        columns={"team": "champion_team", "organization_id": "champion_organization_id"}
    )
    result = coverage.merge(stage_counts, on="season", how="left")
    result = result.merge(champion_names, on="season", how="left", validate="one_to_one")
    return result.merge(champion_metrics, on="season", how="left", validate="one_to_one")


def build_eda_report(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Produce modeling-oriented descriptive statistics without creating model features."""
    performance = build_team_season_performance(tables)
    continuity = build_roster_continuity(tables)
    season_summary = _season_summary(tables, performance)
    franchise = performance.loc[performance["season"].ge(4)].copy()
    champions = franchise.loc[franchise["champion"]]
    non_champions = franchise.loc[~franchise["champion"]]

    regular_matches = tables["matches"].loc[
        tables["matches"]["stage"].eq("regular_season")
        & tables["matches"]["winner_side"].isin(["team_a", "team_b"])
    ]
    team_a_wins = int(regular_matches["winner_side"].eq("team_a").sum())

    champion_top_proxy_count = int(champions["regular_performance_rank_proxy"].eq(1).sum())
    champion_below_half_count = int(champions["match_win_rate_pct"].lt(50).sum())
    mean_champion_win_rate = round(float(champions["match_win_rate_pct"].mean()), 2)
    mean_non_champion_win_rate = round(float(non_champions["match_win_rate_pct"].mean()), 2)
    median_continuity = (
        round(float(continuity["roster_jaccard_pct"].median()), 2) if not continuity.empty else None
    )

    champion_profile_columns = [
        "season",
        "team_id",
        "team_name",
        "organization_id",
        "franchise_slot_id",
        "matches_played",
        "match_won",
        "match_lost",
        "match_drawn",
        "match_win_rate_pct",
        "game_win_rate_pct",
        "game_differential",
        "regular_performance_rank_proxy",
    ]
    report = {
        "report_version": "1.0",
        "scope": {
            "descriptive_seasons": [1, int(performance["season"].max())],
            "primary_franchise_seasons": [4, int(performance["season"].max())],
            "team_season_observations": len(performance),
            "franchise_team_season_observations": len(franchise),
            "franchise_champion_observations": len(champions),
            "positive_class_pct": round(100 * len(champions) / len(franchise), 2),
        },
        "franchise_comparison": {
            "champion_mean_regular_match_win_rate_pct": mean_champion_win_rate,
            "non_champion_mean_regular_match_win_rate_pct": mean_non_champion_win_rate,
            "difference_percentage_points": round(
                mean_champion_win_rate - mean_non_champion_win_rate, 2
            ),
            "champion_mean_regular_game_win_rate_pct": round(
                float(champions["game_win_rate_pct"].mean()), 2
            ),
            "non_champion_mean_regular_game_win_rate_pct": round(
                float(non_champions["game_win_rate_pct"].mean()), 2
            ),
            "champion_top_regular_performance_proxy_count": champion_top_proxy_count,
            "champion_below_50_pct_match_win_rate_count": champion_below_half_count,
        },
        "side_balance": {
            "decided_regular_season_matches": len(regular_matches),
            "team_a_win_count": team_a_wins,
            "team_a_win_pct": round(100 * team_a_wins / len(regular_matches), 2),
            "interpretation": (
                "Posisi team_a/team_b harus diperlakukan simetris oleh feature pipeline."
            ),
        },
        "roster_continuity": {
            "consecutive_slot_season_pairs": len(continuity),
            "median_jaccard_pct": median_continuity,
            "warning": (
                "Ini adalah overlap daftar roster satu musim penuh, bukan roster yang sudah "
                "dipastikan "
                "tersedia pada cutoff historis."
            ),
        },
        "modeling_implications": [
            "Performa regular season berhubungan dengan juara, tetapi bukan penentu tunggal.",
            (
                "Target sangat tidak seimbang pada level team-season; evaluasi harus memakai "
                "probabilitas dan ranking."
            ),
            "Fitur team_a dan team_b harus dibuat secara simetris untuk menghindari bias posisi.",
            "Season 4-17 menjadi scope utama; Season 1-3 hanya analisis sensitivitas.",
            (
                "Player stats, draft, dan duration tidak masuk baseline karena coverage tidak "
                "konsisten."
            ),
            (
                "Roster continuity baru aman sebagai fitur as-of setelah tersedia tanggal "
                "efektif roster."
            ),
        ],
        "season_summary": dataframe_records(season_summary),
        "champion_profiles": dataframe_records(
            performance.loc[performance["champion"], champion_profile_columns]
        ),
    }
    frames = {
        "team_season_performance": performance,
        "roster_continuity": continuity,
        "season_summary": season_summary,
    }
    return report, frames


def write_eda_report(report: dict[str, Any], path: Path) -> None:
    write_json(report, path)


def write_eda_figures(frames: dict[str, pd.DataFrame], directory: Path) -> list[Path]:
    """Write four compact, reproducible EDA figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs: list[Path] = []
    season_summary = frames["season_summary"]
    performance = frames["team_season_performance"]

    figure, axis = plt.subplots(figsize=(11, 5.5))
    stages = season_summary.set_index("season")[
        [
            "regular_season_match_count",
            "playoff_match_count",
        ]
    ]
    stages.plot(kind="bar", stacked=True, ax=axis, color=["#2563EB", "#F59E0B"])
    axis.set(title="Jumlah pertandingan per musim", xlabel="Season", ylabel="Match")
    axis.legend(["Regular season", "Playoff"], frameon=False)
    figure.tight_layout()
    path = directory / "matches_by_season.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    champions = performance.loc[performance["season"].ge(4) & performance["champion"]]
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=champions,
        x="season",
        y="match_win_rate_pct",
        color="#16A34A",
        ax=axis,
    )
    axis.axhline(50, color="#DC2626", linestyle="--", linewidth=1)
    axis.set(
        title="Win rate regular season tim juara (era franchise)",
        xlabel="Season",
        ylabel="Match win rate (%)",
        ylim=(0, 100),
    )
    figure.tight_layout()
    path = directory / "champion_regular_season_win_rate.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    franchise = performance.loc[performance["season"].ge(4)].assign(
        result=lambda frame: frame["champion"].map({True: "Juara", False: "Bukan juara"})
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=franchise,
        x="result",
        y="match_win_rate_pct",
        order=["Bukan juara", "Juara"],
        color="#60A5FA",
        ax=axis,
    )
    axis.set(
        title="Distribusi win rate regular season Season 4+",
        xlabel="Hasil akhir",
        ylabel="Match win rate (%)",
    )
    figure.tight_layout()
    path = directory / "regular_season_win_rate_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    coverage_columns = [
        "match_outcome_coverage_pct",
        "game_outcome_coverage_pct",
        "game_duration_coverage_pct",
        "roster_role_coverage_pct",
        "draft_game_coverage_pct",
    ]
    labels = ["Match result", "Game result", "Duration", "Roster role", "Draft"]
    coverage = season_summary.set_index("season")[coverage_columns].T
    coverage.index = labels
    figure, axis = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(
        coverage,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.25,
        cbar_kws={"label": "Coverage (%)"},
        ax=axis,
    )
    axis.set(title="Coverage data per musim", xlabel="Season", ylabel="Sumber fitur")
    figure.tight_layout()
    path = directory / "data_coverage_by_season.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)
    return outputs
