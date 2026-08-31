import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Cloud may execute this file directly instead of installing the src-layout
# package. Add the src directory before importing mpl_predictor so both entry points work.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mpl_predictor.config import ProjectPaths, get_project_paths  # noqa: E402


def dashboard_file_paths(paths: ProjectPaths) -> dict[str, Path]:
    prediction_dir = paths.processed / "predictions"
    return {
        "predictions": prediction_dir / "season18_snapshot_predictions.parquet",
        "matches": prediction_dir / "season18_snapshot_match_probabilities.parquet",
        "global_importance": prediction_dir / "season18_global_feature_importance.parquet",
        "match_explanations": prediction_dir / "season18_match_explanations.parquet",
        "team_explanations": prediction_dir / "season18_team_explanations.parquet",
        "rosters": paths.data / "season18" / "rosters.csv",
    }


def load_dashboard_data(paths: ProjectPaths) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    files = dashboard_file_paths(paths)
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        return {}, missing
    frames = {
        "predictions": pd.read_parquet(files["predictions"]),
        "matches": pd.read_parquet(files["matches"]),
        "global_importance": pd.read_parquet(files["global_importance"]),
        "match_explanations": pd.read_parquet(files["match_explanations"]),
        "team_explanations": pd.read_parquet(files["team_explanations"]),
        "rosters": pd.read_csv(files["rosters"]),
    }
    return frames, []


def _snapshot_label(row: pd.Series) -> str:
    if row["prediction_type"] == "preseason":
        return f"Pramusim · cutoff {row['feature_cutoff_date']}"
    return f"Week {int(row['completed_week'])} · cutoff {row['feature_cutoff_date']}"


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _prediction_chart(frame: pd.DataFrame):
    chart_data = frame.sort_values("champion_probability", ascending=True)
    figure = px.bar(
        chart_data,
        x="champion_probability",
        y="team_name",
        orientation="h",
        color="champion_probability",
        color_continuous_scale="Blues",
        text=chart_data["champion_probability"].map(_percent),
        labels={"champion_probability": "Peluang juara", "team_name": "Tim"},
    )
    figure.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_tickformat=".0%")
    figure.update_traces(textposition="outside")
    return figure


def _history_chart(predictions: pd.DataFrame):
    labels = (
        predictions[["snapshot_id", "snapshot_order", "prediction_type", "completed_week"]]
        .drop_duplicates()
        .sort_values("snapshot_order")
    )
    label_lookup = {
        row.snapshot_id: "Pramusim"
        if row.prediction_type == "preseason"
        else f"Week {int(row.completed_week)}"
        for row in labels.itertuples(index=False)
    }
    chart_data = predictions.copy()
    chart_data["snapshot_label"] = chart_data["snapshot_id"].map(label_lookup)
    figure = px.line(
        chart_data.sort_values("snapshot_order"),
        x="snapshot_label",
        y="champion_probability",
        color="team_name",
        markers=True,
        labels={
            "snapshot_label": "Snapshot",
            "champion_probability": "Peluang juara",
            "team_name": "Tim",
        },
    )
    figure.update_layout(yaxis_tickformat=".0%")
    return figure


def _render_overview(data: dict[str, pd.DataFrame], snapshot_id: str) -> None:
    predictions = data["predictions"]
    matches = data["matches"]
    current = predictions.loc[predictions["snapshot_id"].eq(snapshot_id)].sort_values(
        "champion_probability", ascending=False
    )
    current_matches = matches.loc[matches["snapshot_id"].eq(snapshot_id)]
    leader = current.iloc[0]

    first, second, third, fourth = st.columns(4)
    first.metric(
        "Favorit juara", str(leader["team_name"]), _percent(leader["champion_probability"])
    )
    second.metric("Match selesai", int(current_matches["status"].eq("completed").sum()), "/ 72")
    third.metric("Match tersisa", int(current_matches["status"].eq("scheduled").sum()))
    fourth.metric("Peluang playoff favorit", _percent(leader["playoff_probability"]))

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Probabilitas juara")
        st.plotly_chart(_prediction_chart(current), width="stretch")
    with right:
        st.subheader("Ranking simulasi")
        table = current[
            [
                "champion_rank",
                "team_name",
                "champion_probability",
                "playoff_probability",
                "expected_regular_rank",
            ]
        ].rename(
            columns={
                "champion_rank": "Rank",
                "team_name": "Tim",
                "champion_probability": "Juara",
                "playoff_probability": "Playoff",
                "expected_regular_rank": "Ekspektasi rank RS",
            }
        )
        st.dataframe(
            table.style.format(
                {"Juara": "{:.2%}", "Playoff": "{:.2%}", "Ekspektasi rank RS": "{:.2f}"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Perubahan probabilitas dari pramusim")
    st.plotly_chart(_history_chart(predictions), width="stretch")


def _render_matches(data: dict[str, pd.DataFrame], snapshot_id: str) -> None:
    matches = data["matches"]
    current = matches.loc[
        matches["snapshot_id"].eq(snapshot_id) & matches["status"].eq("scheduled")
    ].sort_values(["scheduled_at", "official_match_id"])
    st.subheader("Probabilitas pertandingan tersisa")
    if current.empty:
        st.success("Tidak ada pertandingan regular season tersisa pada snapshot ini.")
        return
    table = current[
        [
            "week",
            "scheduled_at",
            "team_a_id",
            "team_b_id",
            "team_a_win_probability",
            "predicted_winner_team_id",
        ]
    ].rename(
        columns={
            "week": "Week",
            "scheduled_at": "Jadwal",
            "team_a_id": "Team A",
            "team_b_id": "Team B",
            "team_a_win_probability": "P(Team A menang)",
            "predicted_winner_team_id": "Favorit",
        }
    )
    st.dataframe(
        table.style.format({"P(Team A menang)": "{:.2%}"}),
        hide_index=True,
        width="stretch",
    )


def _render_explainability(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Explainability model")
    st.caption(
        "Kontribusi pertandingan menjelaskan raw logistic logit sebelum side-symmetry, "
        "kalibrasi Platt, dan simulasi Monte Carlo. Nilai ini bukan hubungan sebab-akibat."
    )
    global_importance = (
        data["global_importance"].head(12).sort_values("absolute_importance", ascending=True)
    )
    figure = px.bar(
        global_importance,
        x="absolute_importance",
        y="feature_label",
        orientation="h",
        color="coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"absolute_importance": "|Koefisien|", "feature_label": "Fitur"},
    )
    figure.update_layout(yaxis_title=None)
    st.plotly_chart(figure, width="stretch")

    explanations = data["match_explanations"]
    match_labels = (
        explanations[["match_id", "week", "team_a_id", "team_b_id"]]
        .drop_duplicates()
        .sort_values(["week", "match_id"])
    )
    label_lookup = {
        row.match_id: f"W{int(row.week)} · {row.team_a_id} vs {row.team_b_id}"
        for row in match_labels.itertuples(index=False)
    }
    selected_match = st.selectbox(
        "Pilih pertandingan", options=list(label_lookup), format_func=label_lookup.get
    )
    local = explanations.loc[explanations["match_id"].eq(selected_match)].nsmallest(
        10, "contribution_rank"
    )
    local = local.sort_values("contribution")
    local_figure = px.bar(
        local,
        x="contribution",
        y="feature_label",
        orientation="h",
        color="favors_team_id",
        labels={"contribution": "Kontribusi raw logit", "feature_label": "Fitur"},
    )
    local_figure.update_layout(yaxis_title=None)
    st.plotly_chart(local_figure, width="stretch")

    st.subheader("Perubahan peluang tim")
    team_table = data["team_explanations"][
        [
            "team_name",
            "preseason_champion_probability",
            "current_champion_probability",
            "champion_probability_change",
            "mean_remaining_match_win_probability",
        ]
    ].rename(
        columns={
            "team_name": "Tim",
            "preseason_champion_probability": "Pramusim",
            "current_champion_probability": "Terbaru",
            "champion_probability_change": "Perubahan",
            "mean_remaining_match_win_probability": "Rata-rata peluang match tersisa",
        }
    )
    st.dataframe(
        team_table.style.format(
            {
                "Pramusim": "{:.2%}",
                "Terbaru": "{:.2%}",
                "Perubahan": "{:+.2%}",
                "Rata-rata peluang match tersisa": "{:.2%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="MPL S18 Predictor", page_icon="🏆", layout="wide")
    st.title("MPL Indonesia Season 18 Champion Predictor")
    st.caption("Prediksi leakage-safe · calibrated match model · Monte Carlo 20.000 iterasi")
    paths = get_project_paths()
    data, missing = load_dashboard_data(paths)
    if missing:
        st.error("Output prediksi belum lengkap.")
        st.code("make update-season18")
        with st.expander("File yang belum tersedia"):
            for path in missing:
                st.write(path)
        return

    snapshot_rows = (
        data["predictions"][
            [
                "snapshot_id",
                "snapshot_order",
                "prediction_type",
                "completed_week",
                "feature_cutoff_date",
            ]
        ]
        .drop_duplicates()
        .sort_values("snapshot_order")
    )
    labels = {str(row["snapshot_id"]): _snapshot_label(row) for _, row in snapshot_rows.iterrows()}
    selected_snapshot = st.sidebar.selectbox(
        "Snapshot prediksi",
        options=list(labels),
        index=len(labels) - 1,
        format_func=labels.get,
    )
    st.sidebar.info(
        "Roster S18 pertama terverifikasi 31 Agustus 2026 dan tidak digunakan pada "
        "rekonstruksi pramusim atau Week 1-3."
    )
    overview_tab, matches_tab, explanation_tab, data_tab = st.tabs(
        ["Ringkasan", "Pertandingan", "Explainability", "Data & asumsi"]
    )
    with overview_tab:
        _render_overview(data, selected_snapshot)
    with matches_tab:
        _render_matches(data, selected_snapshot)
    with explanation_tab:
        _render_explainability(data)
    with data_tab:
        st.subheader("Cakupan data")
        first, second, third = st.columns(3)
        first.metric("Snapshot", data["predictions"]["snapshot_id"].nunique())
        second.metric("Tim", data["predictions"]["team_id"].nunique())
        third.metric("Anggota roster tercatat", len(data["rosters"]))
        st.markdown(
            "- Pramusim memakai hasil historis sampai S17 dan **0 hasil S18**.\n"
            "- Snapshot mingguan hanya membuka hasil sampai week terkait.\n"
            "- Bracket playoff mengikuti struktur S15-S17.\n"
            "- Probabilitas match dibekukan pada setiap snapshot simulasi."
        )


if __name__ == "__main__":
    main()
