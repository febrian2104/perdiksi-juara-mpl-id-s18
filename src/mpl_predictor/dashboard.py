import pandas as pd
import streamlit as st

from mpl_predictor.config import get_project_paths
from mpl_predictor.data.audit import audit_data


def main() -> None:
    st.set_page_config(page_title="MPL S18 Predictor", page_icon="🏆", layout="wide")
    st.title("MPL Indonesia Season 18 Champion Predictor")
    st.caption("Project foundation: historical-data inventory and quality status")

    paths = get_project_paths()
    report = audit_data(paths.data)

    first, second, third, fourth = st.columns(4)
    first.metric("Historical seasons", len({item.season for item in report.files}))
    second.metric("Dataset files", len(report.files))
    third.metric("Data rows", f"{report.total_rows:,}")
    fourth.metric("Audit errors", len(report.errors))

    st.subheader("Dataset inventory")
    inventory = pd.DataFrame(report.summary_rows())
    if inventory.empty:
        st.error("No historical dataset files were found.")
    else:
        pivot = inventory.pivot(index="season", columns="table", values="rows").fillna(0)
        st.dataframe(pivot, use_container_width=True)

    st.subheader("Quality status")
    if not report.issues:
        st.success("The current foundation audit found no structural errors or warnings.")
    else:
        for issue in report.issues:
            message = f"{issue.code}: {issue.message}"
            if issue.severity == "error":
                st.error(message)
            else:
                st.warning(message)

    st.info(
        "Prediction results will be added after canonicalization, feature engineering, "
        "temporal backtesting, and Season 18 roster integration."
    )


if __name__ == "__main__":
    main()
