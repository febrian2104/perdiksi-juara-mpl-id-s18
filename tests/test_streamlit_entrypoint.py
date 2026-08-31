from streamlit_app import load_dashboard_main


def test_streamlit_cloud_entrypoint_loads_src_package() -> None:
    dashboard_main = load_dashboard_main()

    assert dashboard_main.__module__ == "mpl_predictor.dashboard"
    assert dashboard_main.__name__ == "main"
