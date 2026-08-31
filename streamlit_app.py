import sys
from collections.abc import Callable
from pathlib import Path


def load_dashboard_main() -> Callable[[], None]:
    """Load the src-layout dashboard when Streamlit executes this root entry point."""
    source_root = Path(__file__).resolve().parent / "src"
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from mpl_predictor.dashboard import main

    return main


if __name__ == "__main__":
    load_dashboard_main()()
