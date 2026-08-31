#!/usr/bin/env bash
set -euo pipefail

pipeline_mode="${1:-update}"
observation_date="${2:-$(date +%F)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cli="${project_root}/.venv/bin/mpl-predictor"
python_bin="${project_root}/.venv/bin/python"
ruff_bin="${project_root}/.venv/bin/ruff"

if [[ ! -x "${cli}" ]]; then
    echo "Environment belum siap. Jalankan: make setup" >&2
    exit 1
fi

cd "${project_root}"

train_historical() {
    "${cli}" audit
    "${cli}" semantic-audit
    "${cli}" normalize
    "${cli}" canonicalize
    "${cli}" quality-report
    "${cli}" eda
    "${cli}" prediction-policy
    "${cli}" build-features
    "${cli}" baseline
    "${cli}" build-match-features
    "${cli}" backtest
    "${cli}" train-final
}

update_season18() {
    if [[ ! -f "artifacts/final_match_model.joblib" ]]; then
        echo "Model final belum tersedia. Jalankan mode 'train' atau 'all'." >&2
        exit 1
    fi
    "${cli}" sync-season18 --observed-at "${observation_date}"
    "${cli}" update-season18-predictions
    "${cli}" explain-season18
}

verify_project() {
    "${ruff_bin}" check .
    "${python_bin}" -m pytest
    test -s README.md
    test -s docs/OPERATIONS.md
}

case "${pipeline_mode}" in
    train)
        train_historical
        ;;
    update)
        update_season18
        ;;
    verify)
        verify_project
        ;;
    all)
        train_historical
        update_season18
        verify_project
        ;;
    *)
        echo "Mode tidak dikenal: ${pipeline_mode}. Gunakan train, update, verify, atau all." >&2
        exit 2
        ;;
esac
