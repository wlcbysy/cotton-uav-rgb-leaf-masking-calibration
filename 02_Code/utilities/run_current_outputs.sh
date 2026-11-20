#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PATH="/opt/homebrew/bin:$PATH"
if [[ -x "/opt/homebrew/bin/python3" ]]; then
  PY="/opt/homebrew/bin/python3"
else
  PY="python3"
fi
export PYTHONDONTWRITEBYTECODE=1
case "${1:-qa}" in
  build)
    "$PY" 02_Code/src/upgrade_q1/plot_q1_upgraded_figures.py
    "$PY" 02_Code/src/upgrade_q1/build_and_verify_q1_package.py
    "$PY" -m unittest 02_Code/tests/test_project_layout.py
    ;;
  qa)
    "$PY" 02_Code/src/upgrade_q1/build_and_verify_q1_package.py
    "$PY" -m unittest 02_Code/tests/test_project_layout.py
    ;;
  *)
    echo "Usage: $0 [qa|build]" >&2
    exit 2
    ;;
esac
