#!/usr/bin/env bash
# Installs dependencies and launches the AntMap application on macOS.
set -e

VENV_DIR=".venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install it via Homebrew: brew install python" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip >/dev/null

if ! python -c "import PyQt5" >/dev/null 2>&1; then
  pip install PyQt5
fi

python main.py "$@"
