#!/usr/bin/env bash
# Usage: bash scripts/eval/install_opencompass.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ensure_bash
setup_eval_env

OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-scripts/eval/opencompass}

"$PYTHON_BIN" -m ensurepip --upgrade 2>/dev/null || true

echo "Python:           $($PYTHON_BIN --version) ($PYTHON_BIN)"
echo "OpenCompass path: $OPENCOMPASS_PATH"
echo "Repo root:        $REPO_ROOT"
echo

if [ ! -d "$OPENCOMPASS_PATH/opencompass" ]; then
    echo "Error: OpenCompass repo not found at $OPENCOMPASS_PATH" >&2
    exit 1
fi

echo "[1/3] Installing runtime dependencies..."
"$PYTHON_BIN" -m pip install -r "$OPENCOMPASS_PATH/requirements/runtime.txt"

echo "[2/3] Installing OpenCompass (editable)..."
"$PYTHON_BIN" -m pip install -e "$OPENCOMPASS_PATH" --no-deps

echo "[3/3] Verifying installation..."
export PYTHONPATH="$REPO_ROOT${OPENCOMPASS_PATH:+:$OPENCOMPASS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

if "$PYTHON_BIN" -c "from opencompass.cli.main import main; print('  opencompass import OK')"; then
    echo "OpenCompass installed successfully!"
else
    echo "Warning: opencompass import failed. Set before running evals:" >&2
    echo "  export OPENCOMPASS_PATH=$OPENCOMPASS_PATH" >&2
    echo "  export PYTHONPATH=$REPO_ROOT:\$OPENCOMPASS_PATH:\$PYTHONPATH" >&2
fi

echo
echo "Add to your shell before running evals:"
echo "  export OPENCOMPASS_PATH=$OPENCOMPASS_PATH"
echo "  export PYTHONPATH=$REPO_ROOT:\$OPENCOMPASS_PATH:\$PYTHONPATH"
