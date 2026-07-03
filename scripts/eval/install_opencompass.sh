#!/usr/bin/env bash
# Usage:
#   bash scripts/eval/install_opencompass.sh

set -euo pipefail

python -m ensurepip --upgrade

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

OPENCOMPASS_PATH=${OPENCOMPASS_PATH:scripts/eval/opencompass}

# ── Detect python ──
if [ -n "${PYTHON_BIN:-}" ]; then
    true
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Error: python/python3 not found. Please activate your environment first." >&2
    exit 1
fi

echo "Python:          $($PYTHON_BIN --version) ($PYTHON_BIN)"
echo "OpenCompass path: $OPENCOMPASS_PATH"
echo "Repo root:       $REPO_ROOT"
echo

# ── Check opencompass repo exists ──
if [ ! -d "$OPENCOMPASS_PATH/opencompass" ]; then
    echo "Error: OpenCompass repo not found at $OPENCOMPASS_PATH" >&2
    exit 1
fi

# ── Step 1: Install runtime dependencies ──
echo "[1/3] Installing runtime dependencies from $OPENCOMPASS_PATH/requirements/runtime.txt ..."
$PYTHON_BIN -m pip install -r "$OPENCOMPASS_PATH/requirements/runtime.txt"

# ── Step 2: Install opencompass itself (editable) ──
echo "[2/3] Installing OpenCompass in editable mode..."
$PYTHON_BIN -m pip install -e "$OPENCOMPASS_PATH" --no-deps

# ── Step 3: Verify ──
echo "[3/3] Verifying installation..."
export PYTHONPATH="$REPO_ROOT${OPENCOMPASS_PATH:+:$OPENCOMPASS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

if $PYTHON_BIN -c "from opencompass.cli.main import main; print('  opencompass import OK')" 2>&1; then
    echo "OpenCompass installed successfully!"
else
    echo "Warning: opencompass import failed." >&2
    echo "Make sure to set:" >&2
    echo "  export OPENCOMPASS_PATH=$OPENCOMPASS_PATH" >&2
    echo "  export PYTHONPATH=$REPO_ROOT:\$OPENCOMPASS_PATH:\$PYTHONPATH" >&2
fi

echo
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Add to your shell before running evals:"
echo "    export OPENCOMPASS_PATH=$OPENCOMPASS_PATH"
echo "    export PYTHONPATH=$REPO_ROOT:\$OPENCOMPASS_PATH:\$PYTHONPATH"
echo "============================================"
