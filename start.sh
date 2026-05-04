#!/usr/bin/env bash
set -euo pipefail

# Always run from this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: activate local virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Runtime configuration (can be overridden by environment variables)
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" binance_signal_scanner.py \
  --interval-seconds "$INTERVAL_SECONDS" \
  --email \
  --log-level "$LOG_LEVEL"

