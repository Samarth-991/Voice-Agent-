#!/usr/bin/env bash
# (Re)build the Python venv and fetch the web bundle. Safe to rerun; needed
# after every compute restart because /mnt is wiped.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
source "$ROOT/scripts/env.sh"

mkdir -p "$DRA_CACHE" "$TMPDIR" "$(dirname "$DRA_VENV")"

if [ ! -x "$DRA_VENV/bin/python" ]; then
  echo "Creating venv at $DRA_VENV (base: $("$DRA_BASE_PYTHON" --version))"
  "$DRA_BASE_PYTHON" -m venv "$DRA_VENV"
fi

"$DRA_VENV/bin/python" -m pip install --upgrade pip wheel
"$DRA_VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
for extra in "$ROOT"/requirements-*.txt; do
  [ -f "$extra" ] || continue
  echo "Installing $(basename "$extra")"
  "$DRA_VENV/bin/python" -m pip install -r "$extra"
done

if [ ! -f "$ROOT/web/vendor/openai-realtime-agents.umd.js" ]; then
  bash "$ROOT/scripts/setup_web.sh"
fi

echo "Done. Start with: bash scripts/run.sh"
