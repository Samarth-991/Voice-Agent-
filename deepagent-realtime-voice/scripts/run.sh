#!/usr/bin/env bash
# Start the server. Forward port 7860 in VS Code (Ports tab) and open
# http://localhost:7860 — mic access requires localhost or HTTPS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
source "$ROOT/scripts/env.sh"
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a

cd "$ROOT"
exec "$DRA_VENV/bin/python" -m uvicorn app.server:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-7860}" "$@"
