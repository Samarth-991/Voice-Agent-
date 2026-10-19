#!/usr/bin/env bash
# Install the exact npm-pinned OpenAI Agents SDK and copy its browser bundle
# into web/vendor/. node_modules lives on the cache disk (symlinks don't work
# on the CIFS-mounted cloudfiles share); tests also read the SDK's event
# schemas from there.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
source "$ROOT/scripts/env.sh"
WEB_NODE="$DRA_CACHE/web-node"

mkdir -p "$WEB_NODE"
cp "$ROOT/web/package.json" "$ROOT/web/package-lock.json" "$ROOT/web/.npmrc" "$WEB_NODE/"
(cd "$WEB_NODE" && npm ci --omit=dev --no-audit --no-fund)

mkdir -p "$ROOT/web/vendor"
cp "$WEB_NODE/node_modules/@openai/agents-realtime/dist/bundle/openai-realtime-agents.umd.js" \
   "$ROOT/web/vendor/openai-realtime-agents.umd.js"
echo "Bundle installed at web/vendor/openai-realtime-agents.umd.js"
