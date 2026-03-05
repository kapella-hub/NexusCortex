#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping NexusCortex services..."
docker compose down

echo "NexusCortex stopped."
echo "Data volumes preserved. Use 'docker compose down -v' to remove volumes."
