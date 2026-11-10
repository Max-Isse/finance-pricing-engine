#!/usr/bin/env bash
# Build the image, start the compose stack and check the service end to end.
# Exits non-zero if the build fails, the service does not become healthy,
# or /price does not return a number. Always tears the stack down.
set -euo pipefail

PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- container logs ---"
    docker compose logs --no-color --tail 80 || true
  fi
  docker compose down -v >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

docker compose up --build -d

echo "Waiting for ${BASE}/health ..."
for _ in $(seq 1 60); do
  if curl -fsS "${BASE}/health" | grep -q '"status":"ok"'; then
    healthy=1
    break
  fi
  sleep 5
done
[ "${healthy:-0}" = "1" ] || { echo "Service did not become healthy"; exit 1; }

echo "Checking POST /price ..."
resp=$(curl -fsS -X POST "${BASE}/price" -H 'content-type: application/json' \
  -d '{"features":[1,2,3,4,5],"treasury_10y_yield":0.042}')
echo "$resp"
echo "$resp" | grep -Eq '"price":[ ]*-?[0-9]+(\.[0-9]+)?' || { echo "Unexpected /price response"; exit 1; }

echo "Checking GET /metrics ..."
curl -fsS "${BASE}/metrics" >/dev/null

echo "Docker smoke test passed."
