#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== spark-brain status ==="
echo

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  STARTED_AT=$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')
  echo "  Container: running (started $STARTED_AT)"
else
  echo "  Container: NOT RUNNING"
fi

if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
  echo "  API:       healthy (http://localhost:$VLLM_PORT)"
else
  echo "  API:       not reachable"
fi

echo
echo "=== System memory ==="
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "  'free' is not available on this system"
fi

echo
echo "=== vLLM metrics ==="
METRICS="$(curl -sf "http://localhost:$VLLM_PORT/metrics" 2>/dev/null || true)"
if [[ -n "$METRICS" ]]; then
  KV_CACHE="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /gpu_cache_usage_perc/ {printf "%.1f%%", $2 * 100; exit}')"
  RUNNING="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /(^|:)num_requests_running([[:space:]]|$)/ {print $2; exit}')"
  WAITING="$(printf '%s\n' "$METRICS" | awk '!/^#/ && /(^|:)num_requests_waiting([[:space:]]|$)/ {print $2; exit}')"
  echo "  KV cache used:     ${KV_CACHE:-unknown}"
  echo "  Requests running:  ${RUNNING:-0}"
  echo "  Requests waiting:  ${WAITING:-0}"
else
  echo "  Metrics endpoint not available"
fi

echo
echo "=== Recent logs ==="
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker logs "$CONTAINER_NAME" --tail 10 2>&1 | sed 's/^/  /'
else
  echo "  No container found"
fi
