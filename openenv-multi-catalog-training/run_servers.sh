#!/usr/bin/env bash
# Starts all 8 openenv/* catalog environment servers as local Docker
# containers, one port each, matching the *_URL defaults in
# train_multi_env.py. Run this before train_multi_env.py.
#
# Flags:
#   --dry-run, -n   Print planned docker run commands without executing them
#   --status, -s    Inspect status of running openenv-* containers
#   --help, -h      Show help message
#
# Pull the image names via each Space's "Run locally" menu if these tags
# have moved; these match the openenv/<name> convention as of this writing.

set -euo pipefail

DRY_RUN=false
CHECK_STATUS=false

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n)
      DRY_RUN=true
      ;;
    --status|-s)
      CHECK_STATUS=true
      ;;
    --help|-h)
      echo "Usage: $0 [--dry-run|-n] [--status|-s] [--help|-h]"
      echo "Starts 8 openenv/* catalog environment containers matching train_multi_env.py ports."
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      echo "Usage: $0 [--dry-run|-n] [--status|-s] [--help|-h]" >&2
      exit 1
      ;;
  esac
done

CONTAINERS=(
  "echo:8001:registry.hf.space/openenv-echo-env:latest"
  "sudoku:8002:registry.hf.space/openenv-sudoku:latest"
  "coding:8003:registry.hf.space/openenv-coding-env:latest"
  "chat:8004:registry.hf.space/openenv-chat-env:latest"
  "atari:8005:registry.hf.space/openenv-atari-env:latest"
  "openspiel:8006:registry.hf.space/openenv-openspiel-env:latest"
  "repl:8007:registry.hf.space/openenv-repl:latest"
  "sumo:8008:registry.hf.space/openenv-sumo-rl-env:latest"
)

if [ "$CHECK_STATUS" = true ]; then
  echo "Checking status of openenv catalog containers..."
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker command not found." >&2
    exit 1
  fi
  docker ps -a --filter "name=openenv-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
  exit 0
fi

for entry in "${CONTAINERS[@]}"; do
  IFS=":" read -r name port image <<< "$entry"
  if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] docker run -d --name \"openenv-$name\" -p \"$port:8000\" --platform linux/amd64 \"$image\""
  else
    echo "starting $name on :$port ..."
    docker run -d --name "openenv-$name" -p "$port:8000" --platform linux/amd64 "$image"
  fi
done

if [ "$DRY_RUN" = true ]; then
  echo "[dry-run] Plan complete: 8 containers planned."
else
  echo "all 8 containers started. 'docker ps' to verify, 'docker logs openenv-<name>' to debug."
fi

