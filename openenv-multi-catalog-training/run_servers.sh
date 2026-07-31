#!/usr/bin/env bash
# Starts all 8 openenv/* catalog environment servers as local Docker
# containers, one port each, matching the *_URL defaults in
# train_multi_env.py. Run this before train_multi_env.py.
#
# Pull the image names via each Space's "Run locally" menu if these tags
# have moved; these match the openenv/<name> convention as of this writing.

set -euo pipefail

run() {
  local name=$1 port=$2 image=$3
  echo "starting $name on :$port ..."
  docker run -d --name "openenv-$name" -p "$port:8000" --platform linux/amd64 "$image"
}

run echo      8001 registry.hf.space/openenv-echo-env:latest
run sudoku    8002 registry.hf.space/openenv-sudoku:latest
run coding    8003 registry.hf.space/openenv-coding-env:latest
run chat      8004 registry.hf.space/openenv-chat-env:latest
run atari     8005 registry.hf.space/openenv-atari-env:latest
run openspiel 8006 registry.hf.space/openenv-openspiel-env:latest
run repl      8007 registry.hf.space/openenv-repl:latest
run sumo      8008 registry.hf.space/openenv-sumo-rl-env:latest

echo "all 8 containers started. 'docker ps' to verify, 'docker logs openenv-<name>' to debug."
