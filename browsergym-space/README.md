---
title: BrowserGym Env
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
short_description: BrowserGym OpenEnv server for SakThai GRPO RL training
app_port: 7860
---

# Nanthasit/browsergym-env

BrowserGym environment server for SakThai GRPO reinforcement-learning training.

Supports MiniWoB++ (training) and WebArena (evaluation) benchmarks via the OpenEnv Gymnasium-compatible API.

## Endpoints
- `POST /reset` — Reset environment, returns observation
- `POST /step` — Step with action, returns observation + reward
- `GET /state` — Current episode state
- `GET /health` — Health check

## Usage
```python
from browsergym_env import BrowserGymEnv

env = BrowserGymEnv(base_url="https://nanthasit-browsergym-env.hf.space")
result = env.reset()
```
