# Plan: OpenEnv RL Training Improvements (GRPO + TRL)

**Goal:** Upgrade, harden, and verify the `openenv-rl-training` codebase for 7B GRPO reinforcement learning, fix GRPOConfig bugs, add in-process contract verification, and standardize tool chat templates.

---

## Task 1: Create CPU Smoke-Test Verification Script (`verify_grpo_contract.py`)

- [ ] Create `openenv-rl-training/verify_grpo_contract.py` to smoke-test `TRL`'s `GRPOTrainer`, `GRPOConfig`, and Gymnasium environment factory locally on CPU.
- [ ] Implement mock rollout reward checking to verify non-zero reward variance detection logic.
- [ ] Run `uv run python openenv-rl-training/verify_grpo_contract.py` to confirm clean execution.

---

## Task 2: Fix GRPO Config & Precision Parameters in `openenv-custom-training`

- [ ] Update `openenv-rl-training/openenv-custom-training/train.py` and `multi_env.py`:
  - Replace deprecated `vllm_server_url` with valid `vllm_server_host` and `vllm_server_port` parameters.
  - Enforce `torch.bfloat16` precision saving on merge-out to prevent fp32 checkpoint bloat.
  - Target `Nanthasit/sakthai-context-7b-tools` as default GRPO base model.
- [ ] Run format/lint checks on modified python files.

---

## Task 3: Harden Multi-Catalog Environment Schemas (`openenv-multi-catalog-training`)

- [ ] Update `openenv-rl-training/openenv-multi-catalog-training/` action schemas and state parsers for all 8 catalog environments.
- [ ] Ensure `environment_factory` compatibility with native `apply_chat_template(tools=...)`.
- [ ] Test environment factory instantiation.

---

## Task 4: Update Documentation & GPU Run Instructions

- [ ] Update `openenv-rl-training/README.md` with 7B GRPO training findings, HF Jobs commands (`hf jobs uv run`), and hardware requirements.
- [ ] Validate markdown links and formatting.
