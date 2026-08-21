# /// script
# dependencies = ["transformers>=4.44", "accelerate", "torch", "huggingface_hub", "peft", "openenv", "pydantic"]
# ///
"""Agentic rollout evaluator for Nanthasit/hermes-tool-use-rl-env.

Drives HermesToolEnvironment in-process (no Docker/server/client.py -- that
path is broken against openenv 0.4.1 as shipped; see investigation notes).
For each model in SAK_MODELS, runs all 6 tasks once each, greedy-decoding one
<tool_call> per turn and feeding the environment's response back as the next
turn, up to max_steps. Records the binary reward (task.check() pass/fail)
per task.

Prompt rendering is copied verbatim from sakthai-bench-v2's eval_bench.py
(render_prompt/_tools_block/_render_msg) rather than tok.apply_chat_template,
because these models were trained on that exact <tools>[json schema]</tools>
+ <tool_response>...</tool_response> ChatML rendering -- confirmed by
eval_bench_peft.py scoring 91.0%/45.7% with it. An earlier version of this
script used an informal text tool list and role="user" for tool results
instead of role="tool"; that scored 0/6 across the board, which is a prompt-
format mismatch, not a capability measurement -- do not trust a 0/6 result
from a harness that hasn't been checked against the proven renderer.

SAFETY: the `terminal` tool runs model-generated shell commands via
subprocess against a tempdir workspace, in-process, with only a 10s timeout
and a workspace-relative path guard for the other tools (no container
isolation beyond the ephemeral HF Jobs runner itself). Only run this against
your own trusted fine-tuned models on an ephemeral job runner -- do not point
it at an untrusted model or run it on a persistent/shared machine.

Usage (HF Jobs):
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
      --env SAK_MODELS=Nanthasit/sakthai-context-0.5b-tools \
      ./eval_hermes_env.py

Env vars:
  SAK_MODELS    comma-separated model ids (required)
  SAK_ENV_REPO  hermes-tool-use-rl-env dataset repo (default: Nanthasit/hermes-tool-use-rl-env)
  SAK_MAX_STEPS max tool-call turns per episode (default: 12, matches TaskSpec default)
  SAK_FORCE_CPU 1 = force CPU even if CUDA available (default: 0)
  SAK_DUMP      1 = print the full first-task transcript per model, for sanity-checking (default: 0)
"""
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [m.strip() for m in os.environ.get("SAK_MODELS", "").split(",") if m.strip()]
ENV_REPO = os.environ.get("SAK_ENV_REPO", "Nanthasit/hermes-tool-use-rl-env")
MAX_STEPS = int(os.environ.get("SAK_MAX_STEPS", "12"))
FORCE_CPU = int(os.environ.get("SAK_FORCE_CPU", "0"))
DUMP = int(os.environ.get("SAK_DUMP", "0"))
assert MODELS, "set SAK_MODELS to a comma-separated list of model ids"

# ── Fetch the environment source and import it directly ───────────────────
env_dir = Path(snapshot_download(ENV_REPO, repo_type="dataset"))
sys.path.insert(0, str(env_dir))
sys.path.insert(0, str(env_dir / "server"))

from models import HermesToolAction  # noqa: E402
from tasks import TASKS_BY_ID  # noqa: E402
from hermes_tool_env import HermesToolEnvironment  # noqa: E402

TASK_IDS = sorted(TASKS_BY_ID)

# ── Renderer: copied verbatim from Nanthasit/sakthai-bench-v2's eval_bench.py
# so these models see prompts in exactly the format they were trained/scored
# on. Do not hand-roll a different tool-listing format here.
def _text(c):
    return "" if c is None else (c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))


def _tools_block(tools):
    if not tools: return ""
    sigs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return ("\n\n# Tools\n\nYou may call one or more functions. Signatures are within "
            "<tools></tools>:\n<tools>\n" + sigs + "\n</tools>\n\nFor each call return:\n"
            "<tool_call>\n{\"name\": <name>, \"arguments\": <json>}\n</tool_call>")


def _assistant_body(m):
    parts = []
    content = _text(m.get("content"))
    if content:
        parts.append(content)
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", tc)
        a = fn.get("arguments", "{}")
        if not isinstance(a, str):
            a = json.dumps(a, ensure_ascii=False)
        name = fn.get("name", "")
        parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {a}}}\n</tool_call>')
    return "\n".join(parts)


def _render_msg(m, tools_sys):
    r = m.get("role")
    if r == "system": return "<|im_start|>system\n" + _text(m.get("content")) + _tools_block(tools_sys) + "<|im_end|>\n"
    if r == "user":   return "<|im_start|>user\n" + _text(m.get("content")) + "<|im_end|>\n"
    if r == "tool":   return "<|im_start|>user\n<tool_response>\n" + _text(m.get("content")) + "\n</tool_response><|im_end|>\n"
    if r == "assistant": return "<|im_start|>assistant\n" + _assistant_body(m) + "<|im_end|>\n"
    return ""


def render_prompt(messages, tools):
    out = []
    if not (messages and messages[0].get("role") == "system") and tools:
        out.append("<|im_start|>system\nYou are a helpful assistant." + _tools_block(tools) + "<|im_end|>\n")
    for i, m in enumerate(messages):
        out.append(_render_msg(m, tools if (i == 0 and m.get("role") == "system") else None))
    out.append("<|im_start|>assistant\n")
    return "".join(out)


_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text):
    m = _TC.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def to_hermes_action(tool_call):
    return HermesToolAction(tool=tool_call["name"], **(tool_call.get("arguments") or {}))


SYSTEM_CONTENT = ("You are a coding agent working in a sandboxed workspace. Use the "
                   "available tools to complete the task, then call submit when you "
                   "believe it is solved.")

# OpenAI-function-schema form of the 5 real Hermes tools (select_task is
# env-internal/harness-only, never offered to the model -- it's used once as
# a free bootstrap step before the episode's first real turn).
TOOLS = [
    {"type": "function", "function": {
        "name": "terminal",
        "description": "Run a shell command in the task workspace; returns combined stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the task workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path of the file to read."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write or overwrite a file in the task workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path of the file to write."},
            "content": {"type": "string", "description": "Full content to write to the file."}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "patch",
        "description": "Find-and-replace a unique substring in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path of the file to patch."},
            "old_string": {"type": "string", "description": "Exact unique substring to replace."},
            "new_string": {"type": "string", "description": "Replacement text."}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "End the episode and grade the task. Call this only when you believe the task is solved.",
        "parameters": {"type": "object", "properties": {}}}},
]


# ── Model loading: same bare-PEFT-adapter handling as eval_bench_peft.py ──
def _peft_base_model(repo_id):
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id, "adapter_config.json")
    except Exception:
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"{repo_id}: adapter_config.json has no base_model_name_or_path")
    return base


def load_model_and_tokenizer(repo_id):
    device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32)
    base_id = _peft_base_model(repo_id)
    load_id = base_id or repo_id

    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
    except Exception:
        if not base_id:
            raise
        tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  loading {load_id} in {dtype} on {device}"
          + (f" (+ adapter {repo_id})" if base_id else "") + "...")
    m = AutoModelForCausalLM.from_pretrained(
        load_id, torch_dtype=dtype, device_map=device if device == "cuda" else None)
    if base_id:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, repo_id)
    if device == "cpu":
        m = m.to("cpu")
    m.eval()
    return m, tok


RENDER_MODE = os.environ.get("SAK_RENDER", "handrolled").strip()  # "handrolled" | "native"


def make_step_fn(model, tok):
    def step_fn(messages):
        if RENDER_MODE == "native":
            # Match how trl trains (SFTTrainer / GRPO environment_factory both use
            # the model's own apply_chat_template with tools). Use this to eval
            # SFT/GRPO-trained checkpoints on the SAME format they were trained on.
            enc = tok.apply_chat_template(
                messages, tools=TOOLS, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            input_len = enc["input_ids"].shape[-1]
        else:
            # bench-v2's hand-rolled renderer -- matches the ORIGINAL -tools models'
            # training format (they score 91% single-shot with it).
            prompt = render_prompt(messages, TOOLS)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            input_len = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, input_len:], skip_special_tokens=True)
    return step_fn


def run_episode(task_id, step_fn, max_steps=MAX_STEPS):
    env = HermesToolEnvironment()
    env.reset()
    obs = env.step(HermesToolAction(tool="select_task", task_id=task_id))
    # First turn is the task briefing -> "user". All later env feedback is
    # real tool output -> "tool" (rendered as <tool_response>...</tool_response>,
    # matching how eval_bench.py renders "tool" role messages).
    messages = [{"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": obs.result}]
    transcript = [{"role": "user", "text": obs.result}]

    for _ in range(max_steps):
        completion = step_fn(messages)
        messages.append({"role": "assistant", "content": completion})
        transcript.append({"role": "model", "text": completion})

        tool_call = parse_tool_call(completion)
        if tool_call is None:
            msg = "No <tool_call> found; you must call a tool."
            messages.append({"role": "tool", "content": msg})
            transcript.append({"role": "tool", "text": msg})
            continue

        try:
            action = to_hermes_action(tool_call)
            obs = env.step(action)
        except Exception as e:
            msg = f"Invalid tool call: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "content": msg})
            transcript.append({"role": "tool", "text": msg})
            continue

        messages.append({"role": "tool", "content": obs.result})
        transcript.append({"role": "tool", "text": obs.result})
        if obs.done:
            return float(obs.reward or 0.0), transcript

    return 0.0, transcript


def evaluate(repo_id):
    t0 = time.time()
    model, tok = load_model_and_tokenizer(repo_id)
    step_fn = make_step_fn(model, tok)

    per_task = {}
    for i, task_id in enumerate(TASK_IDS):
        reward, transcript = run_episode(task_id, step_fn)
        per_task[task_id] = reward
        print(f"  [{repo_id}] {task_id}: reward={reward}")
        if DUMP and i == 0:
            print(f"  --- transcript for {task_id} ---")
            for turn in transcript:
                print(f"  [{turn['role']}] {turn['text'][:300]!r}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_reward = sum(per_task.values()) / len(per_task)
    return {
        "model": repo_id,
        "seconds": round(time.time() - t0, 1),
        "per_task": per_task,
        "mean_reward": mean_reward,
        "pass_count": sum(1 for v in per_task.values() if v >= 1.0),
        "total_tasks": len(per_task),
    }


results = []
for repo in MODELS:
    try:
        results.append(evaluate(repo))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FAILED {repo}] {type(e).__name__}: {e}\n{tb}")
        results.append({"model": repo, "error": f"{type(e).__name__}: {e}", "traceback": tb})

print("\n" + "=" * 60)
print(f"{'model':<40}{'pass':>8}{'mean reward':>14}")
for r in results:
    if "error" in r:
        print(f"{r['model'][-39:]:<40}{'ERROR':>8}")
        continue
    print(f"{r['model'][-39:]:<40}{r['pass_count']}/{r['total_tasks']:<6}"
          f"{r['mean_reward']:13.2f}")

payload = {"benchmark": ENV_REPO, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "max_steps": MAX_STEPS, "results": results}
with open("hermes_env_results.json", "w") as f:
    json.dump(payload, f, indent=2)
print("\nwrote hermes_env_results.json")

print("HERMES_ENV_RESULTS_JSON_BEGIN")
print(json.dumps(payload, separators=(",", ":")))
print("HERMES_ENV_RESULTS_JSON_END")
