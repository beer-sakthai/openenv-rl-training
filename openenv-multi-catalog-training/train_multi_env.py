"""GRPO training of one 0.6B model across all 8 openenv/* catalog environments
at once, via TRL's environment_factory + one meta-environment class.

IMPORTANT CAVEATS — read before running:

1. VISION MISMATCH (atari_env): AtariObservation is pixel data (RGB
   [210,160,3] / grayscale [210,160] / RAM [128]). A 0.6B text-only model
   cannot see images. This script uses obs_type="ram" and renders the 128
   RAM bytes as a compact space-separated string so *something* text-shaped
   reaches the model, but this is a crude proxy, not real vision — expect
   near-random performance on atari relative to what a VLM/pixel policy
   would achieve. Flagged explicitly at the user's request to include all 8
   catalog environments despite the modality mismatch (see conversation).
2. chat_env's action space is raw model tokens (ChatAction.tokens), not a
   text tool-call — it's designed for token-level RL, not the tool-calling
   interface the other 7 environments use. This script bridges it by
   tokenizing the model's chat_reply(message) argument with the same
   model's tokenizer before sending it as the action. If chat_env's real
   API differs from the README summary this was written from, this is the
   first thing to fix.
3. coding_env and openspiel/atari/sumo have no notion of "the right
   answer" baked into a generic run — coding_env in particular returns only
   stdout/stderr/exit_code, no reward. This script gives coding_env a fixed
   toy task (an arithmetic result to print) purely so there's something to
   score; swap TASK_ANSWER for whatever the real coding task should be.
4. Each environment must be running as its own server (see run_servers.sh)
   before training starts, with SUPPORTS_CONCURRENT_SESSIONS handling left
   to each catalog image's defaults — verify concurrency headroom before
   raising num_generations, per SKILL.md's concurrency note.

Run with: python train_multi_env.py --vllm-mode colocate
"""

import argparse
import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from echo_env import EchoEnv, EchoAction
from textarena_env import TextArenaEnv, TextArenaAction
from coding_env import CodingEnv, CodeAction
from envs.chat_env import ChatEnv, ChatAction
from atari_env import AtariEnv, AtariAction
from openspiel_env import OpenSpielEnv, OpenSpielAction
from repl_env import REPLEnv, REPLAction
from envs.sumo_rl_env import SumoRLEnv, SumoAction

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")

ECHO_URL = os.environ.get("ECHO_ENV_URL", "http://localhost:8001")
SUDOKU_URL = os.environ.get("SUDOKU_ENV_URL", "http://localhost:8002")
CODING_URL = os.environ.get("CODING_ENV_URL", "http://localhost:8003")
CHAT_URL = os.environ.get("CHAT_ENV_URL", "http://localhost:8004")
ATARI_URL = os.environ.get("ATARI_ENV_URL", "http://localhost:8005")
OPENSPIEL_URL = os.environ.get("OPENSPIEL_ENV_URL", "http://localhost:8006")
REPL_URL = os.environ.get("REPL_ENV_URL", "http://localhost:8007")
SUMO_URL = os.environ.get("SUMO_ENV_URL", "http://localhost:8008")

ATARI_GAME = os.environ.get("ATARI_GAME", "pong")
CODING_TASK_PROMPT = "Write Python code that prints the result of 17 * 23, nothing else."
CODING_EXPECTED = "391"

_tokenizer = None


def _tokenizer_for_chat():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


class MultiCatalogEnv:
    """Routes one episode to one of 8 openenv/* catalog environments based
    on the `env` dataset column. See references/multi-environment.md for
    the pattern; see module docstring above for the honest caveats on
    atari_env and chat_env before trusting results from those two.
    """

    def __init__(self):
        self._clients = {k: None for k in (
            "echo", "sudoku", "coding", "chat", "atari", "openspiel", "repl", "sumo",
        )}
        self.active = None
        self.reward = 0.0
        self.done = False

    def _close(self, key: str) -> None:
        client = self._clients.get(key)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            self._clients[key] = None

    def reset(self, **kwargs) -> str | None:
        self.active = kwargs.get("env", "echo")
        self.reward = 0.0
        self.done = False
        self._close(self.active)

        if self.active == "echo":
            self._clients["echo"] = EchoEnv(base_url=ECHO_URL).sync()
            self._clients["echo"].__enter__()
            self._clients["echo"].reset()
            self._secret_phrase = "the falcon flies at dawn"
            return f"Echo this phrase back exactly: '{self._secret_phrase}'"

        if self.active == "sudoku":
            self._clients["sudoku"] = TextArenaEnv(base_url=SUDOKU_URL).sync()
            self._clients["sudoku"].__enter__()
            result = self._clients["sudoku"].reset()
            return result.observation.messages[0].content if result.observation.messages else result.observation.prompt

        if self.active == "coding":
            self._clients["coding"] = CodingEnv(base_url=CODING_URL).sync()
            self._clients["coding"].__enter__()
            self._clients["coding"].reset()
            return CODING_TASK_PROMPT

        if self.active == "chat":
            self._clients["chat"] = ChatEnv(base_url=CHAT_URL).sync()
            self._clients["chat"].__enter__()
            result = self._clients["chat"].reset()
            return "Have a short, helpful conversation. Reply with chat_reply(message)."

        if self.active == "atari":
            self._clients["atari"] = AtariEnv(base_url=ATARI_URL).sync()
            self._clients["atari"].__enter__()
            result = self._clients["atari"].reset()
            return self._format_atari_obs(result.observation)

        if self.active == "openspiel":
            self._clients["openspiel"] = OpenSpielEnv(base_url=OPENSPIEL_URL).sync()
            self._clients["openspiel"].__enter__()
            result = self._clients["openspiel"].reset()
            return self._format_openspiel_obs(result.observation)

        if self.active == "repl":
            self._clients["repl"] = REPLEnv(base_url=REPL_URL).sync()
            self._clients["repl"].__enter__()
            result = self._clients["repl"].reset(
                context="alpha beta gamma delta", task_prompt="Count the words in context."
            )
            return result.observation.context_preview

        if self.active == "sumo":
            self._clients["sumo"] = SumoRLEnv(base_url=SUMO_URL).sync()
            self._clients["sumo"].__enter__()
            result = self._clients["sumo"].reset()
            return self._format_sumo_obs(result.observation)

        raise ValueError(f"Unknown env routing value: {self.active!r}")

    # -- formatting helpers -------------------------------------------------

    def _format_atari_obs(self, obs) -> str:
        ram = getattr(obs, "screen", [])[:128]
        return f"Atari RAM state ({len(ram)} bytes): {' '.join(map(str, ram))}. Legal actions: {obs.legal_actions}"

    def _format_openspiel_obs(self, obs) -> str:
        state = ",".join(f"{v:.2f}" for v in obs.info_state[:32])
        return f"Game state (truncated): [{state}...]. Legal actions: {obs.legal_actions}. Phase: {obs.game_phase}"

    def _format_sumo_obs(self, obs) -> str:
        vec = ",".join(f"{v:.2f}" for v in obs.observation[:16])
        return f"Traffic state (truncated): [{vec}...]. Action mask: {obs.action_mask}. Sim time: {obs.sim_time}"

    # -- per-environment tools ------------------------------------------------
    # Every public method below is discovered as a tool regardless of which
    # sub-environment is active this episode; each checks self.active and
    # raises if called out of turn, per the multi-environment pattern.

    def echo_say(self, message: str) -> str:
        """Echo the given message back through the echo environment.

        Args:
            message: the exact text to send.
        """
        if self.active != "echo":
            raise ValueError("echo_say is only valid during the echo task.")
        result = self._clients["echo"].step(EchoAction(message=message))
        self.reward = 1.0 if result.observation.echoed_message == self._secret_phrase else 0.0
        self.done = result.observation.done or True
        return result.observation.echoed_message

    def sudoku_move(self, move: str) -> str:
        """Submit a Sudoku move/message to the TextArena Sudoku environment.

        Args:
            move: the move text in the format the game prompt specifies.
        """
        if self.active != "sudoku":
            raise ValueError("sudoku_move is only valid during the sudoku task.")
        result = self._clients["sudoku"].step(TextArenaAction(message=move))
        self.reward = result.observation.reward or 0.0
        self.done = result.observation.done
        msgs = result.observation.messages
        return msgs[-1].content if msgs else ""

    def coding_run(self, code: str) -> str:
        """Execute Python code in the coding environment and see stdout/stderr.

        Args:
            code: the full Python source to run.
        """
        if self.active != "coding":
            raise ValueError("coding_run is only valid during the coding task.")
        result = self._clients["coding"].step(CodeAction(code=code))
        obs = result.observation
        self.reward = 1.0 if obs.exit_code == 0 and CODING_EXPECTED in obs.stdout else 0.0
        self.done = True
        return f"stdout: {obs.stdout}\nstderr: {obs.stderr}\nexit_code: {obs.exit_code}"

    def chat_reply(self, message: str) -> str:
        """Send a chat reply. Tokenized with the policy model's own tokenizer
        before being sent as the environment action (see module docstring —
        chat_env's native action is raw tokens, not text).

        Args:
            message: the reply text.
        """
        if self.active != "chat":
            raise ValueError("chat_reply is only valid during the chat task.")
        tokens = _tokenizer_for_chat()(message, return_tensors="pt")["input_ids"][0]
        result = self._clients["chat"].step(ChatAction(tokens=tokens))
        self.reward = getattr(result.observation, "reward", 0.0) or 0.0
        self.done = result.observation.done
        return _tokenizer_for_chat().decode(result.observation.tokens, skip_special_tokens=True)

    def atari_press(self, action_id: int) -> str:
        """Press a button in the Atari environment (game fixed at reset).

        Args:
            action_id: the discrete action id from the reported legal_actions list.
        """
        if self.active != "atari":
            raise ValueError("atari_press is only valid during the atari task.")
        result = self._clients["atari"].step(
            AtariAction(action_id=action_id, game_name=ATARI_GAME, obs_type="ram")
        )
        self.reward = result.observation.reward or 0.0
        self.done = result.observation.done
        return self._format_atari_obs(result.observation)

    def openspiel_move(self, action_id: int) -> str:
        """Play a move in the OpenSpiel game.

        Args:
            action_id: the discrete action id from the reported legal_actions list.
        """
        if self.active != "openspiel":
            raise ValueError("openspiel_move is only valid during the openspiel task.")
        result = self._clients["openspiel"].step(OpenSpielAction(action_id=action_id))
        self.reward = result.observation.reward or 0.0
        self.done = result.observation.done
        return self._format_openspiel_obs(result.observation)

    def repl_execute(self, code: str) -> str:
        """Run one line/block of Python in the REPL environment's persistent context.

        Args:
            code: the code to execute this step.
        """
        if self.active != "repl":
            raise ValueError("repl_execute is only valid during the repl task.")
        result = self._clients["repl"].step(REPLAction(code=code))
        self.reward = result.observation.reward or 0.0
        self.done = result.observation.done
        return result.observation.result.stdout

    def repl_finish(self, final_answer: str) -> str:
        """Submit the final answer and end the REPL episode.

        Args:
            final_answer: the computed final answer as a string.
        """
        if self.active != "repl":
            raise ValueError("repl_finish is only valid during the repl task.")
        result = self._clients["repl"].step(REPLAction(code="", is_final=True, final_answer=final_answer))
        self.reward = result.observation.reward or 0.0
        self.done = True
        return result.observation.result.stdout

    def sumo_set_phase(self, phase_id: int) -> str:
        """Set the traffic signal phase in the SUMO environment.

        Args:
            phase_id: the discrete phase id to switch the signal to.
        """
        if self.active != "sumo":
            raise ValueError("sumo_set_phase is only valid during the sumo task.")
        result = self._clients["sumo"].step(SumoAction(phase_id=phase_id))
        self.reward = result.observation.reward or 0.0
        self.done = result.observation.done
        return self._format_sumo_obs(result.observation)


# ============================================================================
# Reward functions — one per environment, None for episodes that aren't its own
# ============================================================================

def _reward_for(task: str):
    def _fn(environments, **kwargs) -> list[float | None]:
        return [env.reward if env.active == task else None for env in environments]

    _fn.__name__ = f"{task}_reward"
    return _fn


echo_reward = _reward_for("echo")
sudoku_reward = _reward_for("sudoku")
coding_reward = _reward_for("coding")
chat_reward = _reward_for("chat")
atari_reward = _reward_for("atari")
openspiel_reward = _reward_for("openspiel")
repl_reward = _reward_for("repl")
sumo_reward = _reward_for("sumo")

REWARD_FUNCS = [
    echo_reward, sudoku_reward, coding_reward, chat_reward,
    atari_reward, openspiel_reward, repl_reward, sumo_reward,
]


# ============================================================================
# Dataset — equal episode count per environment, tagged with routing column
# ============================================================================

def build_dataset(n_per_env: int = 32) -> Dataset:
    prompts = {
        "echo": "You are in the echo task. Call echo_say with the exact phrase you're given.",
        "sudoku": "You are playing Sudoku via TextArena. Read the board and submit moves.",
        "coding": "You are in the coding task. Use coding_run to solve the stated problem.",
        "chat": "You are in a short open-ended chat. Use chat_reply to respond.",
        "atari": f"You are playing Atari {ATARI_GAME} from RAM state. Use atari_press.",
        "openspiel": "You are playing a game via OpenSpiel. Use openspiel_move.",
        "repl": "You have a Python REPL with persistent context. Use repl_execute, then repl_finish.",
        "sumo": "You are controlling a traffic signal. Use sumo_set_phase to optimize flow.",
    }
    envs, rows = [], []
    for task, prompt in prompts.items():
        envs += [task] * n_per_env
        rows += [[{"role": "user", "content": prompt}]] * n_per_env
    return Dataset.from_dict({"prompt": rows, "env": envs})


# ============================================================================
# Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-server-url", default="http://localhost:8000")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--n-per-env", type=int, default=32)
    args = parser.parse_args()

    grpo_kwargs = dict(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )
    if args.vllm_mode == "server":
        grpo_kwargs["vllm_server_url"] = args.vllm_server_url

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=build_dataset(args.n_per_env),
        reward_funcs=REWARD_FUNCS,
        args=GRPOConfig(**grpo_kwargs),
        environment_factory=MultiCatalogEnv,
    )
    trainer.train()

    # trainer.push_to_hub("your-org/your-model-name")


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# Watch train/reward_func_0 .. train/reward_func_7 individually — the combined
# train/reward metric alternates across 8 tasks batch to batch and will look
# noisy even when training is healthy. See run_servers.sh to start all 8
# environment servers before running this script.
# ----------------------------------------------------------------------------
