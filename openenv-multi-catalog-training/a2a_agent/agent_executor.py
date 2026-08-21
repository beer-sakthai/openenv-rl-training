"""AgentExecutor that exposes all 8 openenv/* catalog environments as A2A
skills, independent of the GRPO training script next door — this wraps the
same environment clients for direct, single-episode-per-task use by any
A2A-speaking caller (no TRL/GRPO involved here at all).

Not executed or tested in this session: the a2a-sdk package isn't
installed in this container (no network-verified pip install, no way to
run a server here). The AgentExecutor/RequestContext/TaskUpdater shapes
below follow the published a2a-python SDK sample pattern from training
data, not a live check against your installed version — diff this against
`python -c "import a2a; print(a2a.__version__)"` and the SDK's own
helloworld sample before trusting it, especially the TaskUpdater method
names.

One A2A task == one OpenEnv episode:
  - First message in a task must carry `metadata={"env": "<name>"}`
    (one of: echo, sudoku, coding, chat, atari, openspiel, repl, sumo) —
    this resets the matching environment and returns the initial
    observation, with the task left in `input_required`.
  - Each subsequent message in the same task is one step's action; the
    task stays `input_required` until the environment reports `done`,
    at which point it's marked `completed` with the final reward in the
    artifact.
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart
from a2a.utils import new_agent_text_message
from atari_env import AtariAction, AtariEnv
from coding_env import CodeAction, CodingEnv
from echo_env import EchoAction, EchoEnv
from envs.chat_env import ChatAction, ChatEnv
from envs.sumo_rl_env import SumoAction, SumoRLEnv
from openspiel_env import OpenSpielAction, OpenSpielEnv
from repl_env import REPLAction, REPLEnv
from textarena_env import TextArenaAction, TextArenaEnv

ENV_URLS = {
    "echo": "http://localhost:8001",
    "sudoku": "http://localhost:8002",
    "coding": "http://localhost:8003",
    "chat": "http://localhost:8004",
    "atari": "http://localhost:8005",
    "openspiel": "http://localhost:8006",
    "repl": "http://localhost:8007",
    "sumo": "http://localhost:8008",
}

ATARI_GAME = "pong"

# chat_env's action is raw model tokens (ChatAction.tokens), not text — see
# module docstring. Must match whatever model actually plays this skill;
# override via MODEL_NAME if that's not the training script's default.
import os

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")

_tokenizer = None


def _tokenizer_for_chat():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


class _Episode:
    """One live OpenEnv session, keyed by A2A task_id."""

    def __init__(self, env_name: str, client):
        self.env_name = env_name
        self.client = client
        self.done = False
        self.last_reward = 0.0
        self.extra: dict[str, str] = {}  # per-env scratch (e.g. echo's secret phrase)


class OpenEnvAgentExecutor(AgentExecutor):
    def __init__(self):
        self._episodes: dict[str, _Episode] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if context.current_task is None:
            await updater.submit()
            await self._start_episode(context, updater)
            return

        await self._step_episode(context, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        episode = self._episodes.pop(context.task_id, None)
        if episode is not None:
            await episode.client.__aexit__(None, None, None)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

    # -- episode lifecycle ---------------------------------------------------

    async def _start_episode(
        self, context: RequestContext, updater: TaskUpdater
    ) -> None:
        env_name = (
            (context.message.metadata or {}).get("env") if context.message else None
        )
        if env_name not in ENV_URLS:
            await updater.failed(
                message=new_agent_text_message(
                    f"First message must carry metadata.env in {list(ENV_URLS)}, got {env_name!r}"
                )
            )
            return

        await updater.start_work()
        client = await self._open_client(env_name)
        episode = _Episode(env_name, client)
        self._episodes[context.task_id] = episode

        obs_text = await self._reset(episode)
        await updater.update_status(
            state="input_required" if not episode.done else "completed",
            message=new_agent_text_message(
                obs_text, context_id=context.context_id, task_id=context.task_id
            ),
        )

    async def _step_episode(
        self, context: RequestContext, updater: TaskUpdater
    ) -> None:
        episode = self._episodes.get(context.task_id)
        if episode is None:
            await updater.failed(
                message=new_agent_text_message("Unknown task_id; episode not found.")
            )
            return
        if episode.done:
            await updater.failed(
                message=new_agent_text_message("Episode already completed.")
            )
            return

        action_payload = self._extract_action(context)
        obs_text = await self._step(episode, action_payload)

        if episode.done:
            await updater.update_status(
                state="completed",
                message=new_agent_text_message(
                    f"{obs_text}\n[reward={episode.last_reward}]",
                    context_id=context.context_id,
                    task_id=context.task_id,
                ),
            )
            await episode.client.__aexit__(None, None, None)
            self._episodes.pop(context.task_id, None)
        else:
            await updater.update_status(
                state="input_required",
                message=new_agent_text_message(
                    obs_text, context_id=context.context_id, task_id=context.task_id
                ),
            )

    @staticmethod
    def _extract_action(context: RequestContext):
        """DataPart (JSON) wins if present — needed for int-valued actions
        (atari/openspiel/sumo). Otherwise falls back to the raw text part.
        """
        for part in context.message.parts if context.message else []:
            root = part.root if hasattr(part, "root") else part
            if isinstance(root, DataPart):
                return root.data
        return context.get_user_input()

    # -- per-environment reset/step -------------------------------------------

    @staticmethod
    async def _open_client(env_name: str):
        url = ENV_URLS[env_name]
        clients = {
            "echo": EchoEnv,
            "sudoku": TextArenaEnv,
            "coding": CodingEnv,
            "chat": ChatEnv,
            "atari": AtariEnv,
            "openspiel": OpenSpielEnv,
            "repl": REPLEnv,
            "sumo": SumoRLEnv,
        }
        client = clients[env_name](base_url=url)
        await client.__aenter__()
        return client

    async def _reset(self, episode: _Episode) -> str:
        name, client = episode.env_name, episode.client
        if name == "echo":
            await client.reset()
            episode.extra["secret"] = "the falcon flies at dawn"
            return f"Reply with the exact phrase: '{episode.extra['secret']}'"
        if name == "sudoku":
            result = await client.reset()
            msgs = result.observation.messages
            return msgs[0].content if msgs else result.observation.prompt
        if name == "coding":
            await client.reset()
            return "Write Python code that prints the result of 17 * 23, nothing else."
        if name == "chat":
            await client.reset()
            return "Say something to start the conversation."
        if name == "atari":
            result = await client.reset()
            return self._format_ram(result.observation)
        if name == "openspiel":
            result = await client.reset()
            return self._format_vec(
                "state", result.observation.info_state, result.observation.legal_actions
            )
        if name == "repl":
            result = await client.reset(
                context="alpha beta gamma delta", task_prompt="Count the words."
            )
            return result.observation.context_preview
        if name == "sumo":
            result = await client.reset()
            return self._format_vec(
                "traffic",
                result.observation.observation,
                result.observation.action_mask,
            )
        raise ValueError(name)

    async def _step(self, episode: _Episode, action) -> str:
        name, client = episode.env_name, episode.client

        if name == "echo":
            result = await client.step(EchoAction(message=str(action)))
            episode.last_reward = (
                1.0
                if result.observation.echoed_message == episode.extra["secret"]
                else 0.0
            )
            episode.done = True
            return result.observation.echoed_message

        if name == "sudoku":
            result = await client.step(TextArenaAction(message=str(action)))
            episode.last_reward = result.observation.reward or 0.0
            episode.done = result.observation.done
            msgs = result.observation.messages
            return msgs[-1].content if msgs else ""

        if name == "coding":
            result = await client.step(CodeAction(code=str(action)))
            obs = result.observation
            episode.last_reward = (
                1.0 if obs.exit_code == 0 and "391" in obs.stdout else 0.0
            )
            episode.done = True
            return f"stdout: {obs.stdout}\nstderr: {obs.stderr}\nexit_code: {obs.exit_code}"

        if name == "chat":
            tokens = _tokenizer_for_chat()(str(action), return_tensors="pt")[
                "input_ids"
            ][0]
            result = await client.step(ChatAction(tokens=tokens))
            episode.last_reward = getattr(result.observation, "reward", 0.0) or 0.0
            episode.done = result.observation.done
            return _tokenizer_for_chat().decode(
                result.observation.tokens, skip_special_tokens=True
            )

        if name == "atari":
            action_id = (
                int(action["action_id"]) if isinstance(action, dict) else int(action)
            )
            result = await client.step(
                AtariAction(action_id=action_id, game_name=ATARI_GAME, obs_type="ram")
            )
            episode.last_reward = result.observation.reward or 0.0
            episode.done = result.observation.done
            return self._format_ram(result.observation)

        if name == "openspiel":
            action_id = (
                int(action["action_id"]) if isinstance(action, dict) else int(action)
            )
            result = await client.step(OpenSpielAction(action_id=action_id))
            episode.last_reward = result.observation.reward or 0.0
            episode.done = result.observation.done
            return self._format_vec(
                "state", result.observation.info_state, result.observation.legal_actions
            )

        if name == "repl":
            if isinstance(action, dict) and action.get("final_answer") is not None:
                result = await client.step(
                    REPLAction(
                        code="", is_final=True, final_answer=str(action["final_answer"])
                    )
                )
                episode.done = True
            else:
                result = await client.step(REPLAction(code=str(action)))
                episode.done = result.observation.done
            episode.last_reward = result.observation.reward or 0.0
            return result.observation.result.stdout

        if name == "sumo":
            phase_id = (
                int(action["phase_id"]) if isinstance(action, dict) else int(action)
            )
            result = await client.step(SumoAction(phase_id=phase_id))
            episode.last_reward = result.observation.reward or 0.0
            episode.done = result.observation.done
            return self._format_vec(
                "traffic",
                result.observation.observation,
                result.observation.action_mask,
            )

        raise ValueError(name)

    @staticmethod
    def _format_ram(obs) -> str:
        ram = getattr(obs, "screen", [])[:128]
        return f"RAM ({len(ram)} bytes): {' '.join(map(str, ram))}. Legal actions: {obs.legal_actions}"

    @staticmethod
    def _format_vec(label: str, vec, legal) -> str:
        vec_str = ",".join(f"{v:.2f}" for v in list(vec)[:16])
        return f"{label} (truncated): [{vec_str}...]. Legal/mask: {legal}"
