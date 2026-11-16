"""Contract and lifecycle tests for OpenEnvAgentExecutor (A2A protocol skills)."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Ensure openenv-multi-catalog-training is on sys.path
MULTI_CATALOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if MULTI_CATALOG_DIR not in sys.path:
    sys.path.insert(0, MULTI_CATALOG_DIR)

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import DataPart, Message, TextPart
from a2a_agent.agent_executor import OpenEnvAgentExecutor, _Episode


@pytest.fixture
def executor():
    return OpenEnvAgentExecutor()


@pytest.fixture
def event_queue():
    return EventQueue()


def test_executor_instantiation(executor):
    assert isinstance(executor._episodes, dict)
    assert len(executor._episodes) == 0


@pytest.mark.anyio
async def test_first_message_invalid_env_fails(executor, event_queue):
    context = RequestContext(
        task_id="task_fail",
        context_id="ctx_fail",
        current_task=None,
        message=Message(parts=[TextPart("hello")], metadata={"env": "nonexistent_env"}),
    )

    await executor.execute(context, event_queue)
    assert "task_fail" not in executor._episodes


@pytest.mark.anyio
async def test_first_message_missing_env_fails(executor, event_queue):
    context = RequestContext(
        task_id="task_missing",
        context_id="ctx_missing",
        current_task=None,
        message=Message(parts=[TextPart("hello")], metadata={}),
    )

    await executor.execute(context, event_queue)
    assert "task_missing" not in executor._episodes


@pytest.mark.anyio
async def test_start_episode_echo_success(executor, event_queue):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.reset = AsyncMock(return_value=MagicMock())

    with patch.object(executor, "_open_client", AsyncMock(return_value=mock_client)):
        context = RequestContext(
            task_id="task_echo_1",
            context_id="ctx_echo_1",
            current_task=None,
            message=Message(parts=[], metadata={"env": "echo"}),
        )
        await executor.execute(context, event_queue)

        assert "task_echo_1" in executor._episodes
        episode = executor._episodes["task_echo_1"]
        assert episode.env_name == "echo"
        assert not episode.done
        assert "the falcon flies at dawn" in episode.extra["secret"]


@pytest.mark.anyio
async def test_step_episode_ongoing(executor, event_queue):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    step_result = MagicMock()
    step_result.observation = MagicMock(
        done=False,
        reward=0.0,
        messages=[MagicMock(content="Next puzzle turn")],
    )
    mock_client.step = AsyncMock(return_value=step_result)

    episode = _Episode("sudoku", mock_client)
    executor._episodes["task_step_1"] = episode

    context = RequestContext(
        task_id="task_step_1",
        context_id="ctx_step_1",
        current_task={"status": "working"},
        message=Message(parts=[TextPart("3 4 5")]),
    )

    await executor.execute(context, event_queue)
    assert "task_step_1" in executor._episodes
    assert not episode.done
    assert episode.last_reward == 0.0


@pytest.mark.anyio
async def test_step_episode_completion(executor, event_queue):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    step_result = MagicMock()
    step_result.observation = MagicMock(echoed_message="the falcon flies at dawn")
    mock_client.step = AsyncMock(return_value=step_result)

    episode = _Episode("echo", mock_client)
    episode.extra["secret"] = "the falcon flies at dawn"
    executor._episodes["task_step_done"] = episode

    context = RequestContext(
        task_id="task_step_done",
        context_id="ctx_step_done",
        current_task={"status": "working"},
        message=Message(parts=[TextPart("the falcon flies at dawn")]),
    )

    await executor.execute(context, event_queue)
    # Episode should be completed and removed from live episodes
    assert "task_step_done" not in executor._episodes
    assert episode.done
    assert episode.last_reward == 1.0
    mock_client.__aexit__.assert_awaited_once()


@pytest.mark.anyio
async def test_step_unknown_task_fails(executor, event_queue):
    context = RequestContext(
        task_id="task_unknown",
        context_id="ctx_unknown",
        current_task={"status": "working"},
        message=Message(parts=[TextPart("hello")]),
    )

    await executor.execute(context, event_queue)
    assert "task_unknown" not in executor._episodes


@pytest.mark.anyio
async def test_cancel_episode(executor, event_queue):
    mock_client = MagicMock()
    mock_client.__aexit__ = AsyncMock(return_value=None)
    episode = _Episode("echo", mock_client)
    executor._episodes["task_cancel"] = episode

    context = RequestContext(
        task_id="task_cancel",
        context_id="ctx_cancel",
        current_task={"status": "working"},
    )

    await executor.cancel(context, event_queue)
    assert "task_cancel" not in executor._episodes
    mock_client.__aexit__.assert_awaited_once()


def test_extract_action_datapart_precedence():
    data_payload = {"action_id": 2}
    context = RequestContext(
        task_id="task_data",
        context_id="ctx_data",
        message=Message(parts=[TextPart("fallback text"), DataPart(data=data_payload)]),
    )

    action = OpenEnvAgentExecutor._extract_action(context)
    assert action == data_payload
