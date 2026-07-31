"""A2A server exposing all 8 openenv/* catalog environments as skills.

Start the 8 environment containers first (../run_servers.sh), then:

    python server.py --port 9000

Any A2A client can then fetch the agent card at
http://localhost:9000/.well-known/agent.json and send messages — see
client_smoke_test.py for the minimal round-trip (start an echo episode,
send the phrase back, read the reward).

Untested in this session (see agent_executor.py's module docstring for
why) — this is written from the published a2a-python SDK sample shape,
not verified against a live install.
"""

import argparse

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import OpenEnvAgentExecutor

SKILLS = [
    AgentSkill(
        id="echo",
        name="Echo task",
        description="Echo a given phrase back exactly. Reward 1.0 for an exact match, else 0.0.",
        tags=["text", "openenv"],
        examples=['metadata={"env": "echo"}, then reply with the phrase you were given'],
    ),
    AgentSkill(
        id="sudoku",
        name="Sudoku (TextArena)",
        description="Play Sudoku via TextArena; submit moves in the format the board prompt specifies.",
        tags=["game", "reasoning", "openenv"],
        examples=['metadata={"env": "sudoku"}'],
    ),
    AgentSkill(
        id="coding",
        name="Coding sandbox",
        description="Run Python code and see stdout/stderr/exit_code. Demo task: print 17 * 23.",
        tags=["code", "openenv"],
        examples=['metadata={"env": "coding"}'],
    ),
    AgentSkill(
        id="chat",
        name="Open-ended chat",
        description="Short open-ended conversation with the chat environment.",
        tags=["chat", "openenv"],
        examples=['metadata={"env": "chat"}'],
    ),
    AgentSkill(
        id="atari",
        name="Atari (RAM state)",
        description="Play Atari Pong from a text-rendered RAM byte dump (not real vision — see README caveat).",
        tags=["game", "openenv"],
        examples=['metadata={"env": "atari"}, then a DataPart {"action_id": N}'],
    ),
    AgentSkill(
        id="openspiel",
        name="OpenSpiel game",
        description="Play a game via OpenSpiel; observation is a numeric info-state vector.",
        tags=["game", "openenv"],
        examples=['metadata={"env": "openspiel"}, then a DataPart {"action_id": N}'],
    ),
    AgentSkill(
        id="repl",
        name="Python REPL",
        description="Persistent-context Python REPL; execute steps, then submit a final answer.",
        tags=["code", "openenv"],
        examples=['metadata={"env": "repl"}'],
    ),
    AgentSkill(
        id="sumo",
        name="SUMO traffic control",
        description="Control a traffic signal's phase to optimize simulated traffic flow.",
        tags=["control", "openenv"],
        examples=['metadata={"env": "sumo"}, then a DataPart {"phase_id": N}'],
    ),
]


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="OpenEnv Catalog Agent",
        description="Wraps all 8 openenv/* catalog environments as A2A skills, one episode per task.",
        url=base_url,
        version="0.1.0",
        default_input_modes=["text", "data"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=SKILLS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/"
    agent_card = build_agent_card(base_url)
    handler = DefaultRequestHandler(agent_executor=OpenEnvAgentExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
