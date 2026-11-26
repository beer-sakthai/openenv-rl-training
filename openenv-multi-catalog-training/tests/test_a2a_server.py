"""Contract test for a2a_agent/server.py's agent-card wiring.

server.py's own docstring says it is untested against a live a2a-python
install. build_agent_card() and SKILLS are pure data/construction, though,
and don't need uvicorn or a running server to exercise -- this locks in
that every catalog environment advertised in train_multi_env.py's dataset
also has a matching A2A skill, so the two can't silently drift apart.
"""

import os
import sys

# server.py imports its sibling `agent_executor` module as a bare top-level
# name (not package-relative), i.e. it expects to run as `python server.py`
# from inside a2a_agent/ -- so that directory itself must be on sys.path,
# not its parent.
A2A_AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "a2a_agent"))
if A2A_AGENT_DIR not in sys.path:
    sys.path.insert(0, A2A_AGENT_DIR)

from server import SKILLS, build_agent_card  # noqa: E402

CATALOG_ENV_IDS = {"echo", "sudoku", "coding", "chat", "atari", "openspiel", "repl", "sumo"}


def test_skills_cover_all_eight_catalog_envs():
    """One AgentSkill per openenv/* catalog env, matching train_multi_env.py's dataset envs."""
    assert {skill.id for skill in SKILLS} == CATALOG_ENV_IDS
    assert len(SKILLS) == 8


def test_skills_have_name_and_description():
    for skill in SKILLS:
        assert skill.name
        assert skill.description


def test_build_agent_card_wires_url_and_skills():
    card = build_agent_card("http://localhost:9000/")
    assert card.url == "http://localhost:9000/"
    assert card.skills == SKILLS
    assert card.capabilities.streaming is True
