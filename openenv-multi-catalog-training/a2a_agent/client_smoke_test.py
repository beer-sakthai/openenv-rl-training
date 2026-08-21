"""Minimal round-trip smoke test: run an echo episode end-to-end via A2A.

    python client_smoke_test.py --base-url http://localhost:9000/

Untested in this session (no a2a-sdk install here) — check this against
the SDK's own client sample if `A2ACardResolver`/`A2AClient`'s method
names have moved since this was written.
"""

import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, Client as A2AClient
from a2a.types import Message, Part, SendMessageRequest


async def send(client: A2AClient, context_id: str | None, task_id: str | None, text: str, metadata=None):
    message = Message(
        role="user",
        parts=[Part(text=text)],
        message_id=str(uuid.uuid4()),
        context_id=context_id,
        task_id=task_id,
        metadata=metadata,
    )
    request = SendMessageRequest(message=message)
    response = await client.send_message(request)
    return response


async def main(base_url: str):
    async with httpx.AsyncClient() as http_client:
        card = await A2ACardResolver(http_client, base_url=base_url).get_agent_card()
        print(f"connected to: {card.name} ({len(card.skills)} skills)")

        client = A2AClient(http_client, agent_card=card)

        r1 = await send(client, None, None, "start", metadata={"env": "echo"})
        print("reset ->", r1)

        task = r1.root.result
        r2 = await send(client, task.context_id, task.id, "the falcon flies at dawn")
        print("step ->", r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9000/")
    args = parser.parse_args()
    asyncio.run(main(args.base_url))
