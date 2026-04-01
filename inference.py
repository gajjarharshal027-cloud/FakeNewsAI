"""
inference.py — OpenEnv hackathon required file.
Demonstrates an agent interacting with the FakeNewsAI environment
using the openenv client library.
"""
import asyncio
from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult
from models import HeadlineAction, HeadlineObservation, HeadlineState

BASE_URL = "http://localhost:8000"


class FakeNewsEnv(HTTPEnvClient[HeadlineAction, HeadlineObservation]):
    def _step_payload(self, action: HeadlineAction) -> dict:
        return {"headline": action.headline}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        obs = HeadlineObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> HeadlineState:
        return HeadlineState(**payload)


async def run_inference():
    async with FakeNewsEnv(base_url=BASE_URL) as env:
        # Reset environment
        result = await env.reset()
        print("Reset:", result.observation)

        # Test headlines
        headlines = [
            "Trump says US could seize Iran oil terminal",
            "Elon Musk buys Google for 2 trillion dollars",
            "Scientists confirm humans can photosynthesize sunlight",
        ]

        for headline in headlines:
            result = await env.step(HeadlineAction(headline=headline))
            obs = result.observation
            print(f"\nHeadline : {headline}")
            print(f"Verdict  : {obs.verdict}")
            print(f"Confidence: {obs.confidence}")
            print(f"Engine   : {obs.source_engine}")
            print(f"Reward   : {result.reward}")

        # Get final state
        state = await env.state()
        print("\nFinal state:", state)


if __name__ == "__main__":
    asyncio.run(run_inference())
