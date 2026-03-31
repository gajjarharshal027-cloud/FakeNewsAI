"""
inference.py — OpenEnv hackathon required file
Demonstrates an agent interacting with FakeNewsAI environment.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def run_inference():
    # Reset environment
    r = requests.post(f"{BASE_URL}/reset")
    print("Reset:", r.json())

    # Test headlines
    headlines = [
        "Trump says US could seize Iran oil terminal",
        "Elon Musk buys Google for 2 trillion dollars",
        "Scientists confirm humans can photosynthesize sunlight",
    ]

    for headline in headlines:
        r = requests.post(f"{BASE_URL}/step", json={"headline": headline})
        result = r.json()
        obs = result.get("observation", {})
        print(f"\nHeadline : {headline}")
        print(f"Verdict  : {obs.get('verdict')}")
        print(f"Confidence: {obs.get('confidence')}")
        print(f"Engine   : {obs.get('source_engine')}")
        print(f"Reward   : {result.get('reward')}")

    # Get final state
    r = requests.get(f"{BASE_URL}/state")
    print("\nFinal state:", json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    run_inference()
