"""
server/fake_news_environment.py — Core FakeNewsAI environment logic.
Extends openenv.core.env_server.interfaces.Environment.
"""
import os
import uuid
import httpx
import json
import re
from openenv.core.env_server.interfaces import Environment
from models import HeadlineAction, HeadlineObservation, HeadlineState

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class FakeNewsEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._state = HeadlineState(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self) -> HeadlineObservation:
        self._state = HeadlineState(episode_id=str(uuid.uuid4()), step_count=0)
        return HeadlineObservation(
            headline="",
            verdict=None,
            confidence=None,
            explanation="Environment reset. Submit a headline to verify.",
            source_engine=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: HeadlineAction) -> HeadlineObservation:
        self._state.step_count += 1
        self._state.last_headline = action.headline

        obs = self._classify_sync(action.headline, action.groq_api_key or GROQ_API_KEY)
        self._state.last_verdict = obs.verdict
        return obs

    @property
    def state(self) -> HeadlineState:
        return self._state

    # ─── classification helpers ────────────────────────────────────────────

    def _classify_sync(self, headline: str, groq_key: str) -> HeadlineObservation:
        # 1. Try NewsAPI
        if NEWS_API_KEY:
            try:
                with httpx.Client(timeout=10) as client:
                    r = client.get(
                        "https://newsapi.org/v2/everything",
                        params={"q": headline[:100], "pageSize": 5,
                                "apiKey": NEWS_API_KEY, "language": "en"},
                    )
                    data = r.json()
                    articles = data.get("articles", [])
                    total = data.get("totalResults", 0)

                    if articles:
                        formatted = [
                            {"title": a.get("title"), "source": a.get("source", {}).get("name"),
                             "url": a.get("url"), "publishedAt": a.get("publishedAt")}
                            for a in articles
                        ]
                        verdict = "TRUE" if total >= 3 else "PARTIAL"
                        return HeadlineObservation(
                            headline=headline,
                            verdict=verdict,
                            confidence="HIGH" if total >= 5 else "MEDIUM",
                            source_engine="newsapi",
                            articles=formatted,
                            total_results=total,
                            done=False,
                            reward=self._reward(verdict),
                        )
            except Exception:
                pass

        # 2. Try Groq
        if groq_key and groq_key not in ("your_groq_key_here", ""):
            try:
                return self._groq_classify(headline, groq_key)
            except Exception:
                pass

        # 3. Fallback
        return HeadlineObservation(
            headline=headline,
            verdict="UNVERIFIABLE",
            confidence="LOW",
            source_engine="heuristic",
            explanation="Could not verify with available APIs.",
            done=False,
            reward=self._reward("UNVERIFIABLE"),
        )

    def _groq_classify(self, headline: str, groq_key: str) -> HeadlineObservation:
        prompt = (
            'You are a fact-checking AI. Analyze this headline and respond ONLY with a '
            'JSON object with keys: verdict (TRUE/FALSE/PARTIAL/UNVERIFIABLE), '
            'confidence (HIGH/MEDIUM/LOW), explanation (1-2 sentences), key_claim (string).\n\n'
            f'Headline: "{headline}"\n\nJSON only, no markdown:'
        )
        with httpx.Client(timeout=20) as client:
            r = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={"model": "llama3-8b-8192",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 300, "temperature": 0},
            )
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            text = re.sub(r"```json|```", "", text).strip()
            parsed = json.loads(text)

        verdict = parsed.get("verdict", "UNVERIFIABLE")
        return HeadlineObservation(
            headline=headline,
            verdict=verdict,
            confidence=parsed.get("confidence", "LOW"),
            explanation=parsed.get("explanation", ""),
            key_claim=parsed.get("key_claim", ""),
            source_engine="groq",
            done=False,
            reward=self._reward(verdict),
        )

    @staticmethod
    def _reward(verdict: Optional[str]) -> float:
        return {"TRUE": 1.0, "FALSE": 0.0, "PARTIAL": 0.5, "UNVERIFIABLE": 0.2}.get(verdict or "", 0.0)
