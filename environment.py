import uuid, os, json, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from models import HeadlineAction, HeadlineObservation, HeadlineState

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "9368616979524dd9868030b6d0f3f8e4")

class FakeNewsEnvironment:
    def __init__(self):
        self._state = HeadlineState(episode_id=str(uuid.uuid4()))

    def reset(self) -> HeadlineObservation:
        self._state = HeadlineState(episode_id=str(uuid.uuid4()))
        return HeadlineObservation(
            done=False, reward=0.0,
            headline="",
            metadata={"episode_id": self._state.episode_id, "message": "Ready. Submit a headline."}
        )

    def step(self, action: HeadlineAction) -> HeadlineObservation:
        self._state.step_count += 1
        self._state.last_headline = action.headline
        try:
            obs = self._newsapi(action.headline)
            if obs.verdict is None and action.groq_api_key:
                obs = self._groq(action.headline, action.groq_api_key)
            if obs.verdict is None:
                obs.verdict = "UNVERIFIABLE"
                obs.confidence = "Low"
            rewards = {"TRUE": 1.0, "PARTIAL": 0.5, "FALSE": 0.0, "UNVERIFIABLE": 0.3}
            obs.reward = rewards.get(obs.verdict, 0.0)
            obs.done = False
            obs.metadata["step"] = self._state.step_count
            obs.metadata["episode_id"] = self._state.episode_id
            self._state.last_verdict = obs.verdict
            return obs
        except Exception as e:
            return HeadlineObservation(done=False, reward=0.0,
                headline=action.headline, verdict="UNVERIFIABLE",
                explanation=f"Error: {str(e)}", metadata={"error": str(e)})

    def _newsapi(self, headline: str) -> HeadlineObservation:
        url = (f"https://newsapi.org/v2/everything"
               f"?q={headline[:100]}&pageSize=5&sortBy=relevancy&apiKey={NEWS_API_KEY}")
        with httpx.Client(timeout=10) as c:
            data = c.get(url).json()
        arts = data.get("articles", [])
        total = data.get("totalResults", 0)
        if not arts:
            return HeadlineObservation(headline=headline, source_engine="newsapi", total_results=0)
        verdict = "TRUE" if len(arts) >= 3 else "PARTIAL"
        confidence = "High" if len(arts) >= 3 else "Medium"
        return HeadlineObservation(
            headline=headline, verdict=verdict, confidence=confidence,
            source_engine="newsapi", total_results=total,
            articles=[{"title": a.get("title"), "source": a.get("source", {}).get("name"),
                        "url": a.get("url"), "published_at": a.get("publishedAt")} for a in arts]
        )

    def _groq(self, headline: str, key: str) -> HeadlineObservation:
        prompt = (f'Analyze: "{headline}"\n'
                  'Reply ONLY with JSON: {"verdict":"TRUE"|"FALSE"|"PARTIAL"|"UNVERIFIABLE",'
                  '"confidence":"High"|"Medium"|"Low","explanation":"...","key_claim":"..."}')
        with httpx.Client(timeout=20) as c:
            r = c.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "max_tokens": 300,
                      "messages": [{"role": "system", "content": "Fact-checker. JSON only."},
                                   {"role": "user", "content": prompt}]})
            d = r.json()
        text = re.sub(r"```json|```", "", d["choices"][0]["message"]["content"]).strip()
        p = json.loads(text)
        return HeadlineObservation(
            headline=headline, verdict=p.get("verdict"), confidence=p.get("confidence"),
            explanation=p.get("explanation"), key_claim=p.get("key_claim"),
            source_engine="groq", total_results=0)

    @property
    def state(self): return self._state
