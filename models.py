from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

@dataclass(kw_only=True)
class Observation:
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HeadlineAction:
    headline: str
    groq_api_key: Optional[str] = None

@dataclass(kw_only=True)
class HeadlineObservation(Observation):
    headline: str = ""
    verdict: Optional[str] = None
    confidence: Optional[str] = None
    explanation: Optional[str] = None
    key_claim: Optional[str] = None
    source_engine: Optional[str] = None
    articles: List[Dict] = field(default_factory=list)
    total_results: int = 0

    def to_dict(self):
        return {
            "done": self.done,
            "reward": self.reward,
            "metadata": self.metadata,
            "headline": self.headline,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "key_claim": self.key_claim,
            "source_engine": self.source_engine,
            "articles": self.articles,
            "total_results": self.total_results,
        }

@dataclass
class HeadlineState:
    episode_id: Optional[str] = None
    step_count: int = 0
    last_headline: str = ""
    last_verdict: Optional[str] = None

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "last_headline": self.last_headline,
            "last_verdict": self.last_verdict,
        }
