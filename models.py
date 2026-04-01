"""
models.py — OpenEnv-compliant Action, Observation, and State models.
Uses Pydantic-based openenv.core.env_server.types as base classes.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class HeadlineAction(Action):
    """Action: submit a news headline to verify."""
    headline: str = Field(..., description="The news headline to fact-check")
    groq_api_key: Optional[str] = Field(None, description="Optional Groq API key override")


class HeadlineObservation(Observation):
    """Observation: the fact-check result for a submitted headline."""
    headline: str = Field("", description="The headline that was checked")
    verdict: Optional[str] = Field(None, description="TRUE / FALSE / PARTIAL / UNVERIFIABLE")
    confidence: Optional[str] = Field(None, description="HIGH / MEDIUM / LOW")
    explanation: Optional[str] = Field(None, description="Brief explanation of verdict")
    key_claim: Optional[str] = Field(None, description="The key verifiable claim extracted")
    source_engine: Optional[str] = Field(None, description="newsapi / groq / heuristic")
    articles: List[Dict[str, Any]] = Field(default_factory=list, description="Related articles")
    total_results: int = Field(0, description="Total related articles found")


class HeadlineState(State):
    """State: current episode information."""
    last_headline: str = Field("", description="The last headline submitted")
    last_verdict: Optional[str] = Field(None, description="The last verdict returned")
