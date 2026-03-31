import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from server.environment import FakeNewsEnvironment

env = FakeNewsEnvironment()
app = FastAPI(title="FakeNewsAI", version="1.0.0")

# ── Pydantic schemas for FastAPI ───────────────────────────
class ActionPayload(BaseModel):
    headline: str
    groq_api_key: Optional[str] = None

# ── OpenEnv required endpoints ─────────────────────────────

@app.post("/reset")
async def reset():
    obs = env.reset()
    return JSONResponse(status_code=200, content={
        "observation": obs.to_dict(),
        "info": {"message": "Environment reset successfully"}
    })

@app.post("/step")
async def step(payload: ActionPayload):
    from models import HeadlineAction
    action = HeadlineAction(headline=payload.headline, groq_api_key=payload.groq_api_key)
    obs = env.step(action)
    return JSONResponse(status_code=200, content={
        "observation": obs.to_dict(),
        "reward": obs.reward,
        "done": obs.done,
        "info": {"step": env.state.step_count}
    })

@app.get("/state")
async def state():
    return JSONResponse(status_code=200, content=env.state.to_dict())

@app.get("/health")
async def health():
    return JSONResponse(status_code=200, content={"status": "ok", "env": "FakeNewsAI"})

# ── Frontend GET endpoint (for browser) ───────────────────
@app.get("/api/verify")
async def verify(headline: str, groqKey: str = ""):
    from models import HeadlineAction
    action = HeadlineAction(headline=headline, groq_api_key=groqKey or None)
    obs = env.step(action)
    return JSONResponse(status_code=200, content={
        "source": obs.source_engine,
        "verdict": obs.verdict,
        "confidence": obs.confidence,
        "explanation": obs.explanation,
        "keyClaim": obs.key_claim,
        "totalResults": obs.total_results,
        "articles": obs.articles,
    })

# ── Serve frontend ─────────────────────────────────────────
@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({"status": "FakeNewsAI running", "docs": "/docs"})
