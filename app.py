"""
server/app.py — OpenEnv-compliant FastAPI server.
Uses create_fastapi_app which auto-wires /reset, /step, /state endpoints.
"""
from openenv.core.env_server import create_fastapi_app
from models import HeadlineAction, HeadlineObservation
from server.fake_news_environment import FakeNewsEnvironment

env = FakeNewsEnvironment()
app = create_fastapi_app(env, HeadlineAction, HeadlineObservation)
