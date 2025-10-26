"""
FastAPI application for the Gym Environment.

This module creates an HTTP server that exposes Gym environments
over HTTP endpoints, making them compatible with HTTPEnvClient.

Usage:
    # Development:
    uvicorn envs.gym_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.gym_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables:
    GYM_ENV_ID: Environment ID (default: "CartPole-v1")
    GYM_RENDER_MODE: Render mode (default: None)
    GYM_MAX_EPISODE_STEPS: Max steps per episode (optional)
"""

import os

from core.env_server import create_app

from ..models import GymAction, GymObservation
from .gym_env import GymEnvironment

# Get configuration from environment variables
env_id = os.getenv("GYM_ENV_ID", "CartPole-v1")
render_mode = os.getenv("GYM_RENDER_MODE")  # None, "human", "rgb_array"
max_episode_steps_str = os.getenv("GYM_MAX_EPISODE_STEPS")

# Convert max_episode_steps
max_episode_steps = None
if max_episode_steps_str:
    max_episode_steps = int(max_episode_steps_str)

# Create the environment instance
env = GymEnvironment(
    env_id=env_id,
    render_mode=render_mode,
    max_episode_steps=max_episode_steps,
)

# Create the FastAPI app
app = create_app(env, GymAction, GymObservation, env_name="gym_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)