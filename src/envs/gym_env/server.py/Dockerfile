# Dockerfile for Gym Environment
# This image provides OpenAI Gym environments via HTTP

# Configurable base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install Gym-specific dependencies
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    numpy>=1.24.0

# Optional: Install additional Gym environments
# Uncomment as needed:
# RUN pip install --no-cache-dir gymnasium[box2d]  # For LunarLander, BipedalWalker
# RUN pip install --no-cache-dir gymnasium[mujoco]  # For MuJoCo envs
# RUN pip install --no-cache-dir gymnasium[atari]   # For Atari (alternative to ALE)

# Copy OpenEnv core
COPY src/core/ /app/src/core/

# Copy Gym environment code
COPY src/envs/gym_env/ /app/src/envs/gym_env/

# Copy README for web interface documentation
COPY src/envs/gym_env/README.md /app/README.md

# Gym-specific environment variables (can be overridden at runtime)
ENV GYM_ENV_ID=CartPole-v1
ENV GYM_RENDER_MODE=
ENV GYM_MAX_EPISODE_STEPS=

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.gym_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]