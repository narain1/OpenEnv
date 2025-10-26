import uuid
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from core.env_server import Action, Environment, Observation

from ..models import GymAction, GymObservation, GymState


class GymEnvironment(Environment):
    """
    OpenAI Gym Environment wrapper for OpenEnv.

    This environment wraps any Gym environment and provides a clean
    interface for RL training via HTTP.

    Args:
        env_id: Gym environment ID (e.g., "CartPole-v1").
        render_mode: Render mode ("human", "rgb_array", None).
        max_episode_steps: Override max episode steps (optional).

    Example:
        >>> env = GymEnvironment("CartPole-v1")
        >>> obs = env.reset()
        >>> print(obs.observation_shape)  # [4]
        >>> obs = env.step(GymAction(action=1))  # Move right
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
    ):
        """Initialize Gym environment."""
        super().__init__()

        self.env_id = env_id
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Create Gym environment
        try:
            self.env = gym.make(
                env_id,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create Gym environment '{env_id}': {e}\n"
                f"Make sure the environment is registered with Gym."
            ) from e

        # Initialize state
        self._state = GymState(
            env_id=env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

        self._last_observation = None
        self._last_info = {}

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        # Reset Gym environment
        observation, info = self.env.reset()

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._last_observation = observation
        self._last_info = info

        # Get initial observation
        return self._make_observation(observation, 0.0, False, False, info)

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: GymAction containing the action to execute.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a GymAction.
        """
        if not isinstance(action, GymAction):
            raise ValueError(f"Expected GymAction, got {type(action)}")

        # Convert action based on action space type
        gym_action = self._convert_action(action.action)

        # Execute action
        observation, reward, terminated, truncated, info = self.env.step(gym_action)

        self._state.step_count += 1
        self._last_observation = observation
        self._last_info = info

        done = terminated or truncated

        # Get observation
        return self._make_observation(observation, reward, done, truncated, info)

    @property
    def state(self) -> GymState:
        """Get current environment state."""
        return self._state

    def _convert_action(self, action: Any) -> Any:
        """
        Convert action from API format to Gym format.

        Args:
            action: Action in API format (int, List[float], etc.).

        Returns:
            Action in Gym format (may be numpy array).
        """
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Discrete):
            return int(action)
        elif isinstance(action_space, gym.spaces.Box):
            return np.array(action, dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            return np.array(action, dtype=np.int64)
        elif isinstance(action_space, gym.spaces.MultiBinary):
            return np.array(action, dtype=np.int8)
        else:
            # For other action spaces, try converting to numpy array
            try:
                return np.array(action)
            except Exception:
                return action

    def _make_observation(
        self,
        observation: Any,
        reward: float,
        done: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> GymObservation:
        """
        Create a GymObservation from Gym state.

        Args:
            observation: Raw observation from Gym.
            reward: Reward from last action.
            done: Whether episode is done.
            truncated: Whether episode was truncated.
            info: Additional info from Gym.

        Returns:
            GymObservation for the agent.
        """
        # Flatten observation to list
        if isinstance(observation, np.ndarray):
            obs_flat = observation.flatten().tolist()
            obs_shape = list(observation.shape)
        else:
            # Handle scalar or other types
            obs_flat = [float(observation)]
            obs_shape = [1]

        # Get observation space info
        obs_space = self.env.observation_space
        obs_space_type = type(obs_space).__name__

        # Get action space info
        action_space = self.env.action_space
        action_space_type = type(action_space).__name__

        action_space_n = None
        action_space_shape = None
        action_space_low = None
        action_space_high = None

        if isinstance(action_space, gym.spaces.Discrete):
            action_space_n = int(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            action_space_shape = list(action_space.shape)
            action_space_low = action_space.low.flatten().tolist()
            action_space_high = action_space.high.flatten().tolist()
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            action_space_shape = list(action_space.nvec)
        elif isinstance(action_space, gym.spaces.MultiBinary):
            action_space_n = int(action_space.n)

        # Create observation
        obs = GymObservation(
            observation=obs_flat,
            observation_shape=obs_shape,
            observation_space_type=obs_space_type,
            action_space_type=action_space_type,
            action_space_n=action_space_n,
            action_space_shape=action_space_shape,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
            done=done,
            reward=reward,
            metadata={
                "env_id": self.env_id,
                "truncated": truncated,
                "info": info,
            },
        )

        return obs