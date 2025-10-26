# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Gym Environment.

This module defines the Action, Observation, and State types for OpenAI Gym environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from core.env_server import Action, Observation, State


@dataclass
class GymAction(Action):
    """
    Action for Gym environments.

    Attributes:
        action: The action to take. Type depends on action space:
                - Discrete: int
                - Box: List[float]
                - MultiDiscrete: List[int]
                - MultiBinary: List[int]
        env_id: Gym environment ID (e.g., "CartPole-v1", "MountainCar-v0").
    """
    action: Union[int, List[float], List[int]]
    env_id: str = "CartPole-v1"


@dataclass
class GymObservation(Observation):
    """
    Observation from Gym environment.

    Attributes:
        observation: The observation data as a flattened list.
        observation_shape: Original shape of observation.
        observation_space_type: Type of observation space 
                                ("Box", "Discrete", "MultiDiscrete", etc.).
        action_space_type: Type of action space.
        action_space_n: Number of actions (for Discrete) or None.
        action_space_shape: Shape of action space (for Box/MultiDiscrete) or None.
        action_space_low: Lower bounds (for Box) or None.
        action_space_high: Upper bounds (for Box) or None.
    """
    observation: List[float]
    observation_shape: List[int]
    observation_space_type: str
    action_space_type: str
    action_space_n: Optional[int] = None
    action_space_shape: Optional[List[int]] = None
    action_space_low: Optional[List[float]] = None
    action_space_high: Optional[List[float]] = None


@dataclass
class GymState(State):
    """
    State for Gym environment.

    Attributes:
        env_id: Gym environment ID.
        render_mode: Rendering mode ("human", "rgb_array", None).
        max_episode_steps: Maximum steps per episode (if applicable).
    """
    env_id: str = "CartPole-v1"
    render_mode: Optional[str] = None
    max_episode_steps: Optional[int] = None