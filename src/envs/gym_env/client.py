# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gym Environment HTTP Client.

This module provides the client for connecting to a Gym Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import GymAction, GymObservation, GymState


class GymEnv(HTTPEnvClient[GymAction, GymObservation]):
    """
    HTTP client for Gym Environment.

    This client connects to a GymEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = GymEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.observation_shape)
        >>>
        >>> # Take an action
        >>> result = client.step(GymAction(action=1))
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = GymEnv.from_docker_image("gym-env:latest")
        >>> result = client.reset()
        >>> result = client.step(GymAction(action=0))
    """

    def _step_payload(self, action: GymAction) -> Dict[str, Any]:
        """
        Convert GymAction to JSON payload for step request.

        Args:
            action: GymAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action": action.action,
            "env_id": action.env_id,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GymObservation]:
        """
        Parse server response into StepResult[GymObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with GymObservation.
        """
        obs_data = payload.get("observation", {})

        observation = GymObservation(
            observation=obs_data.get("observation", []),
            observation_shape=obs_data.get("observation_shape", []),
            observation_space_type=obs_data.get("observation_space_type", ""),
            action_space_type=obs_data.get("action_space_type", ""),
            action_space_n=obs_data.get("action_space_n"),
            action_space_shape=obs_data.get("action_space_shape"),
            action_space_low=obs_data.get("action_space_low"),
            action_space_high=obs_data.get("action_space_high"),
                        done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GymState:
        """
        Parse server response into GymState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            GymState object with environment state information.
        """
        return GymState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            env_id=payload.get("env_id", "unknown"),
            render_mode=payload.get("render_mode"),
            max_episode_steps=payload.get("max_episode_steps"),
        )