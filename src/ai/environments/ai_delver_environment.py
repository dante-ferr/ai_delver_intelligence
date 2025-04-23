from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, Any
import tensorflow as tf
from tf_agents.typing.types import NestedArraySpec
import requests

SIMULATION_URL = "http://host.docker.internal:8000/simulation/"


class AIDelverEnvironment(PyEnvironment):
    def __init__(self):
        self.last_action: dict[str, Any] = {
            "move": False,
            "move_angle": 0,
        }

        self.walls_grid = self._get_walls_grid()
        self._action_spec = {
            "move": array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="move"
            ),
            "move_angle": array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.float32,
                minimum=0,
                maximum=360.0,
                name="move_angle",
            ),
        }

        self.observation_shape = (3,)
        self._observation_spec = {
            "walls": array_spec.ArraySpec(
                shape=self.walls_grid.shape, dtype=np.int32, name="walls"
            ),
            "delver_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="delver_position"
            ),
            "goal_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="goal_position"
            ),
        }

        self._episode_ended = False

    def _get_walls_grid(self):
        grid_data = requests.get(SIMULATION_URL + "walls").json()
        return np.array(grid_data)

    # tf_agents.typing.types.NestedArraySpec is a union that includes tf_agents.types.ArraySpec. So I suppose it's safe to cast it to bounded arrays, because they extend ArraySpec.
    def action_spec(self):
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        return cast(NestedArraySpec, self._observation_spec)

    def _reset(self):
        requests.post(SIMULATION_URL + "start_new_simulation")
        self._episode_ended = False
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        action_dict = {
            "move": False if round(float(action["move"])) == 0 else True,
            "move_angle": float(action["move_angle"]),
        }
        print(
            f"Move: {action_dict['move']}, Move angle: {action_dict['move_angle']}, Delver pos: {self._observation['delver_position']}"
        )

        try:
            response = requests.post(
                SIMULATION_URL + "step",
                json=action_dict,
                timeout=1.0,
            )
            response.raise_for_status()
            reward, self._episode_ended, elapsed_time = response.json()
        except Exception as e:
            print(f"Error in step: {str(e)}")
            raise

        return self._create_time_step(reward)

    def _create_time_step(self, reward):
        reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
        discount_tensor = tf.convert_to_tensor(1.0, dtype=tf.float32)

        if self._episode_ended:
            print("Episode ended!")
            return ts.termination(self._observation, reward_tensor)
        return ts.transition(self._observation, reward_tensor, discount_tensor)

    @property
    def _observation(self):
        walls_layer = self.walls_grid.astype(np.float32)
        delver_x, delver_y = requests.get(SIMULATION_URL + "delver_position").json()
        goal_x, goal_y = requests.get(SIMULATION_URL + "goal_position").json()

        observation = {
            "walls": tf.convert_to_tensor(walls_layer, dtype=tf.int32),
            "delver_position": np.array([delver_x, delver_y], dtype=np.float32),
            "goal_position": np.array([goal_x, goal_y], dtype=np.float32),
        }
        return observation

    @property
    def episode_ended(self):
        return self._episode_ended
