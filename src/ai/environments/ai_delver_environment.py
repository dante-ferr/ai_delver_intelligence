from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast
import tensorflow as tf
from tf_agents.typing.types import NestedArraySpec
import requests

SIMULATION_URL = "http://host.docker.internal:8000/simulation/"


class AIDelverEnvironment(PyEnvironment):
    def __init__(self):
        self.walls_grid = self._get_walls_grid()
        self._action_spec = {
            "move": array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=3,
                name="move",
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
            "delver_angle": array_spec.ArraySpec(
                shape=(), dtype=np.float32, name="delver_angle"
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

        move = int(action["move"])
        action_dict = {"move": move}
        print(self._observation["delver_angle"])

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
        delver_angle = requests.get(SIMULATION_URL + "delver_angle").json()
        # print(f"Delver position: {delver_x}, {delver_y}")
        # print(f"Goal position: {goal_x}, {goal_y}")

        if isinstance(delver_angle, list):
            delver_angle = delver_angle[0]

        # IMPORTANT: the following tensors are flattened for now, but in the future it might be a good idea to batch them
        observation = {
            "walls": tf.convert_to_tensor(walls_layer, dtype=tf.int32),
            "delver_position": np.array([delver_x, delver_y], dtype=np.float32),
            "goal_position": np.array([goal_x, goal_y], dtype=np.float32),
            "delver_angle": np.array(delver_angle, dtype=np.float32),
        }
        return observation

    @property
    def episode_ended(self):
        return self._episode_ended
