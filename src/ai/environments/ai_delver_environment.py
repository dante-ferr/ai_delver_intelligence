from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast
import tensorflow as tf
from tf_agents.typing.types import NestedArraySpec
import requests

SIMULATION_URL = "http://localhost:8000/simulation/"


class AIDelverEnvironment(PyEnvironment):
    def __init__(self):
        self.walls_grid: np.ndarray = requests.get(SIMULATION_URL + "walls").json()

        self._action_spec = {
            "move_direction": array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.float32,
                minimum=0.0,
                maximum=360.0,
                name="move_direction",
            )
        }

        self.observation_shape = (3,)
        self._observation_spec = {
            "walls": array_spec.ArraySpec(
                shape=self.walls_grid.shape, dtype=np.float32, name="walls"
            ),
            "delver_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="delver_position"
            ),
            "goal_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="goal_position"
            ),
        }

        self._episode_ended = False

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

        response = requests.post(SIMULATION_URL + "step", json=action)
        reward, self._episode_ended, elapsed_time = response.json()

        if self._episode_ended:
            print("Episode ended")
            print(f"Elapsed time: {elapsed_time}")
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward=reward, discount=1.0)

    @property
    def _observation(self):
        walls_layer = self.walls_grid.astype(np.float32)
        delver_x, delver_y = requests.get(SIMULATION_URL + "delver_position").json()
        goal_x, goal_y = requests.get(SIMULATION_URL + "goal_position").json()

        return {
            "walls": tf.convert_to_tensor(walls_layer, dtype=tf.float32),
            "delver_position": tf.convert_to_tensor(
                [delver_x, delver_y], dtype=tf.float32
            ),
            "goal_position": tf.convert_to_tensor([goal_x, goal_y], dtype=tf.float32),
        }
