from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, Any
import tensorflow as tf
from tf_agents.typing.types import NestedArraySpec
from queue import Queue, Empty
from ._simulation_socket_worker import SimulationSocketWorker
import logging
import math

SIMULATION_WS_URL = "ws://host.docker.internal:8000/ws/simulation"


class AIDelverEnvironment(PyEnvironment):
    def __init__(self):
        self.last_action: dict[str, Any] = {
            "move": False,
            "move_angle_sin": 0,
            "move_angle_cos": 0,
        }
        self.episodes = 0

        self._init_socket_worker()
        self.walls_grid = self._get_walls_grid()

        self._init_specs()

        self._episode_ended = False

    def _init_socket_worker(self):
        self._action_queue = Queue()
        self._result_queue = Queue()

        self.worker = SimulationSocketWorker(
            SIMULATION_WS_URL,
            self._action_queue,
            self._result_queue,
        )
        self.worker.start()

    def _init_specs(self):
        self._action_spec = {
            "move": array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="move"
            ),
            "move_angle_cos": array_spec.BoundedArraySpec(
                (), dtype=np.float32, minimum=-1.0, maximum=1.0, name="move_angle_cos"
            ),
            "move_angle_sin": array_spec.BoundedArraySpec(
                (), dtype=np.float32, minimum=-1.0, maximum=1.0, name="move_angle_sin"
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

    def _get_walls_grid(self):
        self._action_queue.put({"type": "get_walls"})
        try:
            result = self._result_queue.get(timeout=1.0)
        except Empty:
            raise RuntimeError("Timeout getting walls data")
        return np.array(result["walls"])

    def _reset(self):
        self.episodes += 1
        self._action_queue.put({"type": "start_new_simulation"})
        try:
            self._result_queue.get(timeout=1.0)  # Just wait for ack
        except Empty:
            raise RuntimeError("Timeout starting new simulation")
        self._episode_ended = False
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        move_angle_rad = math.atan2(action["move_angle_sin"], action["move_angle_cos"])
        action_dict = {
            "move": False if round(float(action["move"])) == 0 else True,
            "move_angle": float(math.degrees(move_angle_rad)),
        }
        print(
            f"Episode: {self.episodes}, Move: {action_dict['move']}, Move angle: {action_dict['move_angle']}, Delver pos: {self._observation['delver_position']}"
        )

        # Send delver actions to the simulation
        self._action_queue.put({"type": "step", "payload": action_dict})

        try:
            result = self._result_queue.get(timeout=1.0)
        except Empty:
            raise RuntimeError("Timed out waiting for simulation response")

        reward = result["reward"]
        self._episode_ended = result["ended"]
        elapsed_time = result["elapsed_time"]

        return self._create_time_step(reward)

    def _create_time_step(self, reward):
        reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
        discount_tensor = tf.convert_to_tensor(1.0, dtype=tf.float32)

        if self._episode_ended:
            print("Episode ended!")
            return ts.termination(self._observation, reward_tensor)
        return ts.transition(self._observation, reward_tensor, discount_tensor)

    # tf_agents.typing.types.NestedArraySpec is a union that includes tf_agents.types.ArraySpec. So I suppose it's safe to cast it to bounded arrays, because they extend ArraySpec.
    def action_spec(self):
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        return cast(NestedArraySpec, self._observation_spec)

    @property
    def _observation(self):
        walls_layer = self.walls_grid.astype(np.float32)

        self._action_queue.put({"type": "get_delver_position"})
        delver_x, delver_y = self._result_queue.get(timeout=1.0)["position"]

        self._action_queue.put({"type": "get_goal_position"})
        goal_x, goal_y = self._result_queue.get(timeout=1.0)["position"]

        observation = {
            "walls": tf.convert_to_tensor(walls_layer, dtype=tf.int32),
            "delver_position": np.array([delver_x, delver_y], dtype=np.float32),
            "goal_position": np.array([goal_x, goal_y], dtype=np.float32),
        }
        return observation

    @property
    def episode_ended(self):
        return self._episode_ended
