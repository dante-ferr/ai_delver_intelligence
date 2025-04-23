import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.specs import tensor_spec
import random
import json
from typing import Any

with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

seed = config["seed"]
random.seed(seed)

# TODO: study this thing
# I had a good idea: if the Delver's last action was to move, the probability of changing to closer angles is higher than to further ones


class ContinuityRandomPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, env):
        policy_info_spec = {
            "dist_params": {
                "move": {
                    "loc": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                    "scale": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                },
                "move_angle": {
                    "loc": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                    "scale": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                },
            }
        }
        super().__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=(),
            info_spec=policy_info_spec,
        )
        self.env = env

    def _action(self, time_step, policy_state=(), seed=None):
        # Generate move action
        move_probs = np.array([0.05, 0.15, 0.8])
        move_value = [0.0, 1.0, float(self.env.last_action["move"])][
            self._get_action_index(move_probs)
        ]

        move_loc = tf.convert_to_tensor([move_value], dtype=tf.float32)
        move_scale = tf.convert_to_tensor([0.1], dtype=tf.float32)

        # Generate move_angle action
        move_angle_probs = np.array([0.2, 0.8])
        move_angle_value = [
            random.uniform(0, 360),
            float(self.env.last_action["move_angle"]),
        ][self._get_action_index(move_angle_probs)]

        move_angle_loc = tf.convert_to_tensor([move_angle_value], dtype=tf.float32)
        move_angle_scale = tf.convert_to_tensor([10.0], dtype=tf.float32)

        self.env.last_action = {
            "move": bool(move_value > 0.5),
            "move_angle": move_angle_value,
        }

        return policy_step.PolicyStep(
            action={"move": move_loc, "move_angle": move_angle_loc},
            state=(),
            info={
                "dist_params": {
                    "move": {"loc": move_loc, "scale": move_scale},
                    "move_angle": {"loc": move_angle_loc, "scale": move_angle_scale},
                }
            },
        )

    def _get_action_index(self, probs: list[float]):
        probs_array = np.array(probs)
        normalized_probs = probs / np.sum(probs_array)
        action = np.random.choice(len(probs), p=normalized_probs)
        return action
