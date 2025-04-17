from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
import tensorflow as tf
import keras
from typing import TYPE_CHECKING
from ..utils import get_specs_from
import json

if TYPE_CHECKING:
    from tf_agents.environments.tf_py_environment import TFPyEnvironment


with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

LEARNING_RATE = config["learning_rate"]
GAMMA = config["gamma"]


class PPOAgentFactory:
    def __init__(
        self, train_env: "TFPyEnvironment", learning_rate=LEARNING_RATE, gamma=GAMMA
    ):
        walls_preprocessing = keras.Sequential(
            [
                keras.layers.Flatten(),
                keras.layers.Dense(24, activation="relu", name="walls_preprocessing"),
            ],
            name="walls_preprocessing",
        )
        position_preprocessing = keras.Sequential(
            [
                keras.layers.BatchNormalization(input_shape=(2,)),
                keras.layers.Dense(
                    24, activation="relu", name="position_preprocessing"
                ),
            ],
            name="position_preprocessing",
        )
        angle_preprocessing = keras.Sequential(
            [
                keras.layers.Reshape((1,), input_shape=()),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(24, activation="relu", name="angle_preprocessing"),
            ],
            name="angle_preprocessing",
        )

        preprocessing_combiner = keras.layers.Concatenate()

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            preprocessing_layers={
                "walls": walls_preprocessing,
                "delver_position": position_preprocessing,
                "goal_position": position_preprocessing,
                "delver_angle": angle_preprocessing,
            },
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(24,),
        )

        value_net = value_network.ValueNetwork(
            train_env.observation_spec(),
            preprocessing_layers={
                "walls": walls_preprocessing,
                "delver_position": position_preprocessing,
                "goal_position": position_preprocessing,
                "delver_angle": angle_preprocessing,
            },
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(24,),
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        time_step_spec, action_spec = get_specs_from(train_env)
        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            actor_net=actor_net,
            value_net=value_net,
            optimizer=optimizer,
            normalize_observations=True,
            normalize_rewards=True,
            discount_factor=gamma,
            train_step_counter=tf.Variable(0),
        )

        self.agent.initialize()

    def get_agent(self):
        return self.agent
