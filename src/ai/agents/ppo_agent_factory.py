from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import (
    actor_distribution_network,
    value_network,
    normal_projection_network,
)
import tensorflow as tf
import keras
from typing import TYPE_CHECKING, cast
from ..utils import get_specs_from
import json

if TYPE_CHECKING:
    from tf_agents.environments.tf_py_environment import TFPyEnvironment


with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

LEARNING_RATE = config["learning_rate"]
GAMMA = config["gamma"]
ENTROPY_REGULARIZATION = config["entropy_regularization"]


class PPOAgentFactory:
    def __init__(
        self, train_env: "TFPyEnvironment", learning_rate=LEARNING_RATE, gamma=GAMMA
    ):
        walls_spec = train_env.observation_spec()["walls"]
        walls_shape = walls_spec.shape

        walls_preprocessing = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0),
                tf.keras.layers.Reshape((*walls_shape, 1)),
                tf.keras.layers.Conv2D(16, 3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        position_preprocessing = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(input_shape=(2,)),
                tf.keras.layers.Dense(
                    64, activation="relu", name="position_preprocessing"
                ),
            ],
            name="position_preprocessing",
        )
        preprocessing_layers = {
            "walls": walls_preprocessing,
            "delver_position": position_preprocessing,
            "goal_position": position_preprocessing,
        }

        preprocessing_combiner = tf.keras.layers.Concatenate()

        time_step_spec, action_spec = get_specs_from(train_env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(128, 64),
        )

        value_net = value_network.ValueNetwork(
            train_env.observation_spec(),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(128, 64),
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
            entropy_regularization=ENTROPY_REGULARIZATION,
            # initial_adaptive_kl_beta=0.3,  # Add this line
            # use_gae=True,  # Enable Generalized Advantage Estimation
            # num_epochs=10,  # More policy updates per batch
        )

        self.agent.initialize()

    def get_agent(self):
        return self.agent
