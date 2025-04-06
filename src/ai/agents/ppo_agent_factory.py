from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
import tensorflow as tf
import keras
from ..environments import AIDelverEnvironment
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
        custom_model = keras.Sequential(
            [
                keras.layers.Dense(24, activation="relu"),
                keras.layers.Dense(24, activation="relu"),
            ]
        )

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            preprocessing_layers=custom_model,
            fc_layer_params=None,  # No additional FC layers since we use custom layers
        )

        value_net = value_network.ValueNetwork(
            train_env.observation_spec(),
            preprocessing_layers=custom_model,
            fc_layer_params=None,
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
