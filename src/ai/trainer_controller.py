import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from ai.environments.ai_delver_environment import AIDelverEnvironment
from ai.agents import PPOAgentFactory
import json
from .utils import get_specs_from

with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)


class TrainerController:
    def __init__(self, config_path="src/ai/utils/config.json"):
        self._load_config(config_path)
        self._setup_env_and_agent()

    def _load_config(self, path):
        with open(path, "r") as f:
            config = json.load(f)
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.epsilon_greedy = config["epsilon_greedy"]
        self.num_iterations = config["num_iterations"]
        self.initial_collect_steps = config["initial_collect_steps"]
        self.collect_steps_per_iteration = config["collect_steps_per_iteration"]
        self.batch_size = config["batch_size"]
        self.replay_buffer_capacity = config["replay_buffer_capacity"]
        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["checkpoint_interval"]

    def _setup_env_and_agent(self):
        self.train_env = tf_py_environment.TFPyEnvironment(AIDelverEnvironment())
        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=self.learning_rate, gamma=self.gamma
        ).get_agent()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_capacity,
        )

        time_step_spec, observation_spec = get_specs_from(self.train_env)
        self.random_policy = random_tf_policy.RandomTFPolicy(
            time_step_spec, observation_spec
        )

        for _ in range(self.initial_collect_steps):
            self.collect_step(self.random_policy)

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
        ).prefetch(3)
        self.iterator = iter(dataset)

        self.train_fn = self.agent.train  # swap with common.function if needed

    def collect_step(self, policy):
        time_step = self.train_env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

    def train(self):
        for iteration in range(self.num_iterations):
            for _ in range(self.collect_steps_per_iteration):
                self.collect_step(self.agent.collect_policy)

            experience, _ = next(self.iterator)
            loss_info = self.train_fn(experience)

            if iteration % self.log_interval == 0:
                print(f"Iteration {iteration}: Loss = {loss_info.loss.numpy()}")

    def reset(self):
        self._setup_env_and_agent()
