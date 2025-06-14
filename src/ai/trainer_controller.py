from tf_agents.environments import TFPyEnvironment
from ai.environments.ai_delver_environment import AIDelverEnvironment
from ai.agents import PPOAgentFactory
from .policies import ContinuityRandomPolicy
from .replay_buffers import RingReplayBuffer
import json
from tf_agents.environments import parallel_py_environment
import tensorflow as tf
from tf_agents.system import multiprocessing as mp
from typing import TYPE_CHECKING, cast, Callable, Any

if TYPE_CHECKING:
    from .environments import DelverObservation

    class TimeStep:
        observation: "DelverObservation"
        reward: list[Any]
        is_last: Callable[[], Any]


with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

LEARNING_RATE = config["learning_rate"]
GAMMA = config["gamma"]
NUM_ITERATIONS = config["num_iterations"]
INITIAL_COLLECT_STEPS = config["initial_collect_steps"]
REPLAY_BUFFER_BATCH_SIZE = config["replay_buffer_batch_size"]
REPLAY_BUFFER_CAPACITY = config["replay_buffer_capacity"]
LOG_INTERVAL = config["log_interval"]
INITIAL_POLICY_NAME = config["initial_policy"]
ENV_BATCH_SIZE = config["env_batch_size"]


class TrainerController:
    def __init__(self):
        mp.enable_interactive_mode()

        self._setup_env_and_agent()
        self._initial_collect()

    def _setup_env_and_agent(self):
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: AIDelverEnvironment() for _ in range(ENV_BATCH_SIZE)],
            start_serially=False,
        )
        self.train_env = TFPyEnvironment(py_env)

        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=LEARNING_RATE, gamma=GAMMA
        ).get_agent()

        env_time_step_spec = self.train_env.time_step_spec()
        if env_time_step_spec is None:
            raise ValueError("The environment must return a time step spec.")
        env_action_spec = self.train_env.action_spec()
        if env_action_spec is None:
            raise ValueError("The environment must return an action spec.")

        self.replay_buffer = RingReplayBuffer(
            capacity=REPLAY_BUFFER_CAPACITY,
            observation_spec=env_time_step_spec.observation,
            action_spec=env_action_spec,
        )

        self.train_fn = self.agent.train

    def _initial_collect(self):
        print("Collecting initial replay buffer...")
        for _ in range(INITIAL_COLLECT_STEPS):
            done = False
            while not done:
                done = self.collect_step(self._initial_policy)
        print("Initial replay buffer collected.")

    def train(self):
        print(f"Training for {NUM_ITERATIONS} iterations...")
        for iteration in range(NUM_ITERATIONS):
            done = False

            while not done:
                done = self.collect_step(self.agent.collect_policy)

            experience = self.replay_buffer.sample(REPLAY_BUFFER_BATCH_SIZE)
            loss_info = self.train_fn(experience)

            print(f"Iteration {iteration}: Loss = {loss_info.loss.numpy()}")

            # if iteration % LOG_INTERVAL == 0:
            #     logging.info(f"Iteration {iteration}: Loss = {loss_info.loss.numpy()}")

    def collect_step(self, policy):
        time_step = cast("TimeStep", self.train_env.current_time_step())
        action_step = policy.action(time_step)
        next_time_step = cast("TimeStep", self.train_env.step(action_step.action))

        obs = time_step.observation
        next_obs = next_time_step.observation
        reward = next_time_step.reward
        done = next_time_step.is_last()

        for i in range(ENV_BATCH_SIZE):
            single_obs = tf.nest.map_structure(lambda x: x[i], obs)
            single_next_obs = tf.nest.map_structure(lambda x: x[i], next_obs)
            single_action = {
                "move": action_step.action["move"][i].numpy(),
                "move_angle_cos": action_step.action["move_angle_cos"][i].numpy(),
                "move_angle_sin": action_step.action["move_angle_sin"][i].numpy(),
            }
            single_reward = reward[i].numpy()
            single_done = done[i].numpy()

            self.replay_buffer.add(
                single_obs, single_action, single_reward, single_next_obs, single_done
            )

        return tf.reduce_all(done).numpy()

    def reset(self):
        self._setup_env_and_agent()

    @property
    def _initial_policy(self):
        policy_factories = {
            "continuity": lambda: ContinuityRandomPolicy(
                self.train_env.time_step_spec(),
                self.train_env.action_spec(),
                self.train_env,
            )
        }
        return policy_factories[INITIAL_POLICY_NAME]()
