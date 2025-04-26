from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from ai.environments.ai_delver_environment import AIDelverEnvironment
from ai.agents import PPOAgentFactory
from .policies import ContinuityRandomPolicy
from .replay_buffers import RingReplayBuffer
import json

with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

LEARNING_RATE = config["learning_rate"]
GAMMA = config["gamma"]
NUM_ITERATIONS = config["num_iterations"]
INITIAL_COLLECT_STEPS = config["initial_collect_steps"]
BATCH_SIZE = config["batch_size"]
REPLAY_BUFFER_CAPACITY = config["replay_buffer_capacity"]
LOG_INTERVAL = config["log_interval"]
INITIAL_POLICY_NAME = config["initial_policy"]


class TrainerController:
    def __init__(self):
        self._setup_env_and_agent()

    def _setup_env_and_agent(self):
        self.train_env = tf_py_environment.TFPyEnvironment(AIDelverEnvironment())
        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=LEARNING_RATE, gamma=GAMMA
        ).get_agent()

        self.replay_buffer = RingReplayBuffer(
            capacity=REPLAY_BUFFER_CAPACITY,
            observation_spec=self.train_env.time_step_spec().observation,
            action_spec=self.train_env.action_spec(),
        )

        print("Collecting initial replay buffer...")
        for _ in range(INITIAL_COLLECT_STEPS):
            done = False
            while not done:
                done = self.collect_step(self._initial_policy)
        print("Initial replay buffer collected.")

        self.train_fn = self.agent.train

    def train(self):
        print(f"Training for {NUM_ITERATIONS} iterations...")
        for iteration in range(NUM_ITERATIONS):
            done = False

            while not done:
                done = self.collect_step(self.agent.collect_policy)

            experience = self.replay_buffer.sample(BATCH_SIZE)
            loss_info = self.train_fn(experience)

            print(f"Iteration {iteration}: Loss = {loss_info.loss.numpy()}")

            # if iteration % LOG_INTERVAL == 0:
            #     logging.info(f"Iteration {iteration}: Loss = {loss_info.loss.numpy()}")

    def collect_step(self, policy):
        time_step = self.train_env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)

        obs = self.train_env.pyenv.envs[0].observation
        next_obs = self.train_env.pyenv.envs[0].observation

        action = {
            "move": action_step.action["move"].numpy(),
            "move_angle_cos": action_step.action["move_angle_cos"].numpy(),
            "move_angle_sin": action_step.action["move_angle_sin"].numpy(),
        }

        reward = next_time_step.reward.numpy()
        done = self.train_env.pyenv.envs[0].episode_ended

        self.replay_buffer.add(obs, action, reward, next_obs, done)

        return done

    def reset(self):
        self._setup_env_and_agent()

    @property
    def _initial_policy(self):
        policy_factories = {
            "continuity": lambda: ContinuityRandomPolicy(
                self.train_env.time_step_spec(),
                self.train_env.action_spec(),
                self.train_env.pyenv.envs[0],
            )
        }
        return policy_factories[INITIAL_POLICY_NAME]()
