import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from src.ai.environments.ai_delver_environment import AIDelverEnvironment
from src.ai.agents import PPOAgentFactory
import json
from .utils import get_specs_from

with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

LEARNING_RATE = config["learning_rate"]
GAMMA = config["gamma"]
EPSILON_GREEDY = config["epsilon_greedy"]
NUM_ITERATIONS = config["num_iterations"]
INITIAL_COLLECT_STEPS = config["initial_collect_steps"]
COLLECT_STEPS_PER_ITERATION = config["collect_steps_per_iteration"]
BATCH_SIZE = config["batch_size"]
REPLAY_BUFFER_CAPACITY = config["replay_buffer_capacity"]
LOG_INTERVAL = config["log_interval"]
CHECKPOINT_INTERVAL = config["checkpoint_interval"]


def train():
    train_env = tf_py_environment.TFPyEnvironment(AIDelverEnvironment())
    agent = PPOAgentFactory(
        train_env, learning_rate=LEARNING_RATE, gamma=GAMMA
    ).get_agent()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=REPLAY_BUFFER_CAPACITY,
    )

    time_step_spec, observation_spec = get_specs_from(train_env)
    random_policy = random_tf_policy.RandomTFPolicy(time_step_spec, observation_spec)

    for _ in range(INITIAL_COLLECT_STEPS):
        collect_step(train_env, random_policy, replay_buffer)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=BATCH_SIZE, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    # train_fn = common.function(agent.train)
    # TODO: add tensorflow's common.function
    train_fn = agent.train

    for iteration in range(NUM_ITERATIONS):
        for _ in range(COLLECT_STEPS_PER_ITERATION):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience, _ = next(iterator)
        loss_info = train_fn(experience)
        train_loss = loss_info.loss

        if iteration % LOG_INTERVAL == 0:
            print(f"Iteration {iteration}: Loss = {train_loss.numpy()}")


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


if __name__ == "__main__":
    train()
