from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, Any
from tf_agents.typing.types import NestedArraySpec
from ._simulation_socket_worker import SimulationSocketWorker
import math
from functools import cached_property
from multiprocessing import Manager
from ai_delver_runtime.simulation import Simulation
from level_holder import level_holder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import DelverObservation
    from ai_delver_runtime.simulation import DelverAction

SIMULATION_WS_URL = "ws://host.docker.internal:8000/ws/simulation"

manager = Manager()
frame_counter = manager.Value("i", 0)
frame_lock = manager.Lock()


class AIDelverEnvironment(PyEnvironment):
    def __init__(self):
        self.last_action: dict[str, Any] = {
            "move": 0.0,
            "move_angle_sin": 0.0,
            "move_angle_cos": 0.0,
        }
        self._restart_simulation()
        self.episodes = 0

        self._init_specs()

        self.episode_ended = False

    def _restart_simulation(self):
        self.simulation = Simulation(level_holder.level)

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
                shape=self.walls_grid.shape, dtype=np.float32, name="walls"
            ),
            "delver_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="delver_position"
            ),
            "goal_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="goal_position"
            ),
        }

    def _reset(self):
        self.episodes += 1
        self.episode_ended = False
        self._restart_simulation()

        return ts.restart(self.observation)

    def _count_and_print_frame(self):
        with frame_lock:
            frame_counter.value += 1
            current_frame = frame_counter.value

        print(f"Global frame count: {current_frame}")

    def _step(self, action):
        self._count_and_print_frame()

        if self.episode_ended:
            return self._reset()

        action_dict = self._get_dict_of_action(action)
        # print(
        #     f"Episode: {self.episodes}, Move: {action_dict['move']}, Move angle: {action_dict['move_angle']}, Delver pos: {self.observation['delver_position']}"
        # )

        reward, self.episode_ended, elapsed_time = self.simulation.step(action_dict)

        return self._create_time_step(reward)

    def _get_dict_of_action(self, action):
        move_angle_rad = math.atan2(action["move_angle_sin"], action["move_angle_cos"])

        return DelverAction(
            move=False if round(float(action["move"])) == 0 else True,
            move_angle=float(math.degrees(move_angle_rad)),
        )

    def _create_time_step(self, reward):
        if self.episode_ended:
            print(f"Episode {self.episodes} ended!")
            return ts.termination(self.observation, reward)
        return ts.transition(self.observation, reward, 1.0)

    # tf_agents.typing.types.NestedArraySpec is a union that includes tf_agents.types.ArraySpec. So I suppose it's safe to cast it to bounded arrays, because they extend ArraySpec.
    def action_spec(self):
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        return cast(NestedArraySpec, self._observation_spec)

    @property
    def observation(self):
        walls_layer = self.walls_grid.astype(np.float32)

        observation: "DelverObservation" = {
            "walls": np.array(walls_layer, dtype=np.float32),
            "delver_position": np.array([*self.delver_position], dtype=np.float32),
            "goal_position": np.array([*self.goal_position], dtype=np.float32),
        }
        return observation

    @cached_property
    def walls_grid(self):
        walls_grid = self.simulation.tilemap.get_layer("walls").grid
        walls_grid_presence = np.array(
            [[1 if cell is not None else 0 for cell in row] for row in walls_grid],
            dtype=np.uint8,
        )
        return walls_grid_presence

    @property
    def delver_position(self):
        return self.simulation.delver.position

    @property
    def goal_position(self):
        return self.simulation.goal.position
