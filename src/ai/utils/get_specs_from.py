from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tf_agents.environments.tf_py_environment import TFPyEnvironment


def get_specs_from(environment: "TFPyEnvironment"):
    time_step_spec = environment.time_step_spec()
    observation_spec = environment.observation_spec()
    if time_step_spec is None or observation_spec is None:
        raise ValueError(
            "The environment must return a valid time_step_spec and an observation_spec."
        )
    return time_step_spec, observation_spec
