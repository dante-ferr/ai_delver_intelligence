from ai.trainer_controller import TrainerController
from queue import Queue, Empty
from ai.environments._simulation_socket_worker import SimulationSocketWorker

frame = 0
SIMULATION_WS_URL = "ws://host.docker.internal:8000/ws/simulation"


def test_simulation_step():
    global frame
    _action_queue = Queue()
    _result_queue = Queue()

    worker = SimulationSocketWorker(
        SIMULATION_WS_URL,
        _action_queue,
        _result_queue,
    )
    worker.start()

    action_dict = {
        "move": True,
        "move_angle": 180,
    }

    while True:
        _action_queue.put({"type": "step", "payload": action_dict})
        try:
            result = _result_queue.get(timeout=1.0)
        except Empty:
            raise RuntimeError("Timed out waiting for simulation response")

        print(frame)
        frame += 1
