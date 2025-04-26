from fastapi import APIRouter
from ._test_simulation_step import test_simulation_step
from ai import TrainerController

router = APIRouter()


@router.post("/train")
def train_agent():
    # test_simulation_step()

    TrainerController().train()
    return {"message": "Training step complete."}
