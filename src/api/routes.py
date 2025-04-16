from fastapi import APIRouter
from ai.trainer_controller import TrainerController

router = APIRouter()


@router.post("/train")
def train_agent():
    TrainerController().train()
    return {"message": "Training step complete."}
