"""FastAPI inference service for GPU-accelerated deployment."""

from __future__ import annotations

from fastapi import FastAPI
import torch

app = FastAPI()


@app.post("/predict")
def predict(data: list[float]):
    tensor = torch.tensor(data).float()
    # Dummy model: identity
    output = tensor.tolist()
    return {"prediction": output}
