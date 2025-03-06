import torch

def heuristic(distances: torch.Tensor) -> torch.Tensor:
    mask = (distances == 0)
    distances[mask] = 1e-10
    return 1.0 / distances