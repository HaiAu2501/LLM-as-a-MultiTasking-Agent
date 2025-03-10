import torch

def calculate_probabilities(pheromone_values: torch.Tensor, heuristic_values: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return (pheromone_values ** alpha) * (heuristic_values ** beta)