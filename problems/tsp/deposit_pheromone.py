import torch

def deposit_pheromone(pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    n_ants = paths.shape[1]
    for i in range(n_ants):
        path = paths[:, i]
        cost = costs[i]
        pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
        pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost
    return pheromone