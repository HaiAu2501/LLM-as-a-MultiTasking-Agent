import torch
from torch.distributions import Categorical
import numpy as np
import importlib.util

class ACO_TSP:
    def __init__(
        self, 
        distances: np.ndarray,
        n_ants: int = 50, 
        n_iterations: int = 100,
        alpha: float = 1.0, 
        beta: float = 2.0, 
        decay: float = 0.9, 
        device: str = 'cpu'
    ) -> None:
        self.device = device
        self.distances = torch.tensor(distances, device=device)
        
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        
        self.pheromone = torch.ones_like(self.distances, device=device)
        self.heuristic_matrix = ACO_TSP.heuristic(self.distances)
        
        self.best_cost = float('inf')
    
    @staticmethod
    def heuristic(distances: torch.Tensor) -> torch.Tensor:
        mask = (distances == 0)
        distances[mask] = 1e-10
        return 1.0 / distances
    
    @staticmethod
    def calculate_probabilities(pheromone_values: torch.Tensor, heuristic_values: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        return (pheromone_values ** alpha) * (heuristic_values ** beta)
    
    def select_next_cities(self, probabilities: torch.Tensor) -> torch.Tensor:
        distribution = Categorical(probabilities)
        return distribution.sample()
    
    @staticmethod
    def deposit_pheromone(pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            cost = costs[i]
            pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost
        return pheromone
    
    def gen_path_costs(self, paths: torch.Tensor) -> torch.Tensor:
        u = paths.T
        v = torch.roll(u, shifts=1, dims=1)
        return torch.sum(self.distances[u, v], dim=1)
    
    def construct_solutions(self) -> torch.Tensor:
        n_cities = self.n_cities
        n_ants = self.n_ants
        
        start = torch.randint(low=0, high=n_cities, size=(n_ants,), device=self.device)
        mask = torch.ones(size=(n_ants, n_cities), device=self.device)
        mask[torch.arange(n_ants, device=self.device), start] = 0
        
        paths_list = [start]
        current_cities = start
        
        for _ in range(n_cities - 1):
            pheromone_values = self.pheromone[current_cities]
            heuristic_values = self.heuristic_matrix[current_cities]
            
            probs = ACO_TSP.calculate_probabilities(
                pheromone_values=pheromone_values,
                heuristic_values=heuristic_values,
                alpha=self.alpha,
                beta=self.beta
            )
            
            probs = probs * mask
            
            row_sums = probs.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums
            
            next_cities = self.select_next_cities(probs)
            paths_list.append(next_cities)
            
            current_cities = next_cities
            mask[torch.arange(n_ants, device=self.device), next_cities] = 0
        return torch.stack(paths_list)
    
    def update_pheromone(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        self.pheromone *= self.decay
        self.pheromone = ACO_TSP.deposit_pheromone(
            pheromone=self.pheromone,
            paths=paths,
            costs=costs,
        )
    
    def find_best_cost(self, costs: torch.Tensor) -> float:
        best_cost, _ = costs.min(dim=0)
        return best_cost.item()
    
    @torch.no_grad()
    def run(self) -> float:
        for _ in range(self.n_iterations):
            paths = self.construct_solutions()
            costs = self.gen_path_costs(paths)
            best_cost = self.find_best_cost(costs)
            if best_cost < self.best_cost:
                self.best_cost = best_cost
            self.update_pheromone(paths, costs)
        return self.best_cost

if importlib.util.find_spec('heuristic') is not None:
    from heuristic import heuristic
    ACO_TSP.heuristic = heuristic

if importlib.util.find_spec('calculate_probabilities') is not None:
    from calculate_probabilities import calculate_probabilities
    ACO_TSP.calculate_probabilities = calculate_probabilities

if importlib.util.find_spec('deposit_pheromone') is not None:
    from deposit_pheromone import deposit_pheromone
    ACO_TSP.deposit_pheromone = deposit_pheromone
