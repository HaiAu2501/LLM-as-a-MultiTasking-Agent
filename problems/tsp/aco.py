import torch
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class HeuristicStrategy(ABC):
    """Strategy interface for computing heuristic values from distances"""
    
    @abstractmethod
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        pass


class ProbabilityStrategy(ABC):
    """Strategy interface for computing selection probabilities"""
    
    @abstractmethod
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor, 
               alpha: float, beta: float) -> torch.Tensor:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            
        Returns:
            Tensor with probability values
        """
        pass


class PheromoneStrategy(ABC):
    """Strategy interface for pheromone deposition and evaporation"""
    
    @abstractmethod
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor, decay: float) -> torch.Tensor:
        """
        Update pheromone levels based on ant paths and solution costs.
        
        Args:
            pheromone: Tensor of shape (n, n) with current pheromone levels
            paths: Tensor of shape (n_cities, n_ants) with paths taken by ants
            costs: Tensor of shape (n_ants,) with path costs
            decay: Pheromone decay factor
            
        Returns:
            Updated pheromone tensor
        """
        pass


class AntColonyOptimization:
    """
    Modular implementation of Ant Colony Optimization for TSP using strategy pattern.
    """
    
    def __init__(
        self,
        distances: np.ndarray,
        n_ants: int = 50,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        decay: float = 0.9,
        device: str = 'cpu',
        seed: int = 123,
        heuristic_strategy: Optional[HeuristicStrategy] = None,
        probability_strategy: Optional[ProbabilityStrategy] = None,
        pheromone_strategy: Optional[PheromoneStrategy] = None
    ) -> None:
        """
        Initialize the ACO solver with configurable strategies.
        
        Args:
            distances: Distance matrix between cities
            n_ants: Number of ants
            n_iterations: Number of iterations
            alpha: Importance of pheromone
            beta: Importance of heuristic
            decay: Pheromone decay rate
            device: Computation device ('cpu' or 'cuda')
            seed: Random seed
            heuristic_strategy: Strategy for computing heuristic values
            probability_strategy: Strategy for computing probabilities
            pheromone_strategy: Strategy for updating pheromones
        """
        self.device = device
        self.distances = torch.tensor(distances, device=device)
        
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.seed = seed
        
        # Initialize strategies - will be loaded from F1.py, F2.py, F3.py
        self.heuristic_strategy = heuristic_strategy
        self.probability_strategy = probability_strategy
        self.pheromone_strategy = pheromone_strategy
        
        # Check that all strategies are provided
        if not all([heuristic_strategy, probability_strategy, pheromone_strategy]):
            raise ValueError("All strategies must be provided")
        
        # Initialize pheromone matrix
        self.pheromone = torch.ones_like(self.distances, device=device)
        
        # Compute heuristic matrix once
        self.heuristic_matrix = self.heuristic_strategy.compute(self.distances)
        
        self.best_path = None
        self.best_cost = float('inf')
    
    def construct_solutions(self) -> torch.Tensor:
        """
        Construct solutions (paths) for all ants.
        
        Returns:
            Tensor of shape (n_cities, n_ants) with paths
        """
        n_cities = self.n_cities
        n_ants = self.n_ants
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Choose random starting cities
        start = torch.randint(low=0, high=n_cities, size=(n_ants,), device=self.device)
        
        # Track which cities have been visited
        mask = torch.ones(size=(n_ants, n_cities), device=self.device)
        mask[torch.arange(n_ants, device=self.device), start] = 0
        
        # Initialize paths with starting cities
        paths_list = [start]
        current_cities = start
        
        # Construct the rest of the paths
        for _ in range(n_cities - 1):
            # Get pheromone and heuristic values for current cities
            pheromone_values = self.pheromone[current_cities]
            heuristic_values = self.heuristic_matrix[current_cities]
            
            # Calculate probabilities
            probs = self.probability_strategy.compute(
                pheromone=pheromone_values,
                heuristic=heuristic_values,
                alpha=self.alpha,
                beta=self.beta
            )
            
            # Apply mask to consider only unvisited cities
            probs = probs * mask
            
            # Normalize probabilities
            row_sums = probs.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            probs = probs / row_sums
            
            # Select next cities using categorical distribution
            distribution = torch.distributions.Categorical(probs=probs)
            next_cities = distribution.sample()
            
            paths_list.append(next_cities)
            
            # Update current cities and mask
            current_cities = next_cities
            mask[torch.arange(n_ants, device=self.device), next_cities] = 0
            
        # Stack path components to get complete paths
        return torch.stack(paths_list)
    
    def calculate_path_costs(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cost (total distance) of each path.
        
        Args:
            paths: Tensor of shape (n_cities, n_ants) with paths
            
        Returns:
            Tensor of shape (n_ants,) with path costs
        """
        # Transpose to have shape (n_ants, n_cities)
        u = paths.T
        
        # Get next cities (with wrapping to form a cycle)
        v = torch.roll(u, shifts=1, dims=1)
        
        # Sum distances along paths
        return torch.sum(self.distances[u, v], dim=1)
    
    def update_pheromones(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Update pheromone levels based on ant paths.
        
        Args:
            paths: Tensor of shape (n_cities, n_ants) with paths
            costs: Tensor of shape (n_ants,) with path costs
        """
        self.pheromone = self.pheromone_strategy.update(
            pheromone=self.pheromone,
            paths=paths,
            costs=costs,
            decay=self.decay
        )
    
    def update_best_solution(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Update the best solution found so far.
        
        Args:
            paths: Tensor of shape (n_cities, n_ants) with paths
            costs: Tensor of shape (n_ants,) with path costs
        """
        min_cost, min_idx = costs.min(dim=0)
        if min_cost < self.best_cost:
            self.best_cost = min_cost.item()
            self.best_path = paths[:, min_idx].cpu().numpy()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the ACO algorithm.
        
        Returns:
            Dictionary with best path, best cost, and convergence history
        """
        cost_history = []
        
        for iteration in range(self.n_iterations):
            # Update seed for each iteration but keep it deterministic
            self.seed = self.seed + 1
            
            # Construct solutions for all ants
            paths = self.construct_solutions()
            
            # Calculate path costs
            costs = self.calculate_path_costs(paths)
            
            # Update best solution
            self.update_best_solution(paths, costs)
            
            # Update pheromones
            self.update_pheromones(paths, costs)
            
            # Record best cost in this iteration
            cost_history.append(self.best_cost)
        
        return {
            'best_path': self.best_path,
            'best_cost': self.best_cost,
            'cost_history': cost_history
        }