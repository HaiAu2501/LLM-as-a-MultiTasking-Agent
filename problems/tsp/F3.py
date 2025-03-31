import torch
from aco import PheromoneStrategy

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation.
    """
    
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
        # Apply evaporation
        pheromone = pheromone * decay
        
        # Deposit new pheromones
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            cost = costs[i]
            # Add pheromone to edges in the path (both directions for symmetric TSP)
            pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost
            
        return pheromone