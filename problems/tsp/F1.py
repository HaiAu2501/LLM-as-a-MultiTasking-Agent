import torch
from aco import HeuristicStrategy

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values.
    """
    
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Avoid division by zero
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10
        
        # Basic inverse distance heuristic
        return 1.0 / distances_safe