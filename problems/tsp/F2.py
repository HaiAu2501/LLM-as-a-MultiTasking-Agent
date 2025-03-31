import torch
from aco import ProbabilityStrategy

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """
    
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
        # Standard ACO probability calculation
        return (pheromone ** alpha) * (heuristic ** beta)