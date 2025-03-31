GUIDE = """
- Innovate with creative, nature-inspired solutions while strictly following the strategy pattern design.
- Feel free to explore unpopular approaches and briefly justify your design choices.
- You can add additional parameters, methods, and state variables to your strategy implementation.
- Consider adding adaptive mechanisms or leveraging problem-specific knowledge.
"""

F1 = f"""
You are an expert in designing heuristics for nature-inspired optimization algorithms, 
specifically for the Ant Colony Optimization (ACO) applied to the Traveling Salesman Problem (TSP).

Here is the HeuristicStrategy abstract base class you need to implement:

```python
from abc import ABC, abstractmethod
import torch

class HeuristicStrategy(ABC):
    \"\"\"Strategy interface for computing heuristic values from distances\"\"\"
    
    @abstractmethod
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        \"\"\"
        pass
```

Your task is to implement a class named `HeuristicImpl` that inherits from this `HeuristicStrategy` class. Your implementation should follow this pattern:

```python
import torch
from aco import HeuristicStrategy

class HeuristicImpl(HeuristicStrategy):
    # You can add __init__ method with additional parameters if needed
    
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Your implementation here
        pass
```

The function should:
- Take a distance matrix (torch.Tensor of shape (n, n)) as input
- Return a heuristic matrix (torch.Tensor of shape (n, n)) representing the attractiveness values

Suggestions:
{GUIDE}
"""

F2 = f"""
You are an expert in probability modeling and decision-making for swarm intelligence algorithms.

Here is the ProbabilityStrategy abstract base class you need to implement:

```python
from abc import ABC, abstractmethod
import torch

class ProbabilityStrategy(ABC):
    \"\"\"Strategy interface for computing selection probabilities\"\"\"
    
    @abstractmethod
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor, 
               alpha: float, beta: float) -> torch.Tensor:
        \"\"\"
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            
        Returns:
            Tensor with probability values
        \"\"\"
        pass
```

Your assignment is to implement a class named `ProbabilityImpl` that inherits from this `ProbabilityStrategy` class. Your implementation should follow this pattern:

```python
import torch
from aco import ProbabilityStrategy

class ProbabilityImpl(ProbabilityStrategy):
    # You can add __init__ method with additional parameters if needed
    
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor, 
               alpha: float, beta: float) -> torch.Tensor:
        # Your implementation here
        pass
```

The function should:
- Combine pheromone and heuristic information to create attractiveness values
- Use alpha and beta parameters to control the relative importance of each
- Return a tensor containing the computed attractiveness values

Suggestions:
{GUIDE}
"""

F3 = f"""
You are an expert in pheromone update strategies within Ant Colony Optimization (ACO) algorithms.

Here is the PheromoneStrategy abstract base class you need to implement:

```python
from abc import ABC, abstractmethod
import torch

class PheromoneStrategy(ABC):
    \"\"\"Strategy interface for pheromone deposition and evaporation\"\"\"
    
    @abstractmethod
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor, decay: float) -> torch.Tensor:
        \"\"\"
        Update pheromone levels based on ant paths and solution costs.
        
        Args:
            pheromone: Tensor of shape (n, n) with current pheromone levels
            paths: Tensor of shape (n_cities, n_ants) with paths taken by ants
            costs: Tensor of shape (n_ants,) with path costs
            decay: Pheromone decay factor
            
        Returns:
            Updated pheromone tensor
        \"\"\"
        pass
```

Your task is to implement a class named `PheromoneImpl` that inherits from this `PheromoneStrategy` class. Your implementation should follow this pattern:

```python
import torch
from aco import PheromoneStrategy

class PheromoneImpl(PheromoneStrategy):
    # You can add __init__ method with additional parameters if needed
    
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor, decay: float) -> torch.Tensor:
        # Your implementation here
        pass
```

The function should:
- Handle pheromone evaporation using the decay parameter
- Update pheromone levels based on the paths taken by ants and their costs
- Return the updated pheromone matrix

Suggestions:
{GUIDE}
"""