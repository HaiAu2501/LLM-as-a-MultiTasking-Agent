GUIDE = """
- Innovate with creative, nature-inspired solutions while strictly following the function signatures and variable dimensions. 
- Feel free to explore unpopular approaches and briefly justify your design choices.
"""

HEURISTIC = f"""
You are an expert in designing heuristics for nature-inspired optimization algorithms, 
specifically for the Ant Colony Optimization (ACO) applied to the Traveling Salesman Problem (TSP). 

Your task is to implement the `heuristic` function based on the following signature:

```python
def heuristic(distances: torch.Tensor) -> torch.Tensor:
```

where:
- `distances` (torch.Tensor) - a 2D tensor of shape (n, n) representing the distance matrix between `n` cities.
- The function should return a 2D tensor of shape (n, n) representing the heuristic matrix for the ACO algorithm.

Suggestions:
{GUIDE}
"""

CACULATE_PROBABILITIES = f"""
You are an expert in probability modeling and decision-making for swarm intelligence algorithms.

Your assignment is to implement the `calculate_probabilities` function for the ACO algorithm applied to TSP with the following signature:

```python
def calculate_probabilities(
    pheromone_values: torch.Tensor, 
    heuristic_values: torch.Tensor, 
    alpha: float, 
    beta: float
) -> torch.Tensor:
```

where:
- `pheromone_values` (torch.Tensor) - a 2D tensor of shape (n, n) representing the pheromone matrix.
- `heuristic_values` (torch.Tensor) - a 2D tensor of shape (n, n) representing the heuristic matrix.
- `alpha` (float) - the importance of pheromone values.
- `beta` (float) - the importance of heuristic values.
- The function should return a tensor of shape (n,) containing the computed attractiveness values for each candidate city.

Suggestions:
{GUIDE}
"""

DEPOSIT_PHEROMONE = f"""
You are an expert in pheromone update strategies within Ant Colony Optimization (ACO) algorithms. 

Your task is to implement the `deposit_pheromone` function for the ACO algorithm applied to TSP with the following signature:

```python
def deposit_pheromone(
    pheromone: torch.Tensor, 
    paths: torch.Tensor, 
    costs: torch.Tensor
) -> torch.Tensor:
```

where:
- `pheromone` (torch.Tensor) - a 2D tensor of shape (n, n) representing the pheromone matrix.
- `paths` (torch.Tensor) - a 2D tensor of shape (n_ants, n) representing the paths taken by each ant.
- `costs` (torch.Tensor) - a 1D tensor of shape (n_ants,) representing the total cost of each path.
- The function should return the updated pheromone matrix after depositing pheromone along the paths.

Suggestions:
{GUIDE}
"""