import numpy as np
import os
import multiprocessing as mp
import torch
from aco import AntColonyOptimization

# Import strategies from files
try:
    from F1 import HeuristicImpl
    from F2 import ProbabilityImpl
    from F3 import PheromoneImpl
except ImportError as e:
    pass

def run_aco_tsp(args):
    """
    Run ACO-TSP on a single instance with a specific seed.
    
    Args:
        args: Tuple containing (instance_path, seed)
        
    Returns:
        Tuple of (instance_path, seed, best_cost)
    """
    instance_path, seed = args
    distances = np.load(instance_path)
    
    # Create ACO with strategy implementations
    aco = AntColonyOptimization(
        distances=distances,
        seed=seed,
        heuristic_strategy=HeuristicImpl(),
        probability_strategy=ProbabilityImpl(),
        pheromone_strategy=PheromoneImpl()
    )
    
    # Run optimization
    result = aco.run()
    best_cost = result['best_cost']
    
    return (instance_path, seed, best_cost)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get list of instance files (5 problem instances)
    instance_paths = [
        os.path.join(current_dir, f'tsp_datasets/50_42_{i}.npy') 
        for i in range(1, 6)
    ]
    
    # Generate some seeds for each instance
    seeds = [42, 123, 456, 789]

    # Create all combinations of instances and seeds
    tasks = [(path, seed) for path in instance_paths for seed in seeds]
    
    # Run in parallel
    with mp.Pool(processes=min(mp.cpu_count(), len(tasks))) as pool:
        results = pool.map(run_aco_tsp, tasks)
    
    # Calculate overall average cost
    all_costs = [cost for _, _, cost in results]
    avg_cost = sum(all_costs) / len(all_costs)
    
    # Chỉ in trung bình chi phí (không tính improvement)
    print(avg_cost)
    return avg_cost

if __name__ == "__main__":
    main()