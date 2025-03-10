import numpy as np
import os
import multiprocessing as mp
from aco import ACO_TSP

try:
    from F1 import heuristic
    ACO_TSP.heuristic = heuristic
    from F2 import calculate_probabilities
    ACO_TSP.calculate_probabilities = calculate_probabilities
    from F3 import deposit_pheromone
    ACO_TSP.deposit_pheromone = deposit_pheromone
except ImportError:
    pass

BASELINE_COST = 6.220183403581592

def run_aco_tsp(instance_path):
    distances = np.load(instance_path)
    aco = ACO_TSP(distances)
    best_cost = aco.run()
    return best_cost

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    instance_paths = [
        os.path.join(current_dir, f'tsp_datasets/50_42_{i}.npy') 
        for i in range(1, 6)
    ]
    
    with mp.Pool(processes=min(5, mp.cpu_count())) as pool:
        results = pool.map(run_aco_tsp, instance_paths)
    
    avg_cost = sum(results) / len(results)
    
    improvement = ((BASELINE_COST - avg_cost) / BASELINE_COST)
    print(improvement)
    return improvement

if __name__ == "__main__":
    main()