import numpy as np
import os

def generate_tsp_datasets(sizes=[50], n_instances=5, seed=42):
    """
    Generate TSP datasets with random 2D points.
    
    Args:
        sizes: List of problem sizes (number of cities)
        n_instances: Number of instances per size
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_dir = os.path.join(current_dir, "tsp_datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    
    for size in sizes:
        # Generate n_instances TSP problems with 'size' cities in 2D space
        dataset = np.random.rand(n_instances, size, 2)
        
        filename = os.path.join(dataset_dir, f"{size}_{n_instances}.npy")
        np.save(filename, dataset)
        print(f"Generated {n_instances} instances of {size}-city TSP problems: {filename}")

if __name__ == "__main__":
    generate_tsp_datasets()