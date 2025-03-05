import numpy as np
import os

def generate_tsp_datasets(sizes=[50], n_instances=5, seed=42):
    """
    Generate TSP datasets with random 2D points and their corresponding distance matrices.
    
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
        for instance in range(n_instances):
            # Generate random cities in 2D space
            cities = np.random.rand(size, 2)
            
            # Calculate the Euclidean distance matrix
            distance_matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    if i != j:
                        distance_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))
            
            filename = os.path.join(dataset_dir, f"{size}_{seed}_{instance+1}.npy")
            np.save(filename, distance_matrix)
            print(f"Generated {size}-city TSP instance {instance+1} with seed {seed}: {filename}")

if __name__ == "__main__":
    generate_tsp_datasets()