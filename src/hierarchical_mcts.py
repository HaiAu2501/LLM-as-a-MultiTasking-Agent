import os
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from node import Node
from mcts import MCTS
from operators import Operators

class HierarchicalMCTS:
    """
    Implementation of the Hierarchical Monte Carlo Tree Search for function optimization.
    """
    
    def __init__(self, client, prompts, iterations_per_function=10, max_depth=3):
        """
        Initialize the Hierarchical MCTS.
        
        Args:
            client: The LLM client.
            prompts: The prompts dictionary.
            iterations_per_function (int, optional): The number of iterations per function. Defaults to 10.
            max_depth (int, optional): The maximum depth of each tree. Defaults to 3.
        """
        self.client = client
        self.prompts = prompts
        self.iterations_per_function = iterations_per_function
        self.max_depth = max_depth
        self.best_implementations = {}
        
        # Define the functions to optimize in order
        self.functions = [
            {
                "name": "heuristic",
                "path": os.path.join("problems", "tsp", "heuristic.py"),
                "prompt_key": "F1"
            },
            {
                "name": "calculate_probabilities",
                "path": os.path.join("problems", "tsp", "calculate_probabilities.py"),
                "prompt_key": "F2"
            },
            {
                "name": "deposit_pheromone",
                "path": os.path.join("problems", "tsp", "deposit_pheromone.py"),
                "prompt_key": "F3"
            }
        ]
        
        # Define operators
        self.all_operators = {
            "SR": Operators.self_reflection,
            "EF": Operators.ensemble_fusion,
            "DE": Operators.diversity_exploration,
            "MR": Operators.memory_based_reuse,
            "GR": Operators.guided_randomness
        }
        
        # Only DE operator for root node
        self.root_operators = {
            "DE": Operators.diversity_exploration
        }
    
    def run(self):
        """
        Run the Hierarchical MCTS for all functions in sequence.
        
        Returns:
            dict: The best implementations for each function.
        """
        for function in self.functions:
            print(f"\n{'='*20} Optimizing {function['name']} {'='*20}\n")
            
            # Read the initial implementation
            with open(function["path"], 'r') as f:
                initial_code = f.read()
            
            # Create a new MCTS for the function
            mcts = MCTS(
                function["name"],
                initial_code,
                self.client,
                self.prompts,
                max_depth=self.max_depth
            )
            
            # Run MCTS for the function
            best_implementation = self._run_mcts_for_function(mcts, function["name"])
            self.best_implementations[function["name"]] = best_implementation
            
            # Save the best implementation to the file
            with open(function["path"], 'w') as f:
                f.write(best_implementation)
            
            print(f"\n{'='*20} Saved best {function['name']} implementation {'='*20}\n")
        
        return self.best_implementations
    
    def _run_mcts_for_function(self, mcts, function_name):
        """
        Run MCTS for a specific function.
        
        Args:
            mcts (MCTS): The MCTS instance.
            function_name (str): The name of the function.
            
        Returns:
            str: The best implementation of the function.
        """
        # Evaluate the root node first
        mcts.simulate(mcts.root)
        mcts.backpropagate(mcts.root, mcts.root.score)
        
        print(f"Initial implementation score: {mcts.root.score:.4f}, improvement: {mcts.root.improvement:.2f}%")
        
        # Run iterations
        for i in range(self.iterations_per_function):
            print(f"\nIteration {i+1}/{self.iterations_per_function}")
            
            # 1. Selection
            node = mcts.select_node()
            print(f"Selected node at depth {node.depth}")
            
            # 2. Expansion
            if node == mcts.root:
                # Apply DE operator 5 times at the root
                print("Expanding root node with DE operator")
                children = mcts.expand(node, self.root_operators, num_expansions=5)
            else:
                # Apply all operators at other nodes
                print(f"Expanding node at depth {node.depth} with all operators")
                children = mcts.expand(node, self.all_operators)
            
            # 3 & 4. Simulation and Backpropagation
            for child in children:
                score = mcts.simulate(child)
                mcts.backpropagate(child, score)
                print(f"Child node score: {child.score:.4f}, improvement: {child.improvement:.2f}%")
            
            # Print the best score so far
            best_node = mcts.get_best_node()
            print(f"Best implementation so far: improvement={best_node.improvement:.2f}%, score={best_node.score:.4f}, depth={best_node.depth}")
        
        # Return the best implementation
        best_node = mcts.get_best_node()
        print(f"Final best implementation: improvement={best_node.improvement:.2f}%, score={best_node.score:.4f}, depth={best_node.depth}")
        return best_node.function_code