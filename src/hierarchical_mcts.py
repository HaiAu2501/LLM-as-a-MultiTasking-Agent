import os
import time
import shutil
from src.node import Node
from src.mcts import MCTS
from src.operators import Operators
from omegaconf import DictConfig

class HierarchicalMCTS:   
    def __init__(self, client, prompts, cfg: DictConfig):
        """
        Initialize the Hierarchical MCTS.
        
        Args:
            client: The LLM client.
            prompts: The prompts dictionary.
            cfg: The configuration object.
        """
        self.client = client
        self.prompts = prompts
        self.cfg = cfg
        
        # Get MCTS parameters from config
        self.iterations_per_function = cfg.mcts.iterations_per_function
        self.max_depth = cfg.mcts.max_depth
        
        # Get the active problem configuration
        self.active_problem = cfg.problem.active
        self.problem_config = getattr(cfg.problem, self.active_problem)
        
        # Make sure results directory exists
        os.makedirs(cfg.paths.results_dir, exist_ok=True)
        
        # Store best implementations
        self.best_implementations = {}
        
        # Define operators
        self.all_operators = {
            "SR": Operators.self_reflection,
            "EF": Operators.ensemble_fusion,
            "DE": Operators.diversity_exploration,
            "MR": Operators.memory_reuse,
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
        print(f"\n{'='*20} Optimizing functions for {self.problem_config.name} {'='*20}\n")
        
        # Process each function in the problem config
        for function in self.problem_config.functions:
            function_id = function.id
            function_name = function.name
            function_path = function.path
            
            # Use function ID as prompt key if not specified otherwise
            prompt_key = function_id
            
            print(f"\n{'='*20} Optimizing {function_name} (ID: {function_id}) {'='*20}\n")
            
            # Read the initial implementation
            try:
                with open(function_path, 'r') as f:
                    initial_code = f.read()
            except FileNotFoundError:
                print(f"Warning: Function file {function_path} not found. Using default implementation.")
                initial_code = f"import torch\n\ndef {function_name}():\n    pass"
            
            # Create a new MCTS for the function
            mcts = MCTS(
                function_name=function_name,
                function_id=function_id,
                initial_code=initial_code,
                client=self.client,
                prompts=self.prompts,
                prompt_key=prompt_key,
                problem_config=self.problem_config,
                max_depth=self.max_depth
            )
            
            # Run MCTS for the function
            best_implementation = self._run_mcts_for_function(mcts, function_id, function_name)
            self.best_implementations[function_id] = best_implementation
            
            # Save the best implementation
            with open(function_path, 'w') as f:
                f.write(best_implementation)
            
            # Also save to results directory
            result_path = os.path.join(self.cfg.paths.results_dir, f"best_{function_id}_{self.active_problem}.py")
            with open(result_path, 'w') as f:
                f.write(best_implementation)
            
            print(f"\n{'='*20} Saved best {function_name} implementation {'='*20}")
            print(f"Saved to {function_path} and {result_path}")
        
        return self.best_implementations
    
    def _run_mcts_for_function(self, mcts: MCTS, function_id, function_name):
        """
        Run MCTS for a specific function.
        
        Args:
            mcts (MCTS): The MCTS instance.
            function_id (str): The ID of the function (e.g., F1, F2).
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
        print(f"Final best implementation for {function_id} ({function_name}): improvement={best_node.improvement:.2f}%, score={best_node.score:.4f}, depth={best_node.depth}")
        return best_node.function_code