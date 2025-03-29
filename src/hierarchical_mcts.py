import os
import time
import shutil
from src.node import Node
from src.mcts import MCTS
from src.operators import Operators
from omegaconf import DictConfig
from typing import Dict, List, Any

class HierarchicalMCTS:   
    def __init__(self, client, prompts, cfg: DictConfig):
        """
        Initialize the Hierarchical MCTS with parallel optimization approach.
        
        Args:
            client: The LLM client.
            prompts: The prompts dictionary.
            cfg: The configuration object.
        """
        self.client = client
        self.prompts = prompts
        self.cfg = cfg
        
        # Get the active problem configuration
        self.active_problem = cfg.problem.active
        self.problem_config = getattr(cfg.problem, self.active_problem)
        
        # Get MCTS parameters from config
        self.iterations_per_function = cfg.mcts.iterations_per_function
        self.total_iterations = self.iterations_per_function * len(self.problem_config.functions)
        self.max_depth = cfg.mcts.max_depth
        
        # Make sure results directory exists
        os.makedirs(cfg.paths.results_dir, exist_ok=True)
        
        # Dictionary to store MCTS instances for each function
        self.mcts_instances = {}
        
        # Store best implementations
        self.best_implementations = {}
        
        # Track iteration history for analysis
        self.iteration_history = []
        
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
        
        # Initialize MCTS for each function
        self._initialize_mcts_instances()
    
    def _initialize_mcts_instances(self):
        """
        Initialize MCTS instances for each function.
        """
        print(f"\n{'='*20} Initializing MCTS instances for all functions {'='*20}\n")
        
        for function in self.problem_config.functions:
            function_id = function.id
            function_name = function.name
            function_path = function.path
            
            # Use function ID as prompt key
            prompt_key = function_id
            
            print(f"\n{'='*10} Initializing {function_name} (ID: {function_id}) {'='*10}\n")
            
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
            
            # Evaluate the root node
            mcts.simulate(mcts.root)
            mcts.backpropagate(mcts.root, mcts.root.score)
            
            self.mcts_instances[function_id] = mcts
            print(f"Initialized MCTS for {function_name} (ID: {function_id})")
            print(f"Initial implementation score: {mcts.root.score:.4f}, improvement: {mcts.root.improvement:.2f}%")
    
    def get_optimization_status(self):
        """
        Get the current status of optimization for all functions.
        
        Returns:
            str: A string containing information about the current state of all functions.
        """
        status = []
        
        for function in self.problem_config.functions:
            function_id = function.id
            function_name = function.name
            
            mcts = self.mcts_instances[function_id]
            best_node = mcts.get_best_node()
            
            iterations_used = sum(1 for item in self.iteration_history if item["function_id"] == function_id)
            
            status.append(f"Function: {function_name} (ID: {function_id})")
            status.append(f"- Iterations used: {iterations_used}/{self.iterations_per_function}")
            status.append(f"- Best improvement so far: {best_node.improvement:.2f}%")
            status.append(f"- Best score so far: {best_node.score:.4f}")
            status.append(f"- Depth of best node: {best_node.depth}")
            status.append(f"- Total nodes in tree: {self._count_nodes(mcts.root)}")
            
            # Include information about recently explored nodes
            if len(self.iteration_history) > 0 and self.iteration_history[-1]["function_id"] == function_id:
                status.append(f"- Most recent exploration:")
                last_iter = self.iteration_history[-1]
                for metric, value in last_iter["results"].items():
                    status.append(f"  - {metric}: {value}")
            
            # Add code snippet of best implementation (abbreviated)
            if best_node.function_code:
                code_lines = best_node.function_code.split('\n')
                if len(code_lines) > 6:
                    abbreviated_code = '\n'.join(code_lines[:3] + ['...'] + code_lines[-3:])
                else:
                    abbreviated_code = best_node.function_code
                status.append(f"- Current best implementation (abbreviated):")
                status.append(f"```python\n{abbreviated_code}\n```")
            
            status.append("")  # Empty line for separation
        
        # Add information about relationships and dependencies
        status.append("Function Relationships:")
        status.append("- F1 (heuristic): Transforms distances into attractiveness values")
        status.append("- F2 (calculate_probabilities): Uses pheromones and heuristic values to guide ant decisions")
        status.append("- F3 (deposit_pheromone): Updates pheromone trails based on ant paths and costs")
        status.append("")
        status.append("Dependencies: F1 feeds into F2, F2 affects ant path selection, F3 updates environment for next iteration")
        
        return "\n".join(status)
    
    def _count_nodes(self, node):
        """
        Count the total number of nodes in a tree.
        
        Args:
            node: The root node of the tree.
            
        Returns:
            int: The total number of nodes in the tree.
        """
        count = 1  # Count the node itself
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def get_next_function_to_optimize(self):
        """
        Ask the LLM which function to optimize next based on the current status.
        
        Returns:
            tuple: (function_id, rationale, suggestions)
        """
        status = self.get_optimization_status()
        
        system_prompt = """You are an AI specializing in guiding optimization processes for algorithm development.
Based on the current status of multiple function optimizations, decide which function should be optimized in the next iteration.
Consider factors like:
- Current improvement levels and performance scores
- Iterations used so far for each function (balance usage)
- Potential for further improvement
- Interdependencies between functions (F1->F2->F3 workflow)
- Recent exploration results

Your decision should include which function to optimize next (F1, F2, or F3), your rationale, and specific suggestions for how to improve that function."""
        
        user_prompt = f"""Current optimization status:

{status}

Based on this information, which function should be optimized next?

Function descriptions:
- F1 (heuristic): Transforms distances into attractiveness values for paths between cities
- F2 (calculate_probabilities): Calculates probabilities for selecting the next city based on pheromone and heuristic values
- F3 (deposit_pheromone): Updates pheromone levels based on paths taken by ants and their costs

Decide which function (F1, F2, or F3) should be optimized in the next iteration and provide specific suggestions for its improvement."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        function_id, rationale, suggestions = self.client.get_optimization_decision(messages)
        
        print(f"\nLLM Decision: Optimize {function_id} next")
        print(f"Rationale: {rationale}\n")
        print(f"Suggestions: {suggestions}\n")
        
        return function_id, rationale, suggestions
    
    def run_optimization_iteration(self, function_id, suggestions):
        """
        Run one iteration of optimization for a specific function.
        
        Args:
            function_id: The ID of the function to optimize.
            suggestions: Suggestions from the LLM for optimizing the function.
            
        Returns:
            dict: Results of the iteration.
        """
        # Get the corresponding MCTS instance
        mcts = self.mcts_instances[function_id]
        
        # Get function info
        function_name = None
        for func in self.problem_config.functions:
            if func.id == function_id:
                function_name = func.name
                break
        
        # 1. Selection
        node = mcts.select_node()
        print(f"Selected node at depth {node.depth}")
        
        # Create a custom operator that incorporates the LLM's suggestions if provided
        if suggestions:
            def llm_guided_operator(node, tree, client, prompts, prompt_key):
                system_prompt = prompts[prompt_key]
                user_prompt = f"""Implement an improved version of the function based on these specific suggestions:

{suggestions}

Current implementation:
```python
{node.function_code}
```

Create a new implementation that incorporates these suggestions while ensuring it follows the required function signature."""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                code, _ = client.get_code(messages)
                return code
            
            # Combine existing operators with the LLM-guided operator
            combined_operators = self.all_operators.copy()
            combined_operators["LG"] = llm_guided_operator
            
            # 2. Expansion with combined operators
            if node == mcts.root:
                # For root node, still use DE operator 5 times
                print("Expanding root node with DE operator")
                children = mcts.expand(node, self.root_operators, num_expansions=5)
                
                # Also apply the LLM-guided operator once for the root
                print("Additionally expanding root node with LLM-guided operator")
                lg_operators = {"LG": llm_guided_operator}
                lg_children = mcts.expand(node, lg_operators, num_expansions=1)
                children.extend(lg_children)
            else:
                # Apply all operators (including LLM-guided) at other nodes
                print(f"Expanding node at depth {node.depth} with all operators (including LLM-guided)")
                children = mcts.expand(node, combined_operators)
        else:
            # 2. Regular Expansion (without LLM guidance)
            if node == mcts.root:
                # Apply DE operator 5 times at the root
                print("Expanding root node with DE operator")
                children = mcts.expand(node, self.root_operators, num_expansions=5)
            else:
                # Apply all operators at other nodes
                print(f"Expanding node at depth {node.depth} with all operators")
                children = mcts.expand(node, self.all_operators)
        
        # 3 & 4. Simulation and Backpropagation
        results = {}
        for child in children:
            score = mcts.simulate(child)
            mcts.backpropagate(child, score)
            results[f"child_{len(results)}"] = {
                "score": child.score,
                "improvement": child.improvement,
                "depth": child.depth,
            }
            print(f"Child node score: {child.score:.4f}, improvement: {child.improvement:.2f}%")
        
        # Print the best score so far
        best_node = mcts.get_best_node()
        print(f"Best implementation so far: improvement={best_node.improvement:.2f}%, score={best_node.score:.4f}, depth={best_node.depth}")
        
        # Return results for tracking
        return {
            "num_children": len(children),
            "best_score": best_node.score,
            "best_improvement": best_node.improvement,
            "results": results
        }
    
    def run(self):
        """
        Run the Hierarchical MCTS with LLM-guided function selection.
        
        Returns:
            dict: The best implementations for each function.
        """
        print(f"\n{'='*20} Optimizing functions for {self.problem_config.name} {'='*20}\n")
        
        # Track iterations performed for each function
        function_iterations = {func.id: 0 for func in self.problem_config.functions}
        
        for i in range(self.total_iterations):
            print(f"\n{'='*20} Iteration {i+1}/{self.total_iterations} {'='*20}\n")
            
            # Ask LLM which function to optimize next
            function_id, rationale, suggestions = self.get_next_function_to_optimize()
            
            # Check if we've reached the max iterations for this function
            if function_iterations[function_id] >= self.iterations_per_function:
                print(f"Maximum iterations reached for {function_id}. Selecting another function...")
                
                # Find functions that haven't reached their max iterations
                available_functions = [f.id for f in self.problem_config.functions 
                                      if function_iterations[f.id] < self.iterations_per_function]
                
                if not available_functions:
                    print("All functions have reached their maximum iterations. Ending optimization.")
                    break
                
                # Select another function
                function_id = available_functions[0]
                print(f"Selected {function_id} instead.")
            
            # Get function info
            function_name = None
            for func in self.problem_config.functions:
                if func.id == function_id:
                    function_name = func.name
                    break
            
            print(f"Optimizing {function_name} (ID: {function_id})")
            
            # Update iteration count for this function
            function_iterations[function_id] += 1
            print(f"Iteration {function_iterations[function_id]}/{self.iterations_per_function} for this function")
            
            # Run one iteration of optimization
            iteration_results = self.run_optimization_iteration(function_id, suggestions)
            
            # Record the iteration
            self.iteration_history.append({
                "iteration": i+1,
                "function_id": function_id,
                "function_name": function_name,
                "rationale": rationale,
                "suggestions": suggestions,
                "results": iteration_results
            })
        
        # After all iterations, save the best implementations
        for function in self.problem_config.functions:
            function_id = function.id
            function_name = function.name
            function_path = function.path
            
            # Get the best implementation
            mcts = self.mcts_instances[function_id]
            best_node = mcts.get_best_node()
            best_implementation = best_node.function_code
            
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
            print(f"Final improvement: {best_node.improvement:.2f}%")
        
        return self.best_implementations