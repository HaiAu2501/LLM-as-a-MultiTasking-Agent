import os
import time
import shutil
from src.node import Node
from src.mcts import MCTS
from src.operators import Operators
from omegaconf import DictConfig
from typing import Dict, List, Any

class OperatorStats:
    """Lớp theo dõi và tính toán hiệu quả của từng toán tử trong MCTS."""
    
    def __init__(self):
        self.operators = {}  # Từ điển lưu thống kê theo từng toán tử
    
    def add_node(self, node):
        """Thêm một node vào thống kê."""
        if node.creator_operator is None:
            return  # Không theo dõi node gốc hoặc không có toán tử tạo ra
        
        op_name = node.creator_operator
        if op_name not in self.operators:
            self.operators[op_name] = {
                "count": 0,
                "total_improvement": 0.0,
                "total_score": 0.0,
                "success_count": 0,  # Số lượng node cải thiện > 0
                "best_improvement": 0.0,
                "best_score": 0.0,
                "best_node": None
            }
        
        stats = self.operators[op_name]
        stats["count"] += 1
        stats["total_improvement"] += node.improvement
        stats["total_score"] += node.score
        
        if node.improvement > 0:
            stats["success_count"] += 1
        
        if node.improvement > stats["best_improvement"]:
            stats["best_improvement"] = node.improvement
            stats["best_node"] = node
        
        if node.score > stats["best_score"]:
            stats["best_score"] = node.score
            if stats["best_node"] is None or node.score > stats["best_node"].score:
                stats["best_node"] = node
    
    def print_stats(self):
        """In thống kê hiệu quả của từng toán tử."""
        print("\n" + "="*70)
        print("OPERATOR EFFECTIVENESS STATISTICS")
        print("="*70)
        
        if not self.operators:
            print("No operator statistics available yet.")
            return
        
        # Định dạng bảng
        format_str = "{:<10} {:<10} {:<15} {:<15} {:<15} {:<15}"
        print(format_str.format("Operator", "Count", "Avg Improvement", "Success Rate", "Best Improvement", "Avg Score"))
        print("-"*70)
        
        for op_name, stats in self.operators.items():
            count = stats["count"]
            if count == 0:
                continue
                
            avg_improvement = stats["total_improvement"] / count
            success_rate = (stats["success_count"] / count) * 100 if count > 0 else 0
            best_improvement = stats["best_improvement"]
            avg_score = stats["total_score"] / count
            
            print(format_str.format(
                op_name,
                count,
                f"{avg_improvement:.2f}%",
                f"{success_rate:.2f}%",
                f"{best_improvement:.2f}%",
                f"{avg_score:.4f}"
            ))
        
        print("="*70)
        
        # In thêm thông tin chi tiết về node tốt nhất cho mỗi toán tử
        print("\nBEST NODE DETAILS FOR EACH OPERATOR:")
        for op_name, stats in self.operators.items():
            if stats["best_node"] is not None:
                best_node = stats["best_node"]
                print(f"\n{op_name} - Best Node:")
                print(f"  Score: {best_node.score:.4f}")
                print(f"  Improvement: {best_node.improvement:.2f}%")
                print(f"  Depth: {best_node.depth}")
        
        print("="*70)

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
        self.iterations_per_strategy = cfg.mcts.iterations_per_function  # Keep parameter name for backward compatibility
        self.total_iterations = self.iterations_per_strategy * len(self.problem_config.functions)
        self.max_depth = cfg.mcts.max_depth
        
        # Make sure results directory exists
        os.makedirs(cfg.paths.results_dir, exist_ok=True)
        
        # Dictionary to store MCTS instances for each strategy
        self.mcts_instances = {}
        
        # Store best implementations
        self.best_implementations = {}
        
        # Track iteration history for analysis
        self.iteration_history = []
        
        # Initialize operator statistics tracking
        self.operator_stats = OperatorStats()
        
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
        
        # Initialize MCTS for each strategy
        self._initialize_mcts_instances()
    
    def _initialize_mcts_instances(self):
        """
        Initialize MCTS instances for each strategy.
        """
        print(f"\n{'='*20} Initializing MCTS instances for all strategies {'='*20}\n")
        
        for strategy in self.problem_config.functions:
            strategy_id = strategy.id
            strategy_name = strategy.name
            strategy_path = strategy.path
            base_class = strategy.base_class
            
            # Use strategy ID as prompt key
            prompt_key = strategy_id
            
            print(f"\n{'='*10} Initializing {strategy_name} (ID: {strategy_id}) {'='*10}\n")
            print(f"Base class: {base_class}")
            
            # Read the initial implementation
            try:
                with open(strategy_path, 'r') as f:
                    initial_code = f.read()
            except FileNotFoundError:
                print(f"Warning: Strategy file {strategy_path} not found. Using default implementation.")
                initial_code = f"""import torch
from aco import {base_class}

class {strategy_name}({base_class}):
    def __init__(self):
        pass
        
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Default implementation
        pass
"""
            
            # Create a new MCTS for the strategy
            mcts = MCTS(
                function_name=strategy_name,  # Using function_name for backward compatibility
                function_id=strategy_id,
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
            
            self.mcts_instances[strategy_id] = mcts
            print(f"Initialized MCTS for {strategy_name} (ID: {strategy_id})")
            print(f"Initial implementation score: {mcts.root.score:.4f}, improvement: {mcts.root.improvement:.2f}%")
    
    def get_optimization_status(self):
        """
        Get the current status of optimization for all strategies.
        
        Returns:
            str: A string containing information about the current state of all strategies.
        """
        status = []
        
        for strategy in self.problem_config.functions:
            strategy_id = strategy.id
            strategy_name = strategy.name
            base_class = strategy.base_class
            
            mcts = self.mcts_instances[strategy_id]
            best_node = mcts.get_best_node()
            
            iterations_used = sum(1 for item in self.iteration_history if item["strategy_id"] == strategy_id)
            
            status.append(f"Strategy: {strategy_name} (ID: {strategy_id})")
            status.append(f"- Base class: {base_class}")
            status.append(f"- Iterations used: {iterations_used}/{self.iterations_per_strategy}")
            status.append(f"- Best improvement so far: {best_node.improvement:.2f}%")
            status.append(f"- Best score so far: {best_node.score:.4f}")
            status.append(f"- Depth of best node: {best_node.depth}")
            status.append(f"- Total nodes in tree: {self._count_nodes(mcts.root)}")
            
            # Include information about recently explored nodes
            if len(self.iteration_history) > 0 and self.iteration_history[-1]["strategy_id"] == strategy_id:
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
        status.append("Strategy Relationships:")
        status.append("- F1 (HeuristicImpl): Converts distances into attractiveness values")
        status.append("- F2 (ProbabilityImpl): Calculates transition probabilities")
        status.append("- F3 (PheromoneImpl): Updates pheromone trails")
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
    
    def get_next_strategy_to_optimize(self):
        """
        Ask the LLM which strategy to optimize next based on the current status.
        
        Returns:
            tuple: (strategy_id, rationale, suggestions)
        """
        status = self.get_optimization_status()
        
        system_prompt = """You are an AI specializing in guiding optimization processes for algorithm development.
Based on the current status of multiple strategy optimizations, decide which strategy should be optimized in the next iteration.
Consider factors like:
- Current improvement levels and performance scores
- Iterations used so far for each strategy (balance usage)
- Potential for further improvement
- Interdependencies between strategies
- Recent exploration results

Your decision should include which strategy to optimize next (F1, F2, or F3), your rationale, and specific suggestions for how to improve that strategy."""
        
        user_prompt = f"""Current optimization status:

{status}

Based on this information, which strategy should be optimized next?

Strategy descriptions:
- F1 (HeuristicImpl): Converts distances into attractiveness values
- F2 (ProbabilityImpl): Calculates probabilities for selecting the next city
- F3 (PheromoneImpl): Updates pheromone levels based on paths

Decide which strategy (F1, F2, or F3) should be optimized in the next iteration and provide specific suggestions for its improvement."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        strategy_id, rationale, suggestions = self.client.get_optimization_decision(messages)
        
        print(f"\nLLM Decision: Optimize {strategy_id} next")
        print(f"Rationale: {rationale}\n")
        print(f"Suggestions: {suggestions}\n")
        
        return strategy_id, rationale, suggestions
    
    def run_optimization_iteration(self, strategy_id, suggestions):
        """
        Run one iteration of optimization for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy to optimize.
            suggestions: Suggestions from the LLM for optimizing the strategy.
            
        Returns:
            dict: Results of the iteration.
        """
        # Get the corresponding MCTS instance
        mcts = self.mcts_instances[strategy_id]
        
        # Get strategy info
        strategy_name = None
        base_class = None
        for strat in self.problem_config.functions:
            if strat.id == strategy_id:
                strategy_name = strat.name
                base_class = strat.base_class
                break
        
        # 1. Selection
        node = mcts.select_node()
        print(f"Selected node at depth {node.depth}")
        
        # Create a custom operator that incorporates the LLM's suggestions if provided
        if suggestions:
            def llm_guided_operator(node, tree, client, prompts, prompt_key):
                system_prompt = prompts[prompt_key]
                user_prompt = f"""Implement an improved version of the {strategy_name} class based on these specific suggestions:

{suggestions}

Current implementation:
```python
{node.function_code}
```

Create a new implementation that incorporates these suggestions while ensuring it inherits from {base_class} and follows the required interface."""
                
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
            # Thêm node vào thống kê
            self.operator_stats.add_node(child)
            results[f"child_{len(results)}"] = {
                "score": child.score,
                "improvement": child.improvement,
                "depth": child.depth,
                "operator": child.creator_operator
            }
            print(f"Child node score: {child.score:.4f}, improvement: {child.improvement:.2f}%, operator: {child.creator_operator}")
        
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
        Run the Hierarchical MCTS with LLM-guided strategy selection.
        
        Returns:
            dict: The best implementations for each strategy.
        """
        print(f"\n{'='*20} Optimizing strategies for {self.problem_config.name} {'='*20}\n")
        
        # Track iterations performed for each strategy
        strategy_iterations = {strat.id: 0 for strat in self.problem_config.functions}
        
        for i in range(self.total_iterations):
            print(f"\n{'='*20} Iteration {i+1}/{self.total_iterations} {'='*20}\n")
            
            # Ask LLM which strategy to optimize next
            strategy_id, rationale, suggestions = self.get_next_strategy_to_optimize()
            
            # Check if we've reached the max iterations for this strategy
            if strategy_iterations[strategy_id] >= self.iterations_per_strategy:
                print(f"Maximum iterations reached for {strategy_id}. Selecting another strategy...")
                
                # Find strategies that haven't reached their max iterations
                available_strategies = [s.id for s in self.problem_config.functions 
                                      if strategy_iterations[s.id] < self.iterations_per_strategy]
                
                if not available_strategies:
                    print("All strategies have reached their maximum iterations. Ending optimization.")
                    break
                
                # Select another strategy
                strategy_id = available_strategies[0]
                print(f"Selected {strategy_id} instead.")
            
            # Get strategy info
            strategy_name = None
            for strat in self.problem_config.functions:
                if strat.id == strategy_id:
                    strategy_name = strat.name
                    break
            
            print(f"Optimizing {strategy_name} (ID: {strategy_id})")
            
            # Update iteration count for this strategy
            strategy_iterations[strategy_id] += 1
            print(f"Iteration {strategy_iterations[strategy_id]}/{self.iterations_per_strategy} for this strategy")
            
            # Run one iteration of optimization
            iteration_results = self.run_optimization_iteration(strategy_id, suggestions)
            
            # Record the iteration
            self.iteration_history.append({
                "iteration": i+1,
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "rationale": rationale,
                "suggestions": suggestions,
                "results": iteration_results
            })
            
            # In thống kê sau mỗi 5 lần lặp
            if sum(strategy_iterations.values()) % 5 == 0:
                self.operator_stats.print_stats()
        
        # After all iterations, save the best implementations
        for strategy in self.problem_config.functions:
            strategy_id = strategy.id
            strategy_name = strategy.name
            strategy_path = strategy.path
            
            # Get the best implementation
            mcts = self.mcts_instances[strategy_id]
            best_node = mcts.get_best_node()
            best_implementation = best_node.function_code
            
            self.best_implementations[strategy_id] = best_implementation
            
            # Save the best implementation
            with open(strategy_path, 'w') as f:
                f.write(best_implementation)
            
            # Also save to results directory
            result_path = os.path.join(self.cfg.paths.results_dir, f"best_{strategy_id}_{self.active_problem}.py")
            with open(result_path, 'w') as f:
                f.write(best_implementation)
            
            print(f"\n{'='*20} Saved best {strategy_name} implementation {'='*20}")
            print(f"Saved to {strategy_path} and {result_path}")
            print(f"Final improvement: {best_node.improvement:.2f}%")
        
        # Print final statistics
        print("\n\nFINAL OPERATOR EFFECTIVENESS STATISTICS")
        self.operator_stats.print_stats()
        
        return self.best_implementations