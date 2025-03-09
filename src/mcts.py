import math
import random
import os
import subprocess
import numpy as np
from src.node import Node
from src.code_validator import CodeValidator

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class MCTS:
    """
    Implementation of the Monte Carlo Tree Search algorithm for function optimization.
    """
    
    def __init__(self, function_name, initial_code, client, prompts, exploration_weight=1.0, max_depth=3):
        """
        Initialize the MCTS algorithm.
        
        Args:
            function_name (str): The name of the function to optimize.
            initial_code (str): The initial implementation of the function.
            client: The LLM client for generating new function implementations.
            prompts: The prompts dictionary for guiding the LLM.
            exploration_weight (float, optional): The exploration weight for UCB. Defaults to 1.0.
            max_depth (int, optional): The maximum depth of the tree. Defaults to 3.
        """
        self.function_name = function_name
        self.root = Node(initial_code, function_name, depth=0)
        self.client = client
        self.prompts = prompts
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.best_node = self.root
        
        # Get the function prompt key
        if function_name == "heuristic":
            self.prompt_key = "F1"
        elif function_name == "calculate_probabilities":
            self.prompt_key = "F2"
        elif function_name == "deposit_pheromone":
            self.prompt_key = "F3"
        else:
            raise ValueError(f"Unknown function name: {function_name}")
    
    def select_node(self):
        """
        Select a node to expand using UCB1.
        
        Returns:
            Node: The selected node.
        """
        node = self.root
        
        # Traverse the tree to find a node to expand
        while node.children and node.depth < self.max_depth:
            # If there are unvisited children, select one randomly
            unvisited = [child for child in node.children if child.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            
            # Otherwise, select the child with the highest UCB score
            node = self._select_ucb(node)
        
        return node
    
    def _select_ucb(self, node):
        """
        Select a child node using UCB1.
        
        Args:
            node (Node): The parent node.
            
        Returns:
            Node: The selected child node.
        """
        log_n = math.log(node.visits + 1)  # Add 1 to avoid log(0)
        
        def ucb(n):
            if n.visits == 0:
                return float('inf')
            return n.score + self.exploration_weight * math.sqrt(2 * log_n / n.visits)
        
        return max(node.children, key=ucb)
    
    def expand(self, node, operators, num_expansions=1):
        """
        Expand a node by applying operators.
        
        Args:
            node (Node): The node to expand.
            operators (dict): The operators to apply.
            num_expansions (int, optional): The number of times to apply each operator. Defaults to 1.
            
        Returns:
            list: The list of new child nodes.
        """
        new_children = []
        
        for _ in range(num_expansions):
            for op_name, op_func in operators.items():
                try:
                    # Apply the operator to generate new code
                    new_code = op_func(node, self, self.client, self.prompts, self.prompt_key)
                    
                    # Validate and fix the code
                    if not CodeValidator.validate_code(new_code, self.function_name):
                        print(f"Invalid code generated with operator {op_name}, attempting to fix...")
                        new_code = CodeValidator.fix_code(new_code, self.function_name)
                        
                        # Validate again after fixing
                        if not CodeValidator.validate_code(new_code, self.function_name):
                            print(f"Could not fix code generated with operator {op_name}, skipping...")
                            continue
                    
                    # Create a new child node
                    child = Node(new_code, self.function_name, parent=node, depth=node.depth + 1)
                    print(child.function_code)
                    node.add_child(child)
                    new_children.append(child)
                    
                    print(f"Created new node with operator {op_name}")
                except Exception as e:
                    print(f"Error applying operator {op_name}: {e}")
        
        return new_children
    
    def simulate(self, node):
        """
        Simulate the performance of a node by evaluating its function.
        
        Args:
            node (Node): The node to simulate.
            
        Returns:
            float: The score of the node.
        """
        try:
            function_file_path = os.path.join("problems", "tsp", f"{self.function_name}.py")
            
            # Backup the original file
            with open(function_file_path, 'r') as f:
                original_code = f.read()
            
            # Write the new function implementation
            with open(function_file_path, 'w') as f:
                f.write(node.function_code)

            
            try:
                # Run the evaluation script
                result = subprocess.run(
                    ["python", os.path.join("problems", "tsp", "eval.py")],
                    capture_output=True,
                    text=True,
                    timeout=60  # Set a timeout of 60 seconds
                )
                
                # Parse the improvement percentage
                output = result.stdout.strip()
                print(f"Evaluation output: {output}")
                
                improvement = float(output)
                
                # Store the improvement and calculate the score
                node.improvement = improvement
                node.score = max(0, improvement)  # Negative improvements are considered as 0
            except subprocess.TimeoutExpired:
                print("Evaluation timed out, setting score to 0")
                node.improvement = 0.0
                node.score = 0.0
            except Exception as e:
                print(f"Error running evaluation: {e}")
                node.improvement = 0.0
                node.score = 0.0
            finally:
                # Restore the original file
                with open(function_file_path, 'w') as f:
                    f.write(original_code)
            
            return node.score
        except Exception as e:
            print(f"Error simulating node: {e}")
            node.score = 0.0
            return 0.0
    
    def backpropagate(self, node, score):
        """
        Backpropagate the score to all ancestors of the node.
        
        Args:
            node (Node): The node to start backpropagation from.
            score (float): The score to backpropagate.
        """
        while node is not None:
            node.visits += 1
            
            # Update the best node if needed
            if node.score > self.best_node.score or (node.score == self.best_node.score and node.depth < self.best_node.depth):
                self.best_node = node
            
            node = node.parent
    
    def get_best_node(self):
        """
        Get the best node in the tree.
        
        Returns:
            Node: The best node.
        """
        return self.best_node
    
    def get_elite_nodes(self, n=5):
        """
        Get the top n nodes in the tree.
        
        Args:
            n (int, optional): The number of nodes to return. Defaults to 5.
            
        Returns:
            list: The top n nodes.
        """
        def collect_nodes(node, nodes):
            nodes.append(node)
            for child in node.children:
                collect_nodes(child, nodes)
        
        all_nodes = []
        collect_nodes(self.root, all_nodes)
        
        # Sort nodes by score (descending)
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return all_nodes[:n]