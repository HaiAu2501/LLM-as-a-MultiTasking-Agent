import math
import random
import os
import sys
import subprocess
import numpy as np
from src.node import Node
from src.code_validator import CodeValidator
from omegaconf import DictConfig

class MCTS:
    """
    Implementation of the Monte Carlo Tree Search algorithm for strategy optimization.
    """
    
    def __init__(
        self, 
        function_name, 
        function_id, 
        initial_code, 
        client, 
        prompts, 
        prompt_key,
        problem_config,
        exploration_weight=1.0, 
        max_depth=3
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            function_name (str): The name of the strategy class to optimize.
            function_id (str): The ID of the strategy (e.g., F1, F2).
            initial_code (str): The initial implementation of the strategy.
            client: The LLM client for generating new strategy implementations.
            prompts: The prompts dictionary for guiding the LLM.
            prompt_key: The key in the prompts dictionary to use.
            problem_config: The problem configuration.
            exploration_weight (float, optional): The exploration weight for UCB. Defaults to 1.0.
            max_depth (int, optional): The maximum depth of the tree. Defaults to 3.
        """
        self.function_name = function_name
        self.function_id = function_id
        self.root = Node(initial_code, function_name, depth=0)
        self.client = client
        self.prompts = prompts
        self.prompt_key = prompt_key
        self.problem_config = problem_config
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.best_node = self.root
        self.baseline_cost = None  # Sẽ được tính khi đánh giá node gốc
    
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
                    if not CodeValidator.validate_code(new_code, self._get_validator_function_name()):
                        print(f"Invalid code generated with operator {op_name}, attempting to fix...")
                        new_code = CodeValidator.fix_code(new_code, self._get_validator_function_name())
                        
                        # Validate again after fixing
                        if not CodeValidator.validate_code(new_code, self._get_validator_function_name()):
                            print(f"Could not fix code generated with operator {op_name}, skipping...")
                            continue
                    
                    # Create a new child node with operator information
                    child = Node(new_code, self.function_name, parent=node, depth=node.depth + 1, creator_operator=op_name)
                    node.add_child(child)
                    new_children.append(child)
                    
                    print(f"Created new node with operator {op_name}")
                except Exception as e:
                    print(f"Error applying operator {op_name}: {e}")
        
        return new_children
    
    def _get_validator_function_name(self):
        """
        Get the function name used by the validator, which is based on
        legacy function names for backward compatibility.
        
        Returns:
            str: Function name for validator
        """
        # Map strategy class names to legacy function names for validator
        if self.function_name == "HeuristicImpl":
            return "heuristic"
        elif self.function_name == "ProbabilityImpl":
            return "calculate_probabilities"
        elif self.function_name == "PheromoneImpl":
            return "deposit_pheromone"
        else:
            return self.function_name.lower()
    
    def simulate(self, node):
        """
        Simulate the performance of a node by evaluating its strategy.
        
        Args:
            node (Node): The node to simulate.
            
        Returns:
            float: The score of the node.
        """
        try:
            # Get function file path from config
            function_path = None
            for func in self.problem_config.functions:
                if func.id == self.function_id:
                    function_path = func.path
                    break
            
            if not function_path:
                raise ValueError(f"Strategy path not found for {self.function_id}")
            
            # Backup the original file
            with open(function_path, 'r') as f:
                original_code = f.read()
            
            # Write the new strategy implementation
            with open(function_path, 'w') as f:
                f.write(node.function_code)
            
            try:
                # Run the evaluation script
                result = subprocess.run(
                    [sys.executable, self.problem_config.eval_script],
                    capture_output=True,
                    text=True,
                    timeout=300  # Set a timeout of 5 minutes
                )
                
                # Parse the average cost
                output = result.stdout.strip()
                print(f"Evaluation output: {output}")
                
                # Kiểm tra output có trống không
                if not output:
                    print("Empty evaluation output, setting score to 0")
                    node.improvement = 0.0
                    node.score = 0.0
                    return 0.0
                
                # Trích xuất giá trị avg_cost từ output
                try:
                    avg_cost = float(output)
                except ValueError:
                    # Tìm kiếm giá trị số trong output
                    import re
                    matches = re.findall(r"[-+]?\d*\.\d+|\d+", output)
                    if matches:
                        avg_cost = float(matches[0])
                    else:
                        print("Could not parse cost value, defaulting to 0")
                        node.improvement = 0.0
                        node.score = 0.0
                        return 0.0
                
                # Nếu là node gốc, lưu baseline_cost
                if node == self.root:
                    self.baseline_cost = avg_cost
                    node.improvement = 0.0
                    node.score = 0.5  # Điểm trung bình làm cơ sở so sánh
                    return node.score
                
                # Tính improvement so với baseline
                improvement_over_baseline = (self.baseline_cost - avg_cost) / self.baseline_cost
                
                # Tính improvement so với node cha
                parent_avg_cost = self.baseline_cost
                if node.parent != self.root:
                    # Tính ngược từ improvement của parent để có parent_avg_cost
                    parent_improvement = node.parent.improvement / 100  # Chuyển từ % sang tỉ lệ
                    parent_avg_cost = self.baseline_cost * (1 - parent_improvement)
                
                improvement_over_parent = (parent_avg_cost - avg_cost) / parent_avg_cost
                
                # Lưu improvement so với baseline (dạng phần trăm)
                node.improvement = improvement_over_baseline * 100
                
                # Hàm chuẩn hóa bất đối xứng: khuếch đại cải thiện nhỏ và đánh nặng sụt giảm
                def asymmetric_normalization(improvement):
                    if improvement >= 0:
                        # Với cải thiện dương: khuếch đại mạnh các giá trị nhỏ (10-50%)
                        # Hệ số 5.0 làm cho cải thiện 10% đạt khoảng 0.7, 50% đạt khoảng 0.9
                        return 0.5 + 0.5 * (1 - np.exp(-5.0 * improvement))
                    else:
                        # Với sụt giảm: giảm rất nhanh khi có sụt giảm lớn
                        # Hệ số 3.0 làm cho sụt giảm -50% còn khoảng 0.2, -200% gần bằng 0
                        return 0.5 * np.exp(3.0 * improvement)
                
                normalized_baseline_imp = asymmetric_normalization(improvement_over_baseline)
                normalized_parent_imp = asymmetric_normalization(improvement_over_parent)
                
                # Tính điểm kết hợp (trọng số 60% cho baseline, 40% cho parent)
                node.score = 0.6 * normalized_baseline_imp + 0.4 * normalized_parent_imp
                
                # Đảm bảo score không âm
                node.score = max(0, node.score)
                
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
                with open(function_path, 'w') as f:
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