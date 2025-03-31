import random

class Operators:
    @staticmethod
    def self_reflection(node, tree, client, prompts, prompt_key):
        """
        Self-Reflection (SR) operator: Ask LLM to self-evaluate strategies and performance on the current branch.
        
        Args:
            node: The current node.
            tree: The MCTS tree.
            client: The LLM client.
            prompts: The prompts dictionary.
            prompt_key: The key for the prompt to use.
            
        Returns:
            str: The generated strategy code.
        """
        path = node.get_path_to_root()
        
        # Create a prompt that includes the path of strategy implementations
        system_prompt = prompts[prompt_key]
        user_prompt = "Review and improve the following strategy implementations in the current branch. Analyze their strengths and weaknesses, then create a new implementation that addresses the identified issues.\n\n"
        
        for i, n in enumerate(reversed(path)):
            if n.function_code:
                user_prompt += f"Implementation {i} (depth={n.depth}, improvement={n.improvement:.2f}%):\n```python\n{n.function_code}\n```\n\n"
        
        user_prompt += "Based on your analysis, implement a new improved version of the strategy with better performance."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        code, _ = client.get_code(messages)
        return code
    
    @staticmethod
    def ensemble_fusion(node, tree, client, prompts, prompt_key):
        """
        Ensemble Fusion (EF) operator: Ask LLM to combine ideas from multiple elite strategies.
        
        Args:
            node: The current node.
            tree: The MCTS tree.
            client: The LLM client.
            prompts: The prompts dictionary.
            prompt_key: The key for the prompt to use.
            
        Returns:
            str: The generated strategy code.
        """
        # Get top performing nodes from the tree
        elite_nodes = tree.get_elite_nodes(3)  # Get top 3 nodes
        
        system_prompt = prompts[prompt_key]
        user_prompt = "Synthesize ideas from the following high-performing strategy implementations to create a new implementation:\n\n"
        
        for i, n in enumerate(elite_nodes):
            if n.function_code:
                user_prompt += f"Implementation {i} (improvement={n.improvement:.2f}%):\n```python\n{n.function_code}\n```\n\n"
        
        user_prompt += "Create a new implementation that combines the best aspects of these strategies to achieve even better performance."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        code, _ = client.get_code(messages)
        return code
    
    @staticmethod
    def diversity_exploration(node, tree, client, prompts, prompt_key):
        """
        Diversity Exploration (DE) operator: Ask LLM to create a completely different strategy.
        
        Args:
            node: The current node.
            tree: The MCTS tree.
            client: The LLM client.
            prompts: The prompts dictionary.
            prompt_key: The key for the prompt to use.
            
        Returns:
            str: The generated strategy code.
        """
        system_prompt = prompts[prompt_key]
        user_prompt = "Create a completely new strategy implementation with a different approach than previously explored. Be creative and innovative, but ensure the implementation is effective for the Traveling Salesman Problem."
        
        if node.function_code:
            user_prompt += f"\n\nCurrent implementation:\n```python\n{node.function_code}\n```\n\nYour implementation should be distinctly different from this."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        code, _ = client.get_code(messages)
        return code
    
    @staticmethod
    def memory_reuse(node, tree, client, prompts, prompt_key):
        """
        Memory-based Reuse (MR) operator: Reuse patterns from past successful implementations.
        
        Args:
            node: The current node.
            tree: The MCTS tree.
            client: The LLM client.
            prompts: The prompts dictionary.
            prompt_key: The key for the prompt to use.
            
        Returns:
            str: The generated strategy code.
        """
        # Get the best performing node
        best_node = tree.get_best_node()
        
        system_prompt = prompts[prompt_key]
        user_prompt = "Adapt the following high-performing strategy implementation to create a new variant:\n\n"
        
        if best_node and best_node.function_code:
            user_prompt += f"Best implementation (improvement={best_node.improvement:.2f}%):\n```python\n{best_node.function_code}\n```\n\n"
        
        user_prompt += f"Current implementation:\n```python\n{node.function_code}\n```\n\n"
        user_prompt += "Create a new implementation that adapts the best aspects of the top implementation while introducing strategic modifications for improved performance."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        code, _ = client.get_code(messages)
        return code
    
    @staticmethod
    def guided_randomness(node, tree, client, prompts, prompt_key):
        """
        Guided Randomness (GR) operator: Introduce random variations based on statistical insights.
        
        Args:
            node: The current node.
            tree: The MCTS tree.
            client: The LLM client.
            prompts: The prompts dictionary.
            prompt_key: The key for the prompt to use.
            
        Returns:
            str: The generated strategy code.
        """
        system_prompt = prompts[prompt_key]
        user_prompt = "Introduce controlled random variations to the following strategy implementation:\n\n"
        
        if node.function_code:
            user_prompt += f"```python\n{node.function_code}\n```\n\n"
        
        user_prompt += "Add randomness or statistical elements to improve exploration and avoid bias in optimization. This could include random sampling, probabilistic decisions, or adaptive parameters. The goal is to avoid local optima and encourage more diverse solution exploration in ant colony optimization."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        code, _ = client.get_code(messages)
        return code