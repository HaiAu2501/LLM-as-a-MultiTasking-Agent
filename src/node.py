class Node:
    """
    Represents a node in the Monte Carlo Tree Search.
    Each node contains a specific implementation of a function.
    """
    
    def __init__(self, function_code, function_name, parent=None, depth=0, visits=0, score=0.0, creator_operator=None):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            function_code (str): The implementation code of the function.
            function_name (str): Name of the function (heuristic, calculate_probabilities, or deposit_pheromone).
            parent (Node, optional): The parent node. Defaults to None.
            depth (int, optional): The depth of the node in the tree. Defaults to 0.
            visits (int, optional): The number of times the node has been visited. Defaults to 0.
            score (float, optional): The score of the node. Defaults to 0.0.
            creator_operator (str, optional): The name of operator that created this node. Defaults to None.
        """
        self.function_code = function_code
        self.function_name = function_name
        self.parent = parent
        self.children = []
        self.depth = depth
        self.visits = visits
        self.score = score  # Higher is better
        self.improvement = 0.0  # Improvement over baseline (percentage)
        self.creator_operator = creator_operator
        
    def add_child(self, child):
        """
        Add a child node to this node.
        
        Args:
            child (Node): The child node to add.
        """
        self.children.append(child)
        
    def get_path_to_root(self):
        """
        Get the path from this node to the root.
        
        Returns:
            list: A list of nodes from this node to the root.
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path
    
    def __str__(self):
        """String representation of the node."""
        return f"Node({self.function_name}, depth={self.depth}, visits={self.visits}, score={self.score:.4f}, improvement={self.improvement:.2f}%, operator={self.creator_operator})"
    
    def __repr__(self):
        """Representation of the node."""
        return self.__str__()