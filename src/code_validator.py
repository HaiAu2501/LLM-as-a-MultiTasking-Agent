import ast
import re

class CodeValidator:
    """
    Validator for ensuring generated code is syntactically correct and has the correct signature.
    """
    
    @staticmethod
    def validate_code(code, function_name):
        """
        Validate the generated code for syntax errors and correct signature.
        
        Args:
            code (str): The code to validate.
            function_name (str): The name of the function to validate.
            
        Returns:
            bool: True if the code is valid, False otherwise.
        """
        try:
            # Check syntax
            ast.parse(code)
            
            # Get the expected signature based on the function name
            if function_name == "heuristic":
                expected_args = ["distances"]
                expected_return_type = "torch.Tensor"
            elif function_name == "calculate_probabilities":
                expected_args = ["pheromone_values", "heuristic_values", "alpha", "beta"]
                expected_return_type = "torch.Tensor"
            elif function_name == "deposit_pheromone":
                expected_args = ["pheromone", "paths", "costs"]
                expected_return_type = "torch.Tensor"
            else:
                raise ValueError(f"Unknown function name: {function_name}")
            
            # Parse the code to check the function definition
            tree = ast.parse(code)
            
            # Find the function definition
            function_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_def = node
                    break
            
            if not function_def:
                print(f"Function {function_name} not found in the code")
                return False
            
            # Check the function arguments
            args = [arg.arg for arg in function_def.args.args]
            if not all(expected_arg in args for expected_arg in expected_args):
                print(f"Expected arguments {expected_args}, got {args}")
                return False
            
            # Check for import torch
            has_torch_import = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name == "torch":
                            has_torch_import = True
                            break
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "torch":
                        has_torch_import = True
                        break
            
            if not has_torch_import:
                print("Missing 'import torch' statement")
                return False
            
            return True
        except SyntaxError as e:
            print(f"Syntax error in generated code: {e}")
            return False
        except Exception as e:
            print(f"Error validating code: {e}")
            return False
    
    @staticmethod
    def fix_code(code, function_name):
        """
        Attempt to fix common issues in the generated code.
        
        Args:
            code (str): The code to fix.
            function_name (str): The name of the function.
            
        Returns:
            str: The fixed code.
        """
        # Ensure the code has import torch
        if "import torch" not in code:
            code = "import torch\n\n" + code
        
        # Ensure the function has the correct name
        if f"def {function_name}" not in code:
            # Try to find any function definition
            match = re.search(r"def\s+(\w+)", code)
            if match:
                wrong_name = match.group(1)
                code = code.replace(f"def {wrong_name}", f"def {function_name}")
        
        # Get the expected signature based on the function name
        if function_name == "heuristic":
            expected_signature = "def heuristic(distances: torch.Tensor) -> torch.Tensor:"
        elif function_name == "calculate_probabilities":
            expected_signature = "def calculate_probabilities(pheromone_values: torch.Tensor, heuristic_values: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:"
        elif function_name == "deposit_pheromone":
            expected_signature = "def deposit_pheromone(pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:"
        
        # Check if the function signature needs to be fixed
        current_signature = re.search(fr"def\s+{function_name}\s*\([^)]*\)\s*(?:->.*?)?:", code)
        if current_signature:
            code = code.replace(current_signature.group(0), expected_signature)
        
        return code