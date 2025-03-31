import ast
import re

class CodeValidator:
    """
    Validator for ensuring generated code is syntactically correct and has the correct structure.
    """
    
    @staticmethod
    def validate_code(code, function_name):
        """
        Validate the generated code for syntax errors and correct class structure.
        
        Args:
            code (str): The code to validate.
            function_name (str): The name of the function to validate.
            
        Returns:
            bool: True if the code is valid, False otherwise.
        """
        try:
            # Check syntax
            ast.parse(code)
            
            # Define expected class names based on function name
            if function_name == "heuristic":
                expected_class = "HeuristicImpl"
                expected_method = "compute"
                expected_base_class = "HeuristicStrategy"
            elif function_name == "calculate_probabilities":
                expected_class = "ProbabilityImpl"
                expected_method = "compute"
                expected_base_class = "ProbabilityStrategy"
            elif function_name == "deposit_pheromone":
                expected_class = "PheromoneImpl"
                expected_method = "update"
                expected_base_class = "PheromoneStrategy"
            else:
                raise ValueError(f"Unknown function name: {function_name}")
            
            # Parse the code to check the class definition
            tree = ast.parse(code)
            
            # Find the class definition
            class_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == expected_class:
                    class_def = node
                    break
            
            if not class_def:
                print(f"Class {expected_class} not found in the code")
                return False
            
            # Check if the class inherits from the correct base class
            has_correct_base = False
            for base in class_def.bases:
                if isinstance(base, ast.Name) and base.id == expected_base_class:
                    has_correct_base = True
                    break
            
            if not has_correct_base:
                print(f"Class must inherit from {expected_base_class}")
                return False
            
            # Check for the required method
            method_found = False
            for node in ast.walk(class_def):
                if isinstance(node, ast.FunctionDef) and node.name == expected_method:
                    method_found = True
                    break
            
            if not method_found:
                print(f"Required method '{expected_method}' not found in class {expected_class}")
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
            
            # Check for import of the strategy class
            has_strategy_import = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "aco":
                    for name in node.names:
                        if name.name == expected_base_class:
                            has_strategy_import = True
                            break
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name == "aco":
                            has_strategy_import = True  # Assume aco import will make the class available
                            break
            
            if not has_strategy_import:
                print(f"Missing import of {expected_base_class} from aco module")
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
        # Define expected structure based on function name
        if function_name == "heuristic":
            expected_class = "HeuristicImpl"
            expected_method = "compute"
            expected_base_class = "HeuristicStrategy"
            expected_method_signature = "def compute(self, distances: torch.Tensor) -> torch.Tensor:"
        elif function_name == "calculate_probabilities":
            expected_class = "ProbabilityImpl"
            expected_method = "compute"
            expected_base_class = "ProbabilityStrategy"
            expected_method_signature = "def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:"
        elif function_name == "deposit_pheromone":
            expected_class = "PheromoneImpl"
            expected_method = "update"
            expected_base_class = "PheromoneStrategy"
            expected_method_signature = "def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor, decay: float) -> torch.Tensor:"
        else:
            return code  # Unknown function, can't fix
        
        # Ensure the code has import torch
        if "import torch" not in code:
            code = "import torch\n\n" + code
        
        # Ensure the code imports the strategy class
        if f"from aco import {expected_base_class}" not in code:
            code = f"from aco import {expected_base_class}\n" + code
        
        # Ensure the class has the correct name
        class_match = re.search(r"class\s+(\w+)\s*\(", code)
        if class_match:
            current_class = class_match.group(1)
            if current_class != expected_class:
                code = code.replace(f"class {current_class}", f"class {expected_class}")
        
        # Ensure the class inherits from the correct base class
        class_match = re.search(r"class\s+\w+\s*\((.*?)\):", code)
        if class_match:
            current_bases = class_match.group(1)
            if expected_base_class not in current_bases:
                if current_bases.strip():
                    # Replace existing inheritance
                    code = code.replace(f"({current_bases})", f"({expected_base_class})")
                else:
                    # Add inheritance where none exists
                    code = code.replace("class " + expected_class + ":", f"class {expected_class}({expected_base_class}):")
        
        # Check if the method exists and has the correct signature
        method_match = re.search(rf"def\s+{expected_method}\s*\([^)]*\)\s*(?:->.*?)?:", code)
        if not method_match:
            # Try to find any method in the class
            if "def " in code:
                # Replace the first method definition with the expected one
                method_pattern = r"(def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:)"
                replacement = f"{expected_method_signature}"
                code = re.sub(method_pattern, replacement, code, count=1)
            else:
                # Add the method to the class
                class_end_match = re.search(r"class\s+\w+\s*\(.*?\):\s*", code)
                if class_end_match:
                    insert_position = class_end_match.end()
                    indented_method = f"\n    {expected_method_signature}\n        pass\n"
                    code = code[:insert_position] + indented_method + code[insert_position:]
        else:
            # Fix the method signature
            current_signature = method_match.group(0)
            code = code.replace(current_signature, expected_method_signature)
        
        return code