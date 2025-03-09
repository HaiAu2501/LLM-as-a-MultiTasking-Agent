import os
import sys
import time
import traceback
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from utils.client import init_client
from problems.tsp.prompts import F1, F2, F3
from src.hierarchical_mcts import HierarchicalMCTS

load_dotenv()

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
    Main entry point for the Hierarchical Monte Carlo Tree Search.
    """
    print("Starting Hierarchical Monte Carlo Tree Search for Function Optimization")
    start_time = time.time()
    
    try:
        # Initialize the LLM client
        client = init_client(cfg)
        
        # Define the prompts
        prompts = {
            "F1": F1,
            "F2": F2,
            "F3": F3
        }
        
        # Create output directory for results
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize the Hierarchical MCTS
        hmcts = HierarchicalMCTS(
            client,
            prompts,
            iterations_per_function=5,  # Number of iterations per function
            max_depth=3                # Maximum depth of each tree
        )
        
        # Run the Hierarchical MCTS
        best_implementations = hmcts.run()
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n\nOptimization complete in {int(hours)}h {int(minutes)}m {int(seconds)}s!")
        print("Final implementations:")
        
        # Save and print the best implementations
        for function_name, implementation in best_implementations.items():
            # Save to file
            with open(f"results/best_{function_name}.py", "w") as f:
                f.write(implementation)
            
            print(f"\n{'='*20} Best {function_name} implementation {'='*20}")
            print(implementation)
            print(f"{'='*60}")
            print(f"Saved to results/best_{function_name}.py")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        
        # Try to save any intermediate results if available
        try:
            if 'hmcts' in locals() and hasattr(hmcts, 'best_implementations'):
                results_dir = os.path.join(PROJECT_ROOT, "results")
                os.makedirs(results_dir, exist_ok=True)
                for function_name, implementation in hmcts.best_implementations.items():
                    with open(os.path.join(results_dir, f"emergency_save_{function_name}.py"), "w") as f:
                        f.write(implementation)
                print(f"Saved intermediate results to {results_dir} directory.")
        except Exception as save_error:
            print(f"Could not save intermediate results: {save_error}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()