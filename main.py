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
        os.makedirs("results", exist_ok=True)
        
        # Initialize the Hierarchical MCTS
        hmcts = HierarchicalMCTS(
            client,
            prompts,
            iterations_per_function=5,  
            max_depth=3                
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
                for function_name, implementation in hmcts.best_implementations.items():
                    with open(f"results/emergency_save_{function_name}.py", "w") as f:
                        f.write(implementation)
                print("Saved intermediate results to results/ directory.")
        except:
            print("Could not save intermediate results.")
        
        sys.exit(1)

if __name__ == "__main__":
    main()