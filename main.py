import os
import sys
import time
import traceback
import hydra
import logging
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
    print(f"Starting Hierarchical Monte Carlo Tree Search for {cfg.problem.active.upper()} Function Optimization")

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
        os.makedirs(cfg.paths.results_dir, exist_ok=True)
        
        # Initialize the Hierarchical MCTS with the config
        hmcts = HierarchicalMCTS(
            client=client,
            prompts=prompts,
            cfg=cfg
        )
        
        # Run the Hierarchical MCTS
        best_implementations = hmcts.run()
        
        print("\nFinal implementations:")
        
        # Print summary of best implementations
        active_problem = cfg.problem.active
        problem_config = getattr(cfg.problem, active_problem)
        
        for function in problem_config.functions:
            function_id = function.id
            function_name = function.name
            
            if function_id in best_implementations:
                implementation = best_implementations[function_id]
                result_file = f"results/best_{function_id}_{active_problem}.py"
                
                print(f"\n{'='*20} Best {function_name} ({function_id}) implementation {'='*20}")
                print(f"Saved to {result_file}")
                print(f"{'='*60}")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        
        # Try to save any intermediate results if available
        try:
            if 'hmcts' in locals() and hasattr(hmcts, 'best_implementations'):
                for function_id, implementation in hmcts.best_implementations.items():
                    # Find function name from config
                    function_name = function_id
                    for func in getattr(cfg.problem, cfg.problem.active).functions:
                        if func.id == function_id:
                            function_name = func.name
                            break
                    
                    # Save to emergency file
                    with open(os.path.join(cfg.paths.results_dir, f"emergency_save_{function_id}_{cfg.problem.active}.py"), "w") as f:
                        f.write(implementation)
                print(f"Saved intermediate results to {cfg.paths.results_dir} directory.")
        except Exception as save_error:
            print(f"Could not save intermediate results: {save_error}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()