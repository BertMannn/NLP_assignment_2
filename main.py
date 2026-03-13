from src.eval.eval_models import evaluate_nn_models
from src.analysis.error_analysis import run_error_analysis

def main() -> None:
    evaluate_nn_models()
    run_error_analysis(n=20)
    
    
if __name__ == "__main__":
    main()