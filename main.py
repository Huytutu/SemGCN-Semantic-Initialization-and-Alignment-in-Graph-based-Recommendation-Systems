from SELFRec import SELFRec
from util.conf import ModelConf
import time

def print_models(title, models):
    """Display available models in a formatted table.
    
    Args:
        title (str): Header text for the model list
        models (dict): Dictionary mapping categories to model lists
    """
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    # Dictionary of available models organized by category
    # Only graph-based models are included (LightGCN and Matrix Factorization)
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'MF', 'LightGCN_Semantic']
    }

    # Display welcome banner and available models
    print('=' * 80)
    print_models("Available Models", models)

    # Get user input for model selection
    model = input('Please enter the model you want to run (LightGCN/MF/LightGCN_Semantic): ')

    # Start timing the execution
    s = time.time()
    
    # Flatten the models dictionary to a single list for validation
    all_models = sum(models.values(), [])
    
    # Validate model name and execute
    if model in all_models:
        # Load configuration file for the selected model
        conf = ModelConf(f'./conf/{model}.yaml')
        
        # Initialize the recommendation framework
        rec = SELFRec(conf)
        
        # Execute training and evaluation
        rec.execute()
        
        # Display total execution time
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        # Handle invalid model name
        print('Wrong model name!')
        exit(-1)
