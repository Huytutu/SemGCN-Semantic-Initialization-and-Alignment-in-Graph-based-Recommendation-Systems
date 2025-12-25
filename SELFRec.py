# =============================================================================
# SELFRec Framework Core
# =============================================================================
# This module contains the main SELFRec class that orchestrates the entire
# recommendation framework. It handles data loading, configuration management,
# and dynamic model instantiation.
# =============================================================================

from data.loader import FileIO


class SELFRec(object):
    """Core framework class for self-supervised recommendation.
    
    This class is responsible for:
    1. Loading training and test datasets
    2. Managing configuration settings
    3. Loading optional social network data
    4. Dynamically importing and executing the selected model
    
    Attributes:
        config (dict): Configuration dictionary loaded from YAML
        training_data (list): Training interactions (user, item, rating)
        test_data (list): Test interactions (user, item, rating)
        social_data (list): Optional social network edges (user1, user2, weight)
        kwargs (dict): Additional keyword arguments passed to models
    """
    def __init__(self, config):
        # Initialize data structures
        self.social_data = []  # For social network-based models (not used in LightGCN/MF)
        self.feature_data = []  # For feature-based models (not used in LightGCN/MF)
        self.config = config
        
        # Load training dataset from file specified in config
        # Format: user_id item_id rating (one interaction per line)
        self.training_data = FileIO.load_data_set(config['training.set'], config['model']['type'])
        
        # Load test dataset for evaluation
        self.test_data = FileIO.load_data_set(config['test.set'], config['model']['type'])

        # Dictionary to pass additional data to models (e.g., social networks)
        self.kwargs = {}
        
        print('Reading data and preprocessing...')

    def execute(self):
        """Dynamically import and execute the selected recommendation model.
        
        This method uses Python's exec() and eval() to dynamically:
        1. Import the model class from its module path
        2. Instantiate the model with config and data
        3. Call the model's execute() method to train and evaluate
        
        The model path is constructed as:
        model/{type}/{name}.py where type is 'graph' or 'sequential'
        """
        # Construct import statement dynamically
        # Example: "from model.graph.LightGCN import LightGCN"
        import_str = f"from model.{self.config['model']['type']}.{self.config['model']['name']} import {self.config['model']['name']}"
        
        # Execute the import statement to load the model class
        exec(import_str)
        
        # Construct the model instantiation string
        # Example: "LightGCN(self.config, self.training_data, self.test_data, **self.kwargs)"
        recommender = f"{self.config['model']['name']}(self.config,self.training_data,self.test_data,**self.kwargs)"
        
        # Evaluate the string to create model instance and call execute()
        # This runs the model's training and evaluation pipeline
        eval(recommender).execute()
