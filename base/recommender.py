# =============================================================================
# Base Recommender Class
# =============================================================================
# Abstract base class for all recommendation models.
# Provides common functionality:
# - Configuration loading
# - Data handling
# - Logging
# - Model information display
# =============================================================================

from data.data import Data
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time


class Recommender:
    """Base class for all recommendation models.
    
    This class provides common initialization and utility methods
    that all models inherit. Child classes (GraphRecommender, etc.)
    extend this with specific functionality.
    
    Attributes:
        config: Configuration dictionary from YAML
        data: Data object with training/test sets
        model_name (str): Name of the model (e.g., 'LightGCN')
        ranking (list): Top-K values for evaluation (e.g., [10, 20])
        emb_size (int): Embedding dimensionality
        maxEpoch (int): Maximum training epochs
        batch_size (int): Training batch size
        lRate (float): Learning rate
        reg (float): L2 regularization coefficient
        output (str): Directory for saving results
        model_log: Logger instance for this model
    """
    def __init__(self, conf, training_set, test_set, **kwargs):
        # Store configuration
        self.config = conf
        
        # Initialize Data object (child classes may override with specialized versions)
        self.data = Data(self.config, training_set, test_set)

        # Extract commonly used config values and store as attributes
        # This avoids repeated dictionary lookups during training
        model_config = self.config['model']
        self.model_name = model_config['name']  # e.g., 'LightGCN'
        
        # Evaluation settings: which Top-K values to compute
        self.ranking = self.config['item.ranking.topN']  # e.g., [10, 20]
        
        # Model hyperparameters
        self.emb_size = int(self.config['embedding.size'])      # Embedding dimension (e.g., 64)
        self.maxEpoch = int(self.config['max.epoch'])           # Training epochs (e.g., 100)
        self.batch_size = int(self.config['batch.size'])        # Batch size (e.g., 2048)
        self.lRate = float(self.config['learning.rate'])        # Learning rate (e.g., 0.001)
        self.reg = float(self.config['reg.lambda'])             # L2 reg (e.g., 0.0001)
        
        # Output directory for results
        self.output = self.config['output']

        # Initialize logger with timestamp
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, f"{self.model_name} {current_time}")

        # Storage for results
        self.result = []       # Performance metrics
        self.recOutput = []    # Detailed recommendations

    def initializing_log(self):
        """Log model configuration at the start of training.
        
        Writes all config parameters to log file for reproducibility.
        """
        self.model_log.add('### model configuration ###')
        config_items = self.config.config
        for k in config_items:
            self.model_log.add(f"{k}={str(config_items[k])}")

    def print_model_info(self):
        """Display model information and hyperparameters.
        
        Called at the start of training to show configuration.
        Child classes can extend this to show model-specific info.
        """
        print('Model:', self.model_name)
        print('Training Set:', abspath(self.config['training.set']))
        print('Test Set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:', self.reg)

        # Display model-specific parameters if they exist
        model_name = self.config['model']['name']
        if self.config.contain(model_name):
            args = self.config[model_name]
            parStr = '  '.join(f"{key}:{args[key]}" for key in args)
            print('Specific parameters:', parStr)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)
