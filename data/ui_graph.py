# =============================================================================
# User-Item Interaction Graph
# =============================================================================
# This module creates and manages the user-item bipartite graph structure.
# It handles:
# 1. Building adjacency matrices from interaction data
# 2. Normalizing graph for message passing
# 3. Providing data access methods for models
# =============================================================================

import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp


class Interaction(Data, Graph):
    """Manages user-item interaction data and graph structure.
    
    This class combines data handling (from Data) and graph operations (from Graph)
    to create a complete interaction graph with:
    - User/item ID mappings
    - Training and test set dictionaries
    - Sparse adjacency matrices for efficient computation
    - Normalized adjacency for GNN message passing
    
    Attributes:
        user (dict): Maps user string IDs to integer indices
        item (dict): Maps item string IDs to integer indices
        id2user (dict): Reverse mapping from indices to user IDs
        id2item (dict): Reverse mapping from indices to item IDs
        training_set_u (dict): {user: {item: rating}} for training
        training_set_i (dict): {item: {user: rating}} for training
        test_set (dict): {user: {item: rating}} for testing
        ui_adj (scipy.sparse): Bipartite adjacency matrix
        norm_adj (scipy.sparse): Normalized adjacency (D^-1/2 @ A @ D^-1/2)
    """
    def __init__(self, conf, training, test):
        """Initialize interaction graph from training and test data.
        
        Args:
            conf: Configuration dictionary
            training: List of (user, item, rating) training interactions
            test: List of (user, item, rating) test interactions
        """
        # Initialize parent classes
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        # Initialize ID mapping dictionaries
        self.user = {}           # user_string -> user_id
        self.item = {}           # item_string -> item_id
        self.id2user = {}        # user_id -> user_string
        self.id2item = {}        # item_id -> item_string
        
        # Training data structures (for fast lookup)
        self.training_set_u = defaultdict(dict)  # {user: {item: rating}}
        self.training_set_i = defaultdict(dict)  # {item: {user: rating}}
        
        # Test data structures
        self.test_set = defaultdict(dict)        # {user: {item: rating}}
        self.test_set_item = set()               # Set of items in test set

        # Build all data structures
        self.__generate_set()
        
        # Store dimensions
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        
        # Create graph structures
        # Bipartite adjacency: connections between users and items
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        
        # Normalized adjacency for GNN: D^-1/2 @ A @ D^-1/2
        # This is the key matrix used in LightGCN message passing
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        
        # Interaction matrix: users x items (for alternative computations)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        """Generate user/item ID mappings and training/test dictionaries.
        
        This method:
        1. Assigns integer IDs to users and items sequentially
        2. Builds training set dictionaries for fast lookup
        3. Builds test set dictionaries (only for known users/items)
        """
        # Process training data
        for user, item, rating in self.training_data:
            # Assign user ID if new user
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            
            # Assign item ID if new item
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            
            # Store in both user-centric and item-centric views
            self.training_set_u[user][item] = 1  # Binary interaction (1 = interacted)
            self.training_set_i[item][user] = 1

        # Process test data (only keep interactions with known users/items)
        for user, item, rating in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = 1
                self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        """Create sparse bipartite adjacency matrix for user-item graph.
        
        This creates a symmetric adjacency matrix:
        [  0    R  ]  <- Users
        [ R^T   0  ]  <- Items
        
        Where R is the user-item interaction matrix.
        The matrix is symmetric (undirected graph) and sparse (efficient storage).
        
        Args:
            self_connection (bool): Whether to add self-loops (identity matrix)
        
        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix
                Shape: (num_users + num_items) x (num_users + num_items)
        """
        # Total nodes = users + items
        n_nodes = self.user_num + self.item_num
        
        # Get user indices for all training interactions
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        
        # Get item indices and offset by num_users (items start after users)
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        
        # All edges have weight 1.0
        ratings = np.ones_like(user_np, dtype=np.float32)
        
        # Create sparse matrix: user->item edges
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        
        # Make symmetric: add transpose for item->user edges
        adj_mat = tmp_adj + tmp_adj.T
        
        # Optional: add self-loops (diagonal = 1)
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """Create sparse user-item interaction matrix (not bipartite).
        
        This is the standard R matrix in collaborative filtering:
        Rows = users, Columns = items, Values = interactions (1 for implicit)
        
        Returns:
            scipy.sparse.csr_matrix: Shape (num_users x num_items)
        """
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    # =========================================================================
    # Utility Methods for Data Access
    # =========================================================================
    
    def get_user_id(self, u):
        """Get integer ID for user string."""
        return self.user.get(u)

    def get_item_id(self, i):
        """Get integer ID for item string."""
        return self.item.get(i)

    def training_size(self):
        """Get training set statistics.
        
        Returns:
            tuple: (num_users, num_items, num_interactions)
        """
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        """Get test set statistics.
        
        Returns:
            tuple: (num_users, num_items_in_test, num_interactions)
        """
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        """Check if user u has interacted with item i in training set."""
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        """Check if user exists in training data."""
        return u in self.user

    def contain_item(self, i):
        """Check if item exists in training data."""
        return i in self.item

    def user_rated(self, u):
        """Get all items and ratings for a user.
        
        Returns:
            tuple: (list of items, list of ratings)
        """
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        """Get all users and ratings for an item.
        
        Returns:
            tuple: (list of users, list of ratings)
        """
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
