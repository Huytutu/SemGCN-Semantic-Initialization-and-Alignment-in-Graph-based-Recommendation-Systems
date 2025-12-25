# =============================================================================
# Graph Utilities for GNN-based Recommendation
# =============================================================================
# Provides graph normalization functions for Graph Neural Networks:
# - Symmetric normalization: D^-1/2 @ A @ D^-1/2
# - Random walk normalization: D^-1 @ A
# - Laplacian matrix construction
# =============================================================================

import numpy as np
import scipy.sparse as sp


class Graph(object):
    """Utility class for graph operations."""
    
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        """Normalize adjacency matrix for graph convolution.
        
        This function performs symmetric normalization (for square matrices)
        or random walk normalization (for rectangular matrices).
        
        For square matrices (user-item bipartite graph):
            Compute: D^-1/2 @ A @ D^-1/2
            Where D is the degree matrix (diagonal)
            
            This is the normalization used in GCN and LightGCN.
            It ensures that messages from high-degree nodes don't dominate.
            
        For rectangular matrices:
            Compute: D^-1 @ A (row-wise normalization)
        
        Args:
            adj_mat (scipy.sparse matrix): Adjacency matrix to normalize
        
        Returns:
            scipy.sparse matrix: Normalized adjacency matrix
        """
        shape = adj_mat.get_shape()
        
        # Compute row sums (degree of each node)
        rowsum = np.array(adj_mat.sum(1))
        
        if shape[0] == shape[1]:
            # SYMMETRIC NORMALIZATION (for square matrices)
            # D^-1/2 @ A @ D^-1/2
            
            # Compute D^-1/2 (inverse square root of degrees)
            d_inv = np.power(rowsum, -0.5).flatten()
            
            # Handle isolated nodes (degree = 0) by setting to 0
            d_inv[np.isinf(d_inv)] = 0.
            
            # Create diagonal matrix D^-1/2
            d_mat_inv = sp.diags(d_inv)
            
            # Compute D^-1/2 @ A
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            
            # Compute D^-1/2 @ A @ D^-1/2 (symmetric normalization)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            # RANDOM WALK NORMALIZATION (for rectangular matrices)
            # D^-1 @ A (row-wise normalization)
            
            # Compute D^-1 (inverse of degrees)
            d_inv = np.power(rowsum, -1).flatten()
            
            # Handle isolated nodes
            d_inv[np.isinf(d_inv)] = 0.
            
            # Create diagonal matrix D^-1
            d_mat_inv = sp.diags(d_inv)
            
            # Compute D^-1 @ A
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        """Convert adjacency matrix to Laplacian matrix.
        
        Laplacian: L = D - A (unnormalized)
        Normalized Laplacian: L = I - D^-1/2 @ A @ D^-1/2
        
        Used in some spectral graph methods.
        
        Args:
            adj_mat: Adjacency matrix
        
        Returns:
            scipy.sparse matrix: Normalized Laplacian
        """
        # Placeholder - not fully implemented
        pass
