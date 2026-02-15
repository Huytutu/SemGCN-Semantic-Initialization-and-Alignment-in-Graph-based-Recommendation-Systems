# =============================================================================
# LightGCN: Light Graph Convolution Network for Recommendation
# =============================================================================
# Paper: LightGCN: Simplifying and Powering Graph Convolution Network 
#        for Recommendation. SIGIR'20
# 
# Key Ideas:
# 1. Remove feature transformation and nonlinear activation from GCN
# 2. Only keep neighborhood aggregation for collaborative filtering
# 3. Combine embeddings from all layers via simple averaging
# 
# Formula: e_u^(l+1) = Σ(v∈N_u) [1/√(|N_u||N_v|)] * e_v^(l)
#          Final: e_u = (1/(L+1)) * Σ(l=0 to L) e_u^(l)
# =============================================================================

import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss


class LightGCN(GraphRecommender):
    """LightGCN implementation for collaborative filtering.
    """
    def __init__(self, conf, training_set, test_set):
        """Initialize LightGCN model.
        
        Args:
            conf: Configuration object with hyperparameters
            training_set: Training interaction data
            test_set: Test interaction data for evaluation
        """
        # Call parent class constructor to initialize common attributes
        super(LightGCN, self).__init__(conf, training_set, test_set)
        
        # Load model-specific parameters from config
        args = self.config['LightGCN']
        
        # Number of graph convolution layers (typically 2-4)
        self.n_layers = int(args['n_layer'])
        
        # Initialize the LightGCN encoder (graph neural network)
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        """Train the LightGCN model using BPR loss.
        
        Training process:
        1. Sample user-item-negative triplets in batches
        2. Forward pass through graph convolution layers
        3. Compute BPR loss (positive items ranked higher than negative)
        4. Add L2 regularization on embeddings
        5. Backprop and update parameters
        6. Evaluate every 5 epochs
        """
        # Move model to GPU for faster computation
        model = self.model.cuda()
        
        # Adam optimizer with learning rate from config
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        threshold = float(self.config['LightGCN'].get('rating_threshold', 3.0))
        
        # Training loop over epochs
        for epoch in range(self.maxEpoch):
            # Iterate over batches of training data
            # Each batch contains (user_idx, pos_item_idx, neg_item_idx) triplets
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, 
                                              rating_threshold=threshold)):
                user_idx, pos_idx, neg_idx = batch
                
                # Forward pass: propagate embeddings through graph layers
                rec_user_emb, rec_item_emb = model()
                
                # Extract embeddings for current batch
                user_emb = rec_user_emb[user_idx]          # User embeddings
                pos_item_emb = rec_item_emb[pos_idx]       # Positive item embeddings
                neg_item_emb = rec_item_emb[neg_idx]       # Negative item embeddings
                
                # Compute loss = BPR loss + L2 regularization
                # BPR: encourages positive items to rank higher than negative items
                # L2 reg: prevents overfitting by penalizing large embedding values
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + \
                            l2_reg_loss(self.reg, 
                                       model.embedding_dict['user_emb'][user_idx],
                                       model.embedding_dict['item_emb'][pos_idx],
                                       model.embedding_dict['item_emb'][neg_idx]) / self.batch_size
                
                # Backward pass and parameter update
                optimizer.zero_grad()      # Clear previous gradients
                batch_loss.backward()      # Compute gradients
                optimizer.step()           # Update parameters
                
                # Print progress every 100 batches
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            
            # After each epoch, compute final embeddings (no gradients needed)
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            
            # Evaluate on test set every 5 epochs
            if epoch % 5 == 0:
                _, should_stop = self.fast_evaluation(epoch)
                if should_stop:
                    print(f'Stopping training at epoch {epoch + 1}')
                    break
        
        # Use best embeddings from training (saved during fast_evaluation)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb



    def save(self):
        """Save the best model embeddings.
        
        Called by fast_evaluation() when a new best performance is achieved.
        Stores the current user and item embeddings for later use.
        """
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        """Generate recommendation scores for a given user.
        
        Args:
            u (str): User ID string
            
        Returns:
            numpy.array: Scores for all items (higher = more recommended)
        """
        # Convert user string ID to internal integer ID
        u = self.data.get_user_id(u)
        
        # Compute scores as dot product between user and all item embeddings
        # Shape: (1 x emb_dim) @ (emb_dim x num_items) = (1 x num_items)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        
        # Convert to numpy and return
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    """LightGCN Graph Encoder - Implements the graph convolution layers.
    
    This encoder performs neighborhood aggregation through message passing:
    1. Initialize random embeddings for users and items
    2. Propagate embeddings through L layers using normalized adjacency matrix
    3. Combine all layer embeddings by averaging
    
    The key operation is: e^(l+1) = (D^-1/2 A D^-1/2) @ e^(l)
    Where A is the user-item bipartite adjacency matrix
    """
    def __init__(self, data, emb_size, n_layers):
        """Initialize the encoder.
        
        Args:
            data: Interaction data object containing user-item graph
            emb_size: Dimensionality of embeddings (typically 64)
            n_layers: Number of graph convolution layers (typically 2-4)
        """
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        
        # Load pre-computed normalized adjacency matrix from data
        # This matrix encodes the graph structure with normalization weights
        self.norm_adj = data.norm_adj
        
        # Initialize learnable embeddings for users and items
        self.embedding_dict = self._init_model()
        
        # Convert scipy sparse matrix to PyTorch sparse tensor and move to GPU
        # This is the key matrix used in message passing: D^-1/2 @ A @ D^-1/2
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        """Initialize embedding parameters with Xavier uniform initialization.
        
        Xavier initialization helps gradient flow by keeping variance consistent
        across layers. Formula: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        
        Returns:
            nn.ParameterDict: Dictionary containing user and item embeddings
        """
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            # User embedding matrix: (num_users x embedding_dim)
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            # Item embedding matrix: (num_items x embedding_dim)
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        """Perform multi-layer graph convolution and return final embeddings.
        
        This is where the LightGCN magic happens:
        1. Start with initial embeddings (layer 0)
        2. For each layer, aggregate neighbor embeddings via matrix multiplication
        3. Combine all layers by averaging (layer combination)
        
        The key formula implemented here:
        e_u^(l+1) = Σ(v∈N_u) [1/√(|N_u||N_v|)] * e_v^(l)
        Final: e_u = (1/(L+1)) * Σ(l=0 to L) e_u^(l)
        
        Returns:
            tuple: (user_embeddings, item_embeddings) after graph convolution
        """
        # STEP 1: Concatenate user and item embeddings into one tensor
        # Shape: (num_users + num_items) x embedding_dim
        # This creates a single embedding matrix for all nodes in the bipartite graph
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], 
                                     self.embedding_dict['item_emb']], 0)
        
        # Store layer-0 embeddings (original embeddings before any propagation)
        all_embeddings = [ego_embeddings]
        
        # STEP 2: Message passing through k layers
        for k in range(self.layers):
            # THIS IS THE CORE MESSAGE CONSTRUCTION/PROPAGATION:
            # Multiply normalized adjacency matrix with current embeddings
            # This performs: e^(l+1) = (D^-1/2 @ A @ D^-1/2) @ e^(l)
            # 
            # What this does:
            # - For each user: aggregate embeddings from interacted items (weighted by normalization)
            # - For each item: aggregate embeddings from users who interacted (weighted by normalization)
            # - The normalization factor 1/√(|N_u||N_v|) is pre-computed in sparse_norm_adj
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            
            # Store embeddings from this layer
            all_embeddings += [ego_embeddings]
        
        # STEP 3: Layer combination - average all layer embeddings
        # Stack embeddings from all layers: (num_nodes x (L+1) x embedding_dim)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        
        # Compute mean across layers: (num_nodes x embedding_dim)
        # This implements: e_u = (1/(L+1)) * Σ(l=0 to L) e_u^(l)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # STEP 4: Split back into user and item embeddings
        user_all_embeddings = all_embeddings[:self.data.user_num]      # First N users
        item_all_embeddings = all_embeddings[self.data.user_num:]      # Remaining M items
        
        return user_all_embeddings, item_all_embeddings


