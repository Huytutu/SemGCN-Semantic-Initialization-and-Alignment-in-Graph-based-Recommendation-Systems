# =============================================================================
# Matrix Factorization (MF) for Collaborative Filtering
# =============================================================================
# Paper: Matrix Factorization Techniques for Recommender Systems (IEEE, 2009)
#
# Classic collaborative filtering approach that learns latent representations
# for users and items. Unlike LightGCN, MF does NOT use graph structure -
# it directly learns embeddings from interactions.
#
# Model: score(user, item) = user_embedding · item_embedding (dot product)
# Training: BPR loss (positive items ranked higher than negative items)
#
# MF is simpler than LightGCN but often less effective because it doesn't
# leverage the graph neighborhood information.
# =============================================================================

import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss


class MF(GraphRecommender):
    """Matrix Factorization model for collaborative filtering.
    
    This is the classic MF model that learns user and item embeddings
    without using any graph structure. Predictions are simple dot products
    between user and item vectors.
    
    Compared to LightGCN:
    - Simpler: No graph convolution, just lookup embeddings
    - Faster: No message passing overhead
    - Less effective: Doesn't use neighborhood information
    \"\"\"
    def __init__(self, conf, training_set, test_set):
        """Initialize Matrix Factorization model.
        
        Args:
            conf: Configuration object
            training_set: Training interactions
            test_set: Test interactions
        """
        super(MF, self).__init__(conf, training_set, test_set)
        # Initialize the MF encoder (just embedding lookup, no graph ops)
        self.model = Matrix_Factorization(self.data, self.emb_size)

    def train(self):
        """Train the MF model using BPR loss.
        
        Training is similar to LightGCN but WITHOUT graph convolution:
        1. Sample (user, pos_item, neg_item) triplets
        2. Lookup embeddings directly (no message passing)
        3. Compute BPR loss + L2 regularization
        4. Update embeddings via backprop
        """
        # Move model to GPU
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        # Training loop
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                
                # Forward: Simply return the embeddings (no graph propagation)
                rec_user_emb, rec_item_emb = model()
                
                # Extract embeddings for this batch
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]
                
                # Compute loss: BPR + L2 regularization
                # Note: L2 is applied directly to the batch embeddings (not base embeddings)
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + \
                            l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size
                
                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # Log progress
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            
            # Get final embeddings for evaluation
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        
        # Use best embeddings from training
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        """Save the best model embeddings."""
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        """Generate recommendation scores for a user.
        
        Args:
            u (str): User ID
        
        Returns:
            numpy.array: Scores for all items
        """
        with torch.no_grad():
            u = self.data.get_user_id(u)
            # Score = dot product between user and all items
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class Matrix_Factorization(nn.Module):
    """The core MF embedding module.
    
    This is much simpler than LightGCN's encoder:
    - No graph structure
    - No message passing
    - Just learnable embedding lookup tables
    
    The embeddings are learned purely from the BPR loss during training.
    """
    def __init__(self, data, emb_size):
        """Initialize embedding matrices.
        
        Args:
            data: Data object with user/item counts
            emb_size: Embedding dimensionality
        """
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        # Create user and item embedding parameters
        self.embedding_dict = self._init_model()

    def _init_model(self):
        """Initialize embeddings with Xavier uniform.
        
        Returns:
            nn.ParameterDict: User and item embeddings
        """
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            # User embedding matrix: (num_users x emb_dim)
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            # Item embedding matrix: (num_items x emb_dim)
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        """Return user and item embeddings directly.
        
        Unlike LightGCN, there's no graph convolution here.
        Just return the raw embeddings.
        
        Returns:
            tuple: (user_embeddings, item_embeddings)
        """
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']


