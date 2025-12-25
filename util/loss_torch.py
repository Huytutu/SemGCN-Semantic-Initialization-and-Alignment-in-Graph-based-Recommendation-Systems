# =============================================================================
# PyTorch Loss Functions for Recommendation
# =============================================================================
# Collection of loss functions commonly used in recommendation systems:
# - BPR Loss: Bayesian Personalized Ranking (pairwise ranking)
# - Triplet Loss: Distance-based ranking
# - L2 Regularization: Prevents overfitting
# - InfoNCE: Contrastive learning loss
# - Softmax variants: For multi-class classification
# =============================================================================

import torch
import torch.nn.functional as F
import torch.nn as nn


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking (BPR) Loss.
    
    BPR optimizes for the ranking order: positive items should be ranked
    higher than negative items for each user.
    
    Formula: -log(σ(score_pos - score_neg))
    Where σ is sigmoid function, and scores are dot products.
    
    This encourages: score(user, positive_item) > score(user, negative_item)
    
    Args:
        user_emb: User embeddings (batch_size x emb_dim)
        pos_item_emb: Positive item embeddings (batch_size x emb_dim)
        neg_item_emb: Negative item embeddings (batch_size x emb_dim)
    
    Returns:
        torch.Tensor: Mean BPR loss across the batch
    """
    # Compute positive scores: element-wise multiplication + sum
    # Shape: (batch_size,)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    
    # Compute negative scores
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    
    # BPR loss: -log(sigmoid(pos - neg))
    # The 10e-6 prevents log(0) which would be -inf
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    
    # Return mean loss
    return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    """Triplet Loss with margin.
    
    Unlike BPR which uses log-sigmoid, triplet loss uses hinge loss:
    max(0, distance(user, pos) - distance(user, neg) + margin)
    
    This encourages: distance(user, positive) < distance(user, negative) - margin
    
    Args:
        user_emb: User embeddings (batch_size x emb_dim)
        pos_item_emb: Positive item embeddings (batch_size x emb_dim)
        neg_item_emb: Negative item embeddings (batch_size x emb_dim)
    
    Returns:
        torch.Tensor: Mean triplet loss with margin=0.5
    """
    # Euclidean distance squared to positive items
    pos_score = ((user_emb-pos_item_emb)**2).sum(dim=1)
    
    # Euclidean distance squared to negative items
    neg_score = ((user_emb-neg_item_emb)**2).sum(dim=1)
    
    # Hinge loss with margin 0.5: max(0, pos_dist - neg_dist + 0.5)
    loss = F.relu(pos_score-neg_score+0.5)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    """L2 Regularization Loss.
    
    Computes the L2 norm (Euclidean norm) of embeddings and scales by
    regularization coefficient. This prevents overfitting by penalizing
    large embedding values.
    
    Formula: reg * Σ ||emb||_2 / batch_size
    
    Args:
        reg (float): Regularization coefficient (typically 0.0001)
        *args: Variable number of embedding tensors to regularize
    
    Returns:
        torch.Tensor: Weighted sum of L2 norms
    """
    emb_loss = 0
    for emb in args:
        # torch.norm(emb, p=2) computes: sqrt(sum(emb^2))
        # Divide by batch size to normalize
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """InfoNCE Loss for Contrastive Learning.
    
    InfoNCE (Information Noise-Contrastive Estimation) encourages agreement
    between augmented views of the same data while pushing apart different samples.
    
    Used in self-supervised learning where view1 and view2 are different
    augmentations of the same data.
    
    Formula: -log( exp(sim(v1_i, v2_i)/τ) / Σ_j exp(sim(v1_i, v2_j)/τ) )
    
    Args:
        view1: First view embeddings (N x D)
        view2: Second view embeddings (N x D)
        temperature: Temperature parameter for softmax (typical: 0.1-1.0)
        b_cos: If True, normalize embeddings (use cosine similarity)
    
    Returns:
        torch.Tensor: Average InfoNCE loss
    """
    # Optional: normalize to use cosine similarity instead of dot product
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    # Compute similarity matrix: view1 @ view2.T
    # Divide by temperature to control distribution sharpness
    pos_score = (view1 @ view2.T) / temperature
    
    # Extract diagonal: similarity between matching pairs (positive pairs)
    # Apply log-softmax over rows: log(exp(x_i) / sum(exp(x_j)))
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    
    # Return negative mean (we want to maximize similarity)
    return -score.mean()


#this version is from recbole
def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

