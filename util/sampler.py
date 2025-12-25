# =============================================================================
# Data Samplers for Recommendation Models
# =============================================================================
# This module provides batch generation functions for different training paradigms:
# - Pairwise: For BPR loss (user, positive item, negative item)
# - Pointwise: For binary classification (user, item, label)
# - Sequence: For sequential models (padded sequences)
# =============================================================================

from random import shuffle,randint,choice,sample
import numpy as np


def next_batch_pairwise(data, batch_size, n_negs=1, rating_threshold=3.0):
    """Generate batches for pairwise ranking (BPR training).
    
    For each positive user-item interaction (rating >= threshold), 
    samples n_negs negative items (rating < threshold or unrated).
    
    Args:
        data: Data object containing training_data and item mappings
        batch_size: Number of positive interactions per batch
        n_negs: Number of negative samples per positive (default=1)
        rating_threshold: Minimum rating to be considered positive (default=3.0)
    
    Yields:
        tuple: (user_indices, pos_item_indices, neg_item_indices)
    """
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ratings = [training_data[idx][2] for idx in range(ptr, batch_end)]
        ptr = batch_end
        
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        
        for i, user in enumerate(users):
            # Only use as positive if rating >= threshold
            if ratings[i] >= rating_threshold:
                i_idx.append(data.item[items[i]])
                u_idx.append(data.user[user])
                
                # Sample n_negs negative items
                for m in range(n_negs):
                    neg_item = choice(item_list)
                    
                    # Resample until we find an item NOT in user's positive set
                    while neg_item in data.training_set_u[user]:
                        neg_item = choice(item_list)
                    
                    j_idx.append(data.item[neg_item])
        
        if u_idx:  # Only yield if batch is non-empty
            yield u_idx, i_idx, j_idx


def next_batch_pointwise(data,batch_size):
    """Generate batches for pointwise training (binary classification).
    
    For each positive interaction, generates 4 negative samples.
    Each sample is labeled: 1 for positive, 0 for negative.
    
    This is used for models that predict interaction probability
    rather than ranking.
    
    Args:
        data: Data object containing training_data and item mappings
        batch_size: Number of positive interactions per batch
    
    Yields:
        tuple: (user_indices, item_indices, labels)
            user_indices: List of user IDs (size: batch_size * 5)
            item_indices: List of item IDs (size: batch_size * 5)
            labels: List of 0/1 labels (size: batch_size * 5)
    """
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    
    while ptr < data_size:
        # Determine batch boundaries
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        
        # Extract batch
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        
        u_idx, i_idx, y = [], [], []
        
        # For each positive interaction
        for i, user in enumerate(users):
            # Add positive sample
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)  # Label = 1 (positive)
            
            # Add 4 negative samples for this user
            for instance in range(4):
                # Sample random item
                item_j = randint(0, data.item_num - 1)
                
                # Resample if in user's training set
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                
                # Add negative sample
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)  # Label = 0 (negative)
        
        yield u_idx, i_idx, y

def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = [item[1] for item in data.original_seq]
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=int)
        y =np.zeros((batch_end-ptr, max_len),dtype=int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end
        yield seq, pos, y, neg, np.array(seq_len,int)

def next_batch_sequence_for_test(data, batch_size,max_len=50):
    sequences = [item[1] for item in data.original_seq]
    ptr = 0
    data_size = len(sequences)
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(sequences[ptr + n]) > max_len and -max_len or 0
            end =  len(sequences[ptr + n]) > max_len and max_len or len(sequences[ptr + n])
            seq[n, :end] = sequences[ptr + n][start:]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
        ptr=batch_end
        yield seq, pos, np.array(seq_len,int)