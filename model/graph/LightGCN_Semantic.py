# =============================================================================
# LightGCN with Semantic Initialization & Alignment (LLM4Rec-style)
# =============================================================================
# EXTENDS existing LightGCN & LGCN_Encoder with:
#   1. Semantic Warm-Start: Item embeddings initialized from BERT encodings
#   2. Alignment Loss: Constrains collaborative embeddings ≈ semantic embeddings
#
# Total Loss = BPR_Loss + λ * Alignment_Loss
# =============================================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# REUSE existing LightGCN components
from model.graph.LightGCN import LightGCN, LGCN_Encoder
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss


# =============================================================================
# PART 1: EXTENDED ENCODER (Adds Semantic Support to LGCN_Encoder)
# =============================================================================

class LGCN_Semantic_Encoder(LGCN_Encoder):
    """Extended LGCN_Encoder with semantic warm-start and alignment support.
    
    Inherits all LightGCN graph convolution logic from LGCN_Encoder.
    Adds:
    - Item embedding warm-start from BERT semantic vectors
    - Stores frozen semantic embeddings for alignment loss
    """
    
    def __init__(self, data, emb_size, n_layers, item_semantic_emb=None):
        # Call parent to initialize standard LightGCN encoder
        super().__init__(data, emb_size, n_layers)
        
        # Store frozen semantic embeddings for alignment loss
        if item_semantic_emb is not None:
            self.register_buffer('item_semantic_emb', item_semantic_emb)
            
            # WARM-START: Override item embeddings with semantic vectors
            with torch.no_grad():
                self.embedding_dict['item_emb'].copy_(item_semantic_emb)
            print(f"✓ Item embeddings warm-started from BERT semantic vectors")
        else:
            self.register_buffer('item_semantic_emb', None)
            print("✗ No semantic embeddings, using random initialization")


# =============================================================================
# PART 2: TEXT ENCODER (Offline BERT Encoding)
# =============================================================================

class TextEncoder:
    """BERT-based Text Encoder for semantic embeddings (OFFLINE).
    
    Encodes item descriptions into fixed semantic vectors.
    These vectors are FROZEN during LightGCN training.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 output_dim=64, device='cuda'):
        self.device = device
        self.output_dim = output_dim
        
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading BERT: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.bert.eval()
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projector = nn.Linear(self.bert.config.hidden_size, output_dim).to(device)
        print(f"TextEncoder ready: {self.bert.config.hidden_size}D → {output_dim}D")
    
    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=256):
        """Encode texts into semantic embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=max_length, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.bert(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            # Clamp to avoid division by zero for empty sequences
            mask_sum = attention_mask.sum(1).clamp(min=1e-9)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / mask_sum
            embeddings = self.projector(embeddings)
            all_embeddings.append(embeddings.cpu())
            
            if (i // batch_size + 1) % 50 == 0:
                print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")
        
        return torch.cat(all_embeddings, dim=0)


# =============================================================================
# PART 3: ALIGNMENT LOSS (Mutual Regularization)
# =============================================================================

def alignment_loss(collab_emb, semantic_emb, loss_type='mse'):
    """Alignment loss: constrains collaborative embeddings ≈ semantic embeddings."""
    if loss_type == 'mse':
        return F.mse_loss(collab_emb, semantic_emb)
    elif loss_type == 'cosine':
        return (1 - F.cosine_similarity(collab_emb, semantic_emb, dim=1)).mean()
    return F.mse_loss(collab_emb, semantic_emb)


# =============================================================================
# PART 4: MAIN MODEL CLASS (Extends LightGCN)
# =============================================================================

class LightGCN_Semantic(LightGCN):
    """LightGCN with Semantic Initialization & Alignment.
    
    EXTENDS base LightGCN with:
    - Semantic warm-start for item embeddings
    - Alignment loss during training
    
    Inherits: save(), predict() from LightGCN
    Overrides: __init__(), train()
    """
    
    def __init__(self, conf, training_set, test_set):
        # Initialize parent (but we'll override self.model)
        # Note: We call GraphRecommender.__init__ directly to avoid 
        # LightGCN creating its own LGCN_Encoder
        from base.graph_recommender import GraphRecommender
        GraphRecommender.__init__(self, conf, training_set, test_set)
        
        # Load semantic-specific config
        args = self.config['LightGCN_Semantic']
        self.n_layers = int(args.get('n_layer', 2))
        # Note: self.reg is already set by parent class from reg.lambda in config
        self.lambda_align = float(args.get('lambda_align', 0.1))
        self.alignment_type = args.get('alignment_type', 'mse')
        self.bert_model = args.get('bert_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.semantic_data_path = args.get('semantic_data_path', None)
        
        print("=" * 60)
        print("LightGCN_Semantic: Extending LightGCN with Semantic Alignment")
        print("=" * 60)
        
        # Prepare semantic embeddings (OFFLINE STAGE)
        item_semantic_emb = self._prepare_semantic_embeddings()
        
        # Initialize EXTENDED encoder (replaces parent's LGCN_Encoder)
        self.model = LGCN_Semantic_Encoder(
            data=self.data,
            emb_size=self.emb_size,
            n_layers=self.n_layers,
            item_semantic_emb=item_semantic_emb
        )
        
        # Load rating threshold for sampling (consistent with LightGCN)
        self.rating_threshold = float(args.get('rating_threshold', 3.0))
        
        print(f"\nModel Config:")
        print(f"  - Users: {self.data.user_num}, Items: {self.data.item_num}")
        print(f"  - Embedding: {self.emb_size}D, Layers: {self.n_layers}")
        print(f"  - Alignment: λ={self.lambda_align}, type={self.alignment_type}")
    
    def _prepare_semantic_embeddings(self):
        """Load or compute semantic embeddings for items."""
        cache_path = f'dataset/semantic_cache/item_emb_{self.emb_size}d.pt'
        
        if os.path.exists(cache_path):
            print(f"Loading cached semantic embeddings: {cache_path}")
            emb = torch.load(cache_path)
            if emb.shape[0] == self.data.item_num and emb.shape[1] == self.emb_size:
                return emb
            print(f"Cache mismatch (got {emb.shape}, need ({self.data.item_num}, {self.emb_size})), recomputing...")
        
        item_texts = self._load_item_texts()
        if item_texts is None:
            print("No semantic data available, using random init")
            return None
        
        print(f"\nEncoding {len(item_texts)} items with BERT...")
        encoder = TextEncoder(self.bert_model, self.emb_size, 
                             'cuda' if torch.cuda.is_available() else 'cpu')
        
        ordered_texts = []
        missing_count = 0
        for i in range(self.data.item_num):
            item_name = self.data.id2item.get(i, str(i))
            if item_name in item_texts:
                ordered_texts.append(item_texts[item_name])
            else:
                ordered_texts.append(f"Item {item_name}")
                missing_count += 1
        
        if missing_count > 0:
            print(f"WARNING: {missing_count}/{self.data.item_num} items missing text descriptions, using placeholders")
        
        item_emb = encoder.encode(ordered_texts)
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(item_emb, cache_path)
        print(f"Saved semantic embeddings: {cache_path}")
        
        return item_emb
    
    def _load_item_texts(self):
        """Load item descriptions from Video_Games metadata."""
        if not self.semantic_data_path or not os.path.exists(self.semantic_data_path):
            print(f"Semantic data not found: {self.semantic_data_path}")
            return None
        
        print(f"Loading item texts from: {self.semantic_data_path}")
        item_texts = {}
        
        parse_errors = 0
        with open(self.semantic_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    item_id = item.get('parent_asin', item.get('asin', ''))
                    title = item.get('title', '')
                    desc = item.get('description', '')
                    if isinstance(desc, list):
                        desc = ' '.join(desc)
                    text = f"{title}. {desc}".strip()
                    if item_id and text:
                        item_texts[item_id] = text
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    parse_errors += 1
                    if parse_errors <= 3:  # Only log first few errors
                        print(f"  Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        if parse_errors > 3:
            print(f"  ... and {parse_errors - 3} more parse errors")
        print(f"Loaded {len(item_texts)} item descriptions")
        return item_texts
    
    def save(self):
        """Save the best model embeddings.
        
        Called by fast_evaluation() when a new best performance is achieved.
        Stores the current user and item embeddings for later use.
        """
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
    
    def train(self):
        """Training Loop: BPR Loss + Alignment Loss.
        """
        print("\n" + "=" * 60)
        print("Training: BPR Loss + λ * Alignment Loss")
        print("=" * 60)
        
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        for epoch in range(self.maxEpoch):
            model.train()
            total_bpr, total_align, n_batch = 0, 0, 0
            
            # Compute GCN propagation ONCE per epoch (not per batch)
            # This is valid because we use base embeddings for L2 reg, not propagated ones
            user_emb_all, item_emb_all = model()
            
            for batch in next_batch_pairwise(self.data, self.batch_size, 
                                              rating_threshold=self.rating_threshold):
                user_idx, pos_idx, neg_idx = batch
                
                # Convert to tensors on correct device
                user_idx_t = torch.tensor(user_idx, device='cuda')
                pos_idx_t = torch.tensor(pos_idx, device='cuda')
                neg_idx_t = torch.tensor(neg_idx, device='cuda')
                
                # BPR Loss using pre-computed propagated embeddings
                u_emb = user_emb_all[user_idx_t]
                pos_emb = item_emb_all[pos_idx_t]
                neg_emb = item_emb_all[neg_idx_t]
                loss_bpr = bpr_loss(u_emb, pos_emb, neg_emb)
                
                # L2 Regularization on BASE embeddings (before propagation)
                loss_l2 = l2_reg_loss(self.reg, 
                                      model.embedding_dict['user_emb'][user_idx_t],
                                      model.embedding_dict['item_emb'][pos_idx_t],
                                      model.embedding_dict['item_emb'][neg_idx_t]) / self.batch_size
                
                # Alignment Loss on propagated embeddings
                loss_align = torch.tensor(0.0, device='cuda')
                if model.item_semantic_emb is not None:
                    all_items = torch.cat([pos_idx_t, neg_idx_t]).unique()
                    collab = item_emb_all[all_items]
                    semantic = model.item_semantic_emb[all_items].to(collab.device)
                    loss_align = alignment_loss(collab, semantic, self.alignment_type)
                
                # Total Loss = BPR + L2 + λ * Alignment
                loss = loss_bpr + loss_l2 + self.lambda_align * loss_align
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_bpr += loss_bpr.item()
                total_align += loss_align.item()
                n_batch += 1
                
                # Re-propagate after parameter update for next batch
                user_emb_all, item_emb_all = model()
            
            # Guard against empty epoch (n_batch=0)
            if n_batch > 0:
                print(f"Epoch {epoch+1}: BPR={total_bpr/n_batch:.4f}, "
                      f"Align={total_align/n_batch:.4f}")
            else:
                print(f"Epoch {epoch+1}: No valid batches (check rating_threshold)")
            
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            
            # Evaluate every 5 epochs (fast_evaluation handles best model tracking via save())
            if epoch % 5 == 0:
                _, should_stop = self.fast_evaluation(epoch)
                if should_stop:
                    print(f'Stopping training at epoch {epoch + 1}')
                    break
        
        # Use best embeddings from training (saved during fast_evaluation)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        print("\nTraining completed!")

