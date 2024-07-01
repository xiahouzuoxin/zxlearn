import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam

from core.embedding import DynamicEmbedding

class DNN(nn.Module):
    def __init__(self, feat_configs, hidden_units=[256, 128, 64]):
        super(DNN, self).__init__()
        self.feat_configs = feat_configs

        num_sparse = 0
        num_dense = 0
        self.embeddings = nn.ModuleDict()
        for f_config in self.feat_configs:
            emb = self.feature_embedding(f_config)
            if emb is not None:
                self.embeddings[f_config['name']] = emb
                num_sparse += f_config['emb_dim']
            elif ( f_config['type'] == 'dense' ) and f_config.get('islist'):
                num_dense += 3
            else:
                num_dense += 1
            # input_size += f_config['emb_dim'] if emb is not None else 1

        print(f'==> Model Input: dense_size={num_dense}, sparse_size={num_sparse}')
        
        # if num_dense > 0:
        #     self.dense_norm = nn.LayerNorm(num_dense)
        
        layers = []
        for k, hidden_unit in enumerate(hidden_units):
            if k == 0:
                layers.append(nn.Linear(num_sparse + num_dense, hidden_unit))
            else:
                layers.append(nn.Linear(hidden_units[k-1], hidden_unit))
            # layers.append(nn.BatchNorm1d(hidden_unit))
            # layers.append(nn.PReLU())
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
        self.fc_layers = nn.Sequential(*layers)
        self.logits = nn.Linear(hidden_units[-1], 1)
        # self.out = nn.Sigmoid()

    def feature_embedding(self, f_config):
        if f_config['type'] == 'sparse':
            if 'emb_dim' not in f_config:
                raise ValueError('emb_dim must be specified for sparse features.')
            # return nn.Embedding(f_config['num_embeddings'], f_config['emb_dim'])
            return DynamicEmbedding(f_config['num_embeddings'], f_config['emb_dim'])
        else:
            return None

    def target_attention(self, target_emb, candidate_embs):
        # target_emb: [B, E]
        # candidate_embs: [B, N, E]
        # return: [B, E]
        # candidate_embs = torch.stack(candidate_embs, dim=1)  # [B, N, E]
        attn_scores = torch.matmul(candidate_embs, target_emb.unsqueeze(-1)).squeeze(-1)  # [B, N, 1] * [B, 1, E] -> [B, N]
        attn_scores = F.softmax(attn_scores, dim=1)
        weighted_embs = attn_scores.unsqueeze(-1) * candidate_embs  # [B, N, 1] * [B, N, E] -> [B, N, E]
        weighted_emb = weighted_embs.sum(dim=1)  # [B, E]
        return weighted_emb

    def forward(self, input_feats):
        sparse_inputs = []
        dense_inputs = []
        
        # for k, f_config in enumerate(self.feat_configs):
        for f_name, f_emb in self.embeddings.items():
            _input_feat = input_feats[f_name]
            mask = (_input_feat >= 0).float()  
            _input_feat = _input_feat * mask.long()  # Set padding to 0 for embedding lookup, will be masked out later
            masked_emb = f_emb(_input_feat) * mask.unsqueeze(-1)
            embedded_tensor = masked_emb.sum(dim=1)
            sparse_inputs.append(embedded_tensor)

        # print(sparse_inputs[0].shape, sparse_inputs[1].shape, sparse_inputs[2].shape)
        if 'dense_features' in input_feats:
            dense_inputs = [input_feats['dense_features'], ]
        if 'seq_dense_features' in input_feats:
            dense_inputs += input_feats['seq_dense_features']
        
        x = torch.concat(sparse_inputs + dense_inputs, dim=-1)
        x = self.fc_layers(x)
        x = self.logits(x)

        return x
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss