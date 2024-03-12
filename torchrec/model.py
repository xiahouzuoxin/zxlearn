import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam

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
            # layers.append(nn.PReLU())
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(hidden_unit))
        self.fc_layers = nn.Sequential(*layers)
        self.logits = nn.Linear(hidden_units[-1], 1)
        # self.out = nn.Sigmoid()

    def feature_embedding(self, f_config):
        if f_config['type'] == 'sparse':
            if 'emb_dim' not in f_config:
                raise ValueError('emb_dim must be specified for sparse features.')
            return nn.Embedding(len(f_config['vocab']), f_config['emb_dim'])
        elif f_config['type'] == 'dense' and f_config.get('buckets', None) is not None:
            print(f'==> Create embedding for dense feature {f_config["name"]}')
            if 'emb_dim' not in f_config:
                raise ValueError('emb_dim must be specified when buckets exist for dense features.')
            # why +2:
            # 1. buckets = [1,2,3] actually has 4 ranges (-inf,1], (1,2], (2,3], (3,+inf) 
            # 2. nan value will has an independent embedding
            return nn.Embedding(len(f_config['buckets']) + 2, f_config['emb_dim'])
        else:
            return None

    def forward(self, input_feats):
        sparse_inputs = []
        dense_inputs = []
        
        for k, f_config in enumerate(self.feat_configs):
            # print(f_config)
            f_name = f_config['name']
            f_type = f_config['type']
            if f_type == 'sparse':
                _input_feat = input_feats[f_name]
                # Lookup embeddings for each elements in th list
                embedded_list = []
                max_list_length = _input_feat.shape[1]  # Assuming input_feats[k] is a 2D tensor (batch_size, max_list_length)
                mask = (_input_feat >= 0).float()  
                _input_feat = _input_feat * mask.long()  # Set padding to 0 for embedding lookup, will be zeroed out after embedding lookup
                for idx in range(max_list_length):
                    masked_emb = self.embeddings[f_name](_input_feat[:, idx]) * mask[:, idx].unsqueeze(-1)
                    embedded_list.append(masked_emb)
                # sum pooling the embeddings of the list
                # embedded_tensor = sum(embedded_list)
                embedded_tensor = torch.stack(embedded_list, dim=1)
                embedded_tensor = embedded_tensor.sum(dim=1)
                sparse_inputs.append(embedded_tensor)
            else:
                pass

        # print(sparse_inputs[0].shape, sparse_inputs[1].shape, sparse_inputs[2].shape)
        if 'dense_features' in input_feats:
            dense_inputs = [input_feats['dense_features'], ]
        if 'seq_dense_features' in input_feats:
            dense_inputs += input_feats['seq_dense_features']
        
        x = torch.concat(sparse_inputs + dense_inputs, dim=-1)
        x = self.fc_layers(x)
        x = self.logits(x)

        return x
