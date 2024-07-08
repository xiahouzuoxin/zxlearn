import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..embedding import DynamicEmbedding

class DNN(nn.Module):
    def __init__(self, feat_configs, hidden_units=[256, 128, 64]):
        super(DNN, self).__init__()
        self.feat_configs = feat_configs

        num_sparse = 0
        num_dense = 0
        self.embeddings = nn.ModuleDict()
        for f_config in self.feat_configs:
            if f_config['type'] == 'sparse':
                if 'emb_dim' not in f_config:
                    raise ValueError('emb_dim must be specified for sparse features.')
                self.embeddings[f_config['name']] = nn.Embedding(f_config['num_embeddings'], f_config['emb_dim'])
                # self.embeddings[f_config['name']] = DynamicEmbedding(f_config['num_embeddings'], f_config['emb_dim'])
                num_sparse += f_config['emb_dim']
            elif ( f_config['type'] == 'dense' ) and f_config.get('islist'):
                num_dense += 3      # default 3 dense features for seq_dense_features
            elif f_config['type'] == 'dense':
                num_dense += 1
            else:
                raise ValueError(f'Unsupported feature type: {f_config["type"]}')

        print(f'==> Model Input: dense_size={num_dense}, sparse_size={num_sparse}')

        self.tower = self.get_tower(num_sparse + num_dense, hidden_units)

    def get_tower(self, input_dim, hidden_units):
        layers = []
        for k, hidden_unit in enumerate(hidden_units):
            if k == 0:
                layers.append(nn.Linear(input_dim, hidden_unit))
            else:
                layers.append(nn.Linear(hidden_units[k-1], hidden_unit))
            layers.append(nn.BatchNorm1d(hidden_unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
        layers.append( nn.Linear(hidden_units[-1], 1) )
        return nn.Sequential(*layers)

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
        x = self.tower(x)

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