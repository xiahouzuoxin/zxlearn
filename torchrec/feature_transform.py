
import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

def hash(v, hash_buckets):
    hash_object = hashlib.sha256(str(v).encode())
    hash_digest = hash_object.hexdigest()
    hash_integer = int(hash_digest, 16)
    return hash_integer % hash_buckets

def feature_transform(df, feat_configs, is_train=False):
    '''
    Feature transform. The format of `feat_configs`:
    [
        {'name': 'a', 'dtype': 'numerical', 'norm': 'std'}, # 'norm' in ['std','[0,1]']
        {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
        {'name': 'b', 'dtype': 'category', 'islist': False, 'emb_dim': 8},
        {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8},
    ]
    '''
    if is_train:
        print(f'==> Feature transforming (is_train={is_train}), note that feat_configs will be updated when is_train=True...')
    else:
        print(f'==> Feature transforming (is_train={is_train}) ...')

    def process_category(feat_config, s, is_train):
        name = feat_config['name']
        oov = feat_config.get('oov', 'other') # out of vocabulary
        s = s.replace(
            ['','None','none','nan','NaN','NAN','NaT','unknown','Unknown','Other','other','others','Others','REJ','Reject','REJECT','Rejected'], 
            np.nan).fillna(oov).map( lambda x: str(int(x) if type(x) is float else x) )
        s = s.astype(str).str.lower()
        
        hash_buckets = feat_config.get('hash_buckets')
        if hash_buckets:
            s = s.map(lambda x: hash(x, hash_buckets))
        
        if is_train and ('vocab' not in feat_config):
            feat_config['type'] = 'sparse'

            # generate vocab
            raw_vocab = s.value_counts()
            min_freq = feat_config.get('min_freq', 3)
            if min_freq:
                raw_vocab = raw_vocab[raw_vocab >= min_freq]
            feat_config['vocab'] = {v: {'idx': k, 'freq_cnt': freq_cnt} for k, (v, freq_cnt) in enumerate(raw_vocab.items())}
            if oov not in feat_config['vocab']:
                feat_config['vocab'][oov] = {'idx': len(raw_vocab), 'freq_cnt': 0}

        # convert to indices
        oov_index = feat_config['vocab'].get(oov)
        s = s.map(lambda x: feat_config['vocab'].get(x, oov_index)['idx']).astype(int)

        return s, feat_config

    def process_numerical(feat_config, s, is_train):
        import numpy as np
        hash_buckets = feat_config.get('hash_buckets', None)
        emb_dim = feat_config.get('emb_dim', None)
        discretization = True if (hash_buckets or emb_dim) else False
        normalize = feat_config.get('norm', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (discretization and normalize), f'hash_buckets/emb_dim and norm cannot be set at the same time: {feat_config}'

        if is_train:
            feat_config['type'] = 'sparse' if discretization else 'dense'
            if 'mean' not in feat_config:
                feat_config['mean'] = s.mean()
            if 'std' not in feat_config:
                feat_config['std'] = s.std()
            if 'min' not in feat_config:
                feat_config['min'] = s.min()
            if 'max' not in feat_config:
                feat_config['max'] = s.max()
            if discretization:
                hash_buckets = 10 if hash_buckets is None else hash_buckets

                # feat_config['buckets'] = list(
                #     np.linspace(feat_config['min'], feat_config['max'], num=hash_buckets)
                # )
                bins = np.percentile(s[s.notna()], q=np.linspace(0, 100, num=hash_buckets))
                non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                feat_config['vocab'] = list( bins[non_adjacent_duplicates] )
                
                # 1. buckets = [1,2,3] actually has 4 ranges (-inf,1], (1,2], (2,3], (3,+inf) 
                # 2. nan value will has an independent embedding
                feat_config['vocab'] = [np.NaN, float('-inf'), ] + feat_config['vocab'] + [float('inf'), ]

        if normalize == 'std':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['mean']) / feat_config['std']
        elif normalize == '[0,1]':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif discretization:
            # print(feat_config)
            bins = [v for v in feat_config['vocab'] if not np.isnan(v)]
            s = pd.cut(s, bins=bins, labels=False, right=True) + 1
            s = s.fillna(0).astype(int) # index 0 is for nan values

        return s, feat_config
    
    def process_list(feat_configs, s, is_train):
        dtype = feat_configs['dtype']
        flat_s = s.explode()
        if dtype == 'category':
            flat_s, updated_f = process_category(feat_configs, flat_s, is_train)
        elif dtype == 'numerical':
            flat_s, updated_f = process_numerical(feat_configs, flat_s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')
        s = flat_s.groupby(level=0).agg(list)
        return s, updated_f

    # transform features
    for k, f in enumerate( feat_configs ):
        fname = f['name']
        dtype = f['dtype']
        islist = f.get('islist', None)
        
        print(f'Processing feature {fname}...')
        if islist:
            df[fname], updated_f = process_list(f, df[fname], is_train)
        elif dtype == 'category':
            df[fname], updated_f = process_category(f, df[fname], is_train)
        elif dtype == 'numerical':
            df[fname], updated_f = process_numerical(f, df[fname], is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        if is_train:
            # update feat config
            feat_configs[k] = updated_f

    return df

class DataFrameDataset(Dataset):
    '''
    Var length sequence input. Usage example,
        train_dataset = DataFrameDataset(df, 
            dense_cols=['a','b'], sparse_cols=['c','d'], seq_sparse_cols=['e','f'], target_cols='label', 
            df_transform=lambda df: feature_transform(df,feat_configs,is_train=True)
        )
        test_dataset = DataFrameDataset(df, 
            dense_cols=['a','b'], sparse_cols=['c','d'], seq_sparse_cols=['e','f'], target_cols='label', 
            df_transform=lambda df: feature_transform(df,feat_configs,is_train=False)
        )
    '''
    def __init__(self, df, sparse_cols, seq_sparse_cols, dense_cols, seq_dense_cols=None, target_cols=None, padding_value=-100, df_transform=None):
        if df_transform:
            df = df_transform(df)

        self.dense_cols = dense_cols
        self.seq_dense_cols = seq_dense_cols
        self.sparse_cols = sparse_cols
        self.seq_sparse_cols = seq_sparse_cols
        self.target_cols = target_cols

        self.total_samples = len(df)
        self.padding_value = padding_value

        self.convert_to_tensors(df)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        features = {}
        if hasattr(self, 'dense_data'):
            features.update( {'dense_features': self.dense_data[idx, :]} )
        if hasattr(self, 'seq_dense_features'):
            features.update( {'seq_dense_features': self.seq_dense_data[idx, :]} )
        if hasattr(self, 'sparse_data'):
            features.update({f'{k}': v[idx,:] for k,v in self.sparse_data.items()})
        if hasattr(self, 'seq_sparse_data'):
            features.update({f'{k}': v[idx,:] for k,v in self.seq_sparse_data.items()})

        return features, self.target[idx,:] if hasattr(self, 'target') else None

    def convert_to_tensors(self, df):
        if self.dense_cols is not None and len(self.dense_cols) > 0:
            self.dense_data = torch.tensor(df[self.dense_cols].values, dtype=torch.float32)
        
        if self.seq_dense_cols is not None and len(self.seq_dense_cols) > 0: 
            self.seq_dense_data = torch.tensor( 
                df[self.seq_dense_cols].applymap(lambda x: [np.nansum(x), np.nanmax(x), np.nanmin(x)]).values.tolist(), 
                dtype=torch.float32
            )
            self.seq_dense_data = self.seq_dense_data.view(len(df), -1) # num x dim_feature

        if self.sparse_cols is not None and len(self.sparse_cols) > 0:
            self.sparse_data = {}
            for col in self.sparse_cols:
                self.sparse_data[col] = torch.tensor(df[[col]].values, dtype=torch.int)
        
        if self.seq_sparse_cols is not None and len(self.seq_sparse_cols) > 0:
            # for sparse sequences, padding to the maximum length
            self.seq_sparse_data = {}
            df_seq_sparse = df[self.seq_sparse_cols].copy()
            for col in self.seq_sparse_cols:
                max_len = df_seq_sparse[col].apply(len).max()
                df_seq_sparse[col] = df_seq_sparse[col].apply( lambda x: [self.padding_value] * (max_len - len(x)) + x)
                self.seq_sparse_data[col] = torch.tensor(df_seq_sparse[col].values.tolist(), dtype=torch.int)
        
        # print(df_seq_sparse.head())
        # self.seq_sparse_data = torch.tensor(df_seq_sparse.values.tolist(), dtype=torch.int)

        if self.target_cols is not None:
            self.target = torch.tensor(df[self.target_cols].values.tolist(), dtype=torch.float32)
    
    def to(self, device):
        if hasattr(self, 'dense_data'):
            self.dense_data = self.dense_data.to(device)
        if hasattr(self, 'seq_dense_data'):
            self.seq_dense_data = self.seq_dense_data.to(device)
        if hasattr(self, 'sparse_data'):
            self.sparse_data = {k: v.to(device) for k,v in self.sparse_data.items()}
        if hasattr(self, 'seq_sparse_data'):
            self.seq_sparse_data = {k: v.to(device) for k,v in self.seq_sparse_data.items()}
        if hasattr(self, 'target'):
            self.target = self.target.to(device)

        return self
