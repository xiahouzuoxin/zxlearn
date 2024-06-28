
import os
import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

class FeatureTransformer:
    def __init__(self, 
                 feat_configs, 
                 category_force_hash=False, 
                 category_upper_lower_sensitive=False,
                 numerical_update_stats=False,
                 list_padding_value=-100,
                 outliers_category=['','None','none','nan','NaN','NAN','NaT','unknown','Unknown','Other','other','others','Others','REJ','Reject','REJECT','Rejected'], 
                 outliers_numerical=[], 
                 verbose=False):
        """
        Feature transforming for both train and test dataset.
        Args:
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8}, # sequence feature
                ]
            is_train: bool, whether it's training dataset
            category_force_hash: bool, whether to force hash all category features, which will be useful for large category features and online learning scenario, only effective when is_train=True
            numerical_update_stats: bool, whether to update mean, std, min, max for numerical features, only effective when is_train=True
            outliers_category: list, outliers for category features
            outliers_numerical: list, outliers for numerical features
            verbose: bool, whether to print the processing details
            n_jobs: int, number of parallel jobs
        """
        self.feat_configs = feat_configs
        self.category_force_hash = category_force_hash
        self.numerical_update_stats = numerical_update_stats
        self.outliers_category = outliers_category
        self.outliers_numerical = outliers_numerical
        self.category_upper_lower_sensitive = category_upper_lower_sensitive
        self.list_padding_value = list_padding_value
        self.verbose = verbose

    def transform(self, df, is_train=False, n_jobs=1):
        """
        Transforms the DataFrame based on the feature configurations.
        Args:
            df: pandas DataFrame
        Returns:
            df: pandas DataFrame, transformed dataset
        """
        if self.verbose:
            print(f'==> Feature transforming (is_train={is_train}), note that feat_configs will be updated when is_train=True...')

        if n_jobs <= 1:
            for k, f in enumerate(self.feat_configs):
                df[f['name']], updated_f = self._transform_one(df[f['name']], f, is_train)
                if is_train:
                    self.feat_configs[k] = updated_f
            return df

        # parallel process features
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._transform_one)(df[f_config['name']], f_config, is_train) for f_config in self.feat_configs
        )

        # update df & feat_configs
        for k, (updated_s, updated_f) in zip(range(len(self.feat_configs)), results):
            df[updated_f['name']] = updated_s
            if is_train:
                self.feat_configs[k] = updated_f

        return df

    def hash(self, v, hash_buckets):
        """
        Hash function for category features.
        """
        # hash_object = hashlib.sha256(str(v).encode())
        # hash_digest = hash_object.hexdigest()
        # hash_integer = int(hash_digest, 16)

        hash_integer = murmurhash3_32(str(v), seed=42, positive=True)
        return hash_integer % hash_buckets

    def update_meanstd(self, s, his_freq_cnt=0, mean=None, std=None):
        """
        Update mean, std for numerical feature.
        If none, calculate from s, else update the value by new input data.
        """
        s_mean = s.mean()
        s_std = s.std()

        # update mean and std
        mean = s_mean if mean is None else (mean * his_freq_cnt + s_mean * len(s)) / (his_freq_cnt + len(s))
        std = s_std if std is None else np.sqrt((his_freq_cnt * (std ** 2) + len(s) * (s_std ** 2) + his_freq_cnt * (mean - s_mean) ** 2) / (his_freq_cnt + len(s)))

        return mean, std

    def update_minmax(self, s, min_val=None, max_val=None):
        """
        Update min, max for numerical feature.
        If none, calculate from s, else update the value by new input data.
        """
        s_min = s.min()
        s_max = s.max()

        # update min and max
        min_val = s_min if min_val is None else min(min_val, s_min)
        max_val = s_max if max_val is None else max(max_val, s_max)

        return min_val, max_val

    def process_category(self, feat_config, s, is_train=False):
        """
        Process category features.
        """
        name = feat_config['name']
        oov = feat_config.get('oov', 'other')  # out of vocabulary
        s = s.replace(self.outliers_category, np.nan).fillna(oov).map(lambda x: str(int(x) if type(x) is float else x))
        s = s.astype(str)
        if not self.category_upper_lower_sensitive:
            s = s.str.lower()

        hash_buckets = feat_config.get('hash_buckets')
        if self.category_force_hash and hash_buckets is None:
            hash_buckets = s.nunique()
            # Auto choose the experienced hash_buckets for embedding table to avoid hash collision
            if hash_buckets < 100:
                hash_buckets *= 10
            elif hash_buckets < 1000:
                hash_buckets = max(hash_buckets * 5, 1000)
            elif hash_buckets < 10000:
                hash_buckets = max(hash_buckets * 2, 5000)
            elif hash_buckets < 1000000:
                hash_buckets = max(hash_buckets, 20000)
            else:
                hash_buckets = hash_buckets // 10

            if self.verbose:
                print(f'Forcing hash category {name} with hash_buckets={hash_buckets}...')

            if is_train:
                feat_config['hash_buckets'] = hash_buckets

        if is_train:
            # update feat_config
            feat_config['type'] = 'sparse'

            # low frequency category filtering
            raw_vocab = s.value_counts()
            min_freq = feat_config.get('min_freq', None)
            if min_freq:
                raw_vocab = raw_vocab[raw_vocab >= min_freq]

        if hash_buckets:
            if self.verbose:
                print(f'Hashing category {name} with hash_buckets={hash_buckets}...')
            if is_train:
                # update feat_config
                feat_config['num_embeddings'] = hash_buckets
                if min_freq:
                    feat_config['vocab'] = {v: freq_cnt for k, (v, freq_cnt) in enumerate(raw_vocab.items())}

            if 'vocab' in feat_config:
                s = s.map(lambda x: x if x in feat_config['vocab'] else oov)
            s = s.map(lambda x: self.hash(x, hash_buckets)).astype(int)
        else:
            if self.verbose:
                print(f'Converting category {name} to indices...')
            if is_train:
                if 'vocab' not in feat_config:
                    feat_config['vocab'] = {}
                    new_start_idx = 0
                else:
                    new_start_idx = max([v['idx'] for v in feat_config['vocab'].values()]) + 1

                # update dynamic vocab (should combine with dynamic embedding module when online training)
                for k, (v, freq_cnt) in enumerate(raw_vocab.items()):
                    idx = new_start_idx + k
                    if v not in feat_config['vocab']:
                        feat_config['vocab'][v] = {'idx': idx, 'freq_cnt': freq_cnt}
                    else:
                        feat_config['vocab'][v]['freq_cnt'] += freq_cnt

                if oov not in feat_config['vocab']:
                    feat_config['vocab'][oov] = {'idx': len(raw_vocab), 'freq_cnt': 0}

                if self.verbose:
                    print(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

                feat_config['num_embeddings'] = len(feat_config['vocab'])

            # convert to indices
            oov_index = feat_config['vocab'].get(oov)
            s = s.map(lambda x: feat_config['vocab'].get(x, oov_index)['idx']).astype(int)

        return s, feat_config

    def process_numerical(self, feat_config, s, is_train=False):
        """
        Process numerical features.
        """
        hash_buckets = feat_config.get('hash_buckets', None)
        emb_dim = feat_config.get('emb_dim', None)
        discretization = True if (hash_buckets or emb_dim) else False
        normalize = feat_config.get('norm', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (discretization and normalize), f'hash_buckets/emb_dim and norm cannot be set at the same time: {feat_config}'

        if is_train:
            # update mean, std, min, max
            feat_config['type'] = 'sparse' if discretization else 'dense'

            if 'mean' not in feat_config or 'std' not in feat_config or self.numerical_update_stats:
                feat_config['mean'], feat_config['std'] = self.update_meanstd(s, feat_config.get('freq_cnt', 0), mean=feat_config.get('mean'), std=feat_config.get('std'))
                feat_config['freq_cnt'] = feat_config.get('freq_cnt', 0) + len(s)

            if 'min' not in feat_config or 'max' not in feat_config or self.numerical_update_stats:
                feat_config['min'], feat_config['max'] = self.update_minmax(s, min_val=feat_config.get('min'), max_val=feat_config.get('max'))

            if self.verbose:
                print(f'Feature {feat_config["name"]} mean: {feat_config["mean"]}, std: {feat_config["std"]}, min: {feat_config["min"]}, max: {feat_config["max"]}')

            if discretization:
                hash_buckets = 10 if hash_buckets is None else hash_buckets

                bins = np.percentile(s[s.notna()], q=np.linspace(0, 100, num=hash_buckets))
                non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                feat_config['vocab'] = list(bins[non_adjacent_duplicates])

                feat_config['vocab'] = [np.NaN, float('-inf')] + feat_config['vocab'] + [float('inf')]

        if normalize == 'std':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['mean']) / feat_config['std']
        elif normalize == '[0,1]':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif discretization:
            bins = [v for v in feat_config['vocab'] if not np.isnan(v)]
            s = pd.cut(s, bins=bins, labels=False, right=True) + 1
            s = s.fillna(0).astype(int)  # index 0 is for nan values

        return s, feat_config

    def process_list(self, feat_config, s, is_train=False):
        """
        Process list features.
        """
        dtype = feat_config['dtype']
        maxlen = feat_config.get('maxlen', None)
        if maxlen:
            s = s.map(lambda x: x[-maxlen:] if isinstance(x, list) else x)
        flat_s = s.explode()
        if dtype == 'category':
            flat_s, updated_f = self.process_category(feat_config, flat_s, is_train)
        elif dtype == 'numerical':
            flat_s, updated_f = self.process_numerical(feat_config, flat_s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')
        s = flat_s.groupby(level=0).agg(list)
        # padding
        max_len = s.map(len).max()
        padding_maxlen = feat_config.get('maxlen', None)
        if padding_maxlen:
            max_len = min([padding_maxlen, max_len])
        s = s.map(lambda x: [self.list_padding_value] * (max_len - len(x)) + x if len(x) < max_len else x[-max_len:])
        return s, updated_f

    def _transform_one(self, s, f, is_train=False):
        """
        Transform a single feature based on the feature configuration.
        """
        fname = f['name']
        dtype = f['dtype']
        islist = f.get('islist', None)
        pre_transform = f.get('pre_transform', None)

        # pre-process
        if pre_transform:
            s = s.map(pre_transform)

        if self.verbose:
            print(f'Processing feature {fname}...')

        if islist:
            updated_s, updated_f = self.process_list(f, s, is_train)
        elif dtype == 'category':
            updated_s, updated_f = self.process_category(f, s, is_train)
        elif dtype == 'numerical':
            updated_s, updated_f = self.process_numerical(f, s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        return updated_s, updated_f
    
    def get_feat_configs(self):
        return self.feat_configs

class DataFrameDataset(Dataset):
    '''
    Var-length supported pytorch dataset for DataFrame.
    '''
    def __init__(self, df, feat_configs, target_cols=None, is_raw=False, **kwargs):
        """
        Args:
            df: pandas DataFrame
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8}, # sequence feature
                ]
            target_cols: list of str, target columns
            is_raw: bool, whether the input DataFrame is raw data without feature transforming
            kwargs: mainly for feature transforming parameters when is_raw=True
                n_jobs: int, number of parallel jobs
                list_padding_value: int, padding value for list features
                category_force_hash: bool, whether to force hash all category features
                category_upper_lower_sensitive: bool, whether category features are upper/lower case sensitive
                numerical_update_stats: bool, whether to update mean, std, min, max for numerical features
                outliers_category: list, outliers for category features
                outliers_numerical: list, outliers for numerical features
        """
        n_jobs = kwargs.get('n_jobs', 1) # os.cpu_count()
        verbose = kwargs.get('verbose', False)
        
        if is_raw:
            assert 'is_train' in kwargs, 'is_train parameter should be provided when is_raw=True'
            is_train = kwargs['is_train']
            
            self.transformer = FeatureTransformer(
                feat_configs,
                category_force_hash=kwargs.get('category_force_hash', False),
                category_upper_lower_sensitive=kwargs.get('category_upper_lower_sensitive', False),
                numerical_update_stats=kwargs.get('numerical_update_stats', False),
                list_padding_value=kwargs.get('list_padding_value', -100),
                outliers_category=kwargs.get('outliers_category', []),
                outliers_numerical=kwargs.get('outliers_numerical', []),
                verbose=verbose
            )
            
            df = self.transformer.transform(df, is_train=is_train, n_jobs=n_jobs)
            if verbose:
                print(f'==> Feature transforming (is_train={is_train}) done...')

        self.dense_cols = [f['name'] for f in feat_configs if f['type'] == 'dense' and not f.get('islist')]
        self.seq_dense_cols = [f['name'] for f in feat_configs if f['type'] == 'dense' and f.get('islist')]
        self.sparse_cols = [f['name'] for f in feat_configs if f['type'] == 'sparse' and not f.get('islist')]
        self.seq_sparse_cols = [f['name'] for f in feat_configs if f['type'] == 'sparse' and f.get('islist')]
        self.weight_cols_mapping = {f['name']: f.get('weight_col') for f in feat_configs if f['type'] == 'sparse' if f.get('weight')}    
        self.target_cols = target_cols

        if verbose:
            print(f'==> Dense features: {self.dense_cols}')
            print(f'==> Sparse features: {self.sparse_cols}')
            print(f'==> Sequence dense features: {self.seq_dense_cols}')
            print(f'==> Sequence sparse features: {self.seq_sparse_cols}')
            print(f'==> Weight columns mapping: {self.weight_cols_mapping}')
            print(f'==> Target columns: {self.target_cols}')

        self.total_samples = len(df)
        self.padding_value = kwargs.get('list_padding_value', -100)

        self.convert_to_tensors(df)

        if verbose:
            print(f'==> Finished dataset initialization, total samples: {self.total_samples}')

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        features = {}
        if hasattr(self, 'dense_data'):
            features.update( {'dense_features': self.dense_data[idx, :]} )
        if hasattr(self, 'seq_dense_features'):
            features.update( {'seq_dense_features': self.seq_dense_data[idx, :]} )
            features.update({f'{k}': v[idx,:] for k,v in self.sparse_data.items()})
            features.update({f'{k}': v[idx,:] for k,v in self.sparse_data.items()})
        features.update({f'{k}': v[idx,:] for k,v in self.sparse_data.items()})
        features.update({f'{k}': v[idx,:] for k,v in self.seq_sparse_data.items()})
        features.update({f'{k}': v[idx,:] for k,v in self.weight_data.items()})

        if hasattr(self, 'target'):
            return features, self.target[idx,:]
        else:
            return features

    def convert_to_tensors(self, df):
        if self.dense_cols is not None and len(self.dense_cols) > 0:
            self.dense_data = torch.tensor(df[self.dense_cols].values, dtype=torch.float32)
        
        if self.seq_dense_cols is not None and len(self.seq_dense_cols) > 0: 
            self.seq_dense_data = torch.tensor( 
                df[self.seq_dense_cols].applymap(lambda x: [np.nansum(x), np.nanmax(x), np.nanmin(x)]).values.tolist(), 
                dtype=torch.float32
            )
            self.seq_dense_data = self.seq_dense_data.view(len(df), -1) # num x dim_feature

        self.weight_data = {}
        self.sparse_data = {}
        for col in self.sparse_cols:
            self.sparse_data[col] = torch.tensor(df[[col]].values, dtype=torch.int)
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                self.weight_data[f'{col}_wgt'] = torch.tensor(df[[weight_col]].values, dtype=torch.float)
        
        # for sparse sequences, padding to the maximum length
        self.seq_sparse_data = {}
        for col in self.seq_sparse_cols:
            max_len = df[col].apply(len).max()
            self.seq_sparse_data[col] = df[col].apply( 
                lambda x: [self.padding_value] * (max_len - len(x)) + x if len(x) < max_len else x[-max_len:])
            self.seq_sparse_data[col] = torch.tensor(self.seq_sparse_data[col].values.tolist(), dtype=torch.int)
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                self.weight_data[f'{col}_wgt'] = df[weight_col].apply( 
                    lambda x: [0.] * (max_len - len(x)) + x if len(x) < max_len else x[-max_len:])
                self.weight_data[f'{col}_wgt'] = torch.tensor(
                    self.weight_data[f'{col}_wgt'].values.tolist(), dtype=torch.int)

        if self.target_cols is not None:
            self.target = torch.tensor(df[self.target_cols].values.tolist(), dtype=torch.float32)
    
    def to(self, device):
        if hasattr(self, 'dense_data'):
            self.dense_data = self.dense_data.to(device)
        if hasattr(self, 'seq_dense_data'):
            self.seq_dense_data = self.seq_dense_data.to(device)

        self.sparse_data = {k: v.to(device) for k,v in self.sparse_data.items()}
        self.seq_sparse_data = {k: v.to(device) for k,v in self.seq_sparse_data.items()}
        self.weight_data = {k: v.to(device) for k,v in self.weight_data.items()}

        if hasattr(self, 'target'):
            self.target = self.target.to(device)

        return self

class IterableDataFrameDataset(IterableDataset):
    '''
    Var-length supported pytorch iterable dataset for DataFrame.
    '''
    def __init__(self, inputs, feat_configs, target_cols=None, is_raw=False, chunksize=None, **kwargs):
        """
        Args:
            inputs: list of str, input files
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8}, # sequence feature
                ]
            target_cols: list of str, target columns
            is_raw: bool, whether the input DataFrame is raw data without feature transforming
            kwargs: mainly for feature transforming parameters when is_raw=True
                n_jobs: int, number of parallel jobs
                list_padding_value: int, padding value for list features
                category_force_hash: bool, whether to force hash all category features
                category_upper_lower_sensitive: bool, whether category features are upper/lower case sensitive
                numerical_update_stats: bool, whether to update mean, std, min, max for numerical features
                outliers_category: list, outliers for category features
                outliers_numerical: list, outliers for numerical features
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        self.inputs = inputs
        self.chunksize = chunksize

        self.n_jobs = kwargs.get('n_jobs', 1) # os.cpu_count()

        if is_raw:
            assert 'is_train' in kwargs, 'is_train parameter should be provided when is_raw=True'
            is_train = kwargs['is_train']
            
            self.transformer = FeatureTransformer(
                feat_configs,
                category_force_hash=kwargs.get('category_force_hash', False),
                category_upper_lower_sensitive=kwargs.get('category_upper_lower_sensitive', False),
                numerical_update_stats=kwargs.get('numerical_update_stats', False),
                list_padding_value=kwargs.get('list_padding_value', -100),
                outliers_category=kwargs.get('outliers_category', []),
                outliers_numerical=kwargs.get('outliers_numerical', []),
                verbose=kwargs.get('verbose', False)
            )

            if is_train == True:
                # update feat_configs by go through all data
                print('==> Start to update feat_configs (is_train=True) by going through all data...')
                for df in self.read_inputs(inputs, chunksize=self.chunksize):
                    self.transformer.transform(df, is_train=True, n_jobs=self.n_jobs)
                print('==> Finished updating feat_configs (is_train=True) ...')

            self.feat_configs = self.transformer.get_feat_configs()
        else:
            self.transformer = None
            self.feat_configs = feat_configs

        self.target_cols = target_cols
    
    def read_inputs(self, inputs: list, chunksize=None):
        def read_pickle(path, chunksize):
            try:
                df = pd.read_pickle(path)
            except:
                import joblib
                df = joblib.load(path)
          
            # for i in range(0, len(df), chunksize):
            #     yield df.iloc[i:i+chunksize]
            return [df]

        for input in inputs:
            if isinstance(input, pd.DataFrame):
                reader = lambda input, chunksize: [input]
            elif input.endswith('.h5') or input.endswith('.hdf5'):
                reader = pd.read_hdf
            elif input.endswith('.csv'):
                reader = pd.read_csv
            elif input.endswith('.pkl') or input.endswith('.pickle'):
                reader = read_pickle
            else:
                raise ValueError(f'Unsupported input format: {input}')

            for chunk in reader(input, chunksize):
                yield chunk

    def __iter__(self):
        for df in self.read_inputs(self.inputs, chunksize=self.chunksize):
            if self.transformer:
                self.transformer.verbose = False
                df = self.transformer.transform(df, is_train=False, n_jobs=self.n_jobs)
            ds = DataFrameDataset(df, self.feat_configs, target_cols=self.target_cols, is_raw=False, verbose=False)
            if hasattr(self, 'device'):
                ds = ds.to(self.device)
            yield from ds

    def to(self, device):
        self.device = device
        return self