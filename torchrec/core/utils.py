import pandas as pd
import numpy as np

def auto_generate_feature_configs(
        df: pd.DataFrame, 
        columns: list = None,
        min_emb_dim: int = 6,
        max_emb_dim: int = 30, 
        max_hash_buckets: int = 1000000,
        seq_max_len: int = 256
    ):
    feat_configs = []

    if columns is None:
        columns = df.columns
    
    for col in columns:
        col_info = {"name": col}
        
        # Check if column contains sequences (lists)
        if df[col].apply(lambda x: isinstance(x, list)).any():
            col_info["dtype"] = "category"
            col_info["islist"] = True
            unique_values = set(val for sublist in df[col] for val in sublist)
            num_unique = len(unique_values)
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_info["dtype"] = "numerical"
            col_info["norm"] = "std"  # Standard normalization
            col_info["mean"] = df[col].mean()
            col_info["std"] = df[col].std()
            feat_configs.append(col_info)
            continue
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            col_info["dtype"] = "category"
            unique_values = df[col].unique()
            num_unique = len(unique_values)
        else:
            continue
        
        if col_info["dtype"] == "category":
            # Calculate embedding dimension
            # emb_dim = int(np.sqrt(num_unique))
            emb_dim = int(np.log2(num_unique))
            emb_dim = min(max(emb_dim, min_emb_dim), max_emb_dim)  # Example bounds
            col_info["emb_dim"] = emb_dim

            # Use hash bucket for high cardinality categorical features or unique values is high
            if num_unique > 0.2 * len(df) or num_unique > max_hash_buckets:
                # Use hash bucket for high cardinality categorical features
                col_info["hash_buckets"] = min(num_unique, max_hash_buckets)
            
            col_info["min_freq"] = 3  # Example minimum frequency

        # If islist features too long, set max_len to truncate
        if col_info.get("islist", False):
            max_len = max(len(x) for x in df[col])
            col_info["max_len"] = min(max_len, seq_max_len)
        
        # Add the column info to feature configs
        feat_configs.append(col_info)
    
    return feat_configs
