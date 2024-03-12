import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def traintest_split(df, test_size=0.2, shuffle=True, group_id=None, random_state=0):
    if group_id is None:
        train_df, test_df = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)
    else:
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=random_state)
        split = splitter.split(df, groups=df[group_id])
        train_inds, test_inds = next(split)
        
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(train_inds)
            np.random.shuffle(test_inds)

        train_df = df.iloc[train_inds]
        test_df = df.iloc[test_inds]
    return train_df, test_df

def traintest_split_by_date(df, date_col, test_size=0.2, shuffle=True, random_state=0):
    uniq_dates = df[date_col].unique()
    uniq_dates = np.sort(uniq_dates)
    n_train_dates = int(len(uniq_dates) * (1 - test_size)) + 1
    assert n_train_dates < len(uniq_dates)
    split_pos = uniq_dates[n_train_dates]
    print(f'Train set date range [{uniq_dates[0]}, {split_pos}]')
    print(f'Test  set date range ({split_pos}, {uniq_dates[-1]}]')
    
    train_df = df[df[date_col] <= split_pos]
    test_df  = df[df[date_col] >  split_pos]
    if shuffle:
        train_df = train_df.sample(frac=1)
    return train_df, test_df