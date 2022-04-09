# coding: utf-8

import pandas as pd
import numpy as np
import deepchem as dc
from sklearn.preprocessing import StandardScaler


def load_dataset(data_file, descriptor_set_file, encoding="GBK"):
    """Load raw data and return pd.DataFrame format.

    Args:
        data_file (*.csv): data containing SMILES, molecular properties and other essential information.
        descriptor_set_file (*.pkl.zip): descriptor dictionary, contianing descriptor of specific SMILES and specific descriptor types.
        encoding (str, optional): encoding type, utf-8 or GBK is OK. Defaults to "GBK".

    Returns:
        (pd.DataFrame, pd.DataFrame): (data_df, desc_set_df)
    """
    data_df = pd.read_csv(data_file, encoding=encoding)
    desc_set_df = pd.read_pickle(descriptor_set_file)

    # NaN Inf
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_df.fillna(value=0, inplace=True)

    return data_df, desc_set_df


def single_descriptor(data_df, desc_set_df, smiles_col, task_col, feat_type, split_mode="RandomSplitter", frac_train=0.9, rnd=42):
    """Data processing and splitting--Generate one type of molecular descriptor.

    Args:
        data_df (pd.DataFrame): data containing SMILES, molecular properties and other essential information.
        desc_set_df (pd.DataFrame): descriptor dictionary, contianing descriptor of specific SMILES and specific descriptor types.
        smiles_col (str): SMILES column name
        task_col (str): target training/prediction property
        feat_type (str): molecular descriptor type.
        split_mode (str, optional): [RandomSplitter, ScaffoldSplitter, SingletaskStratifiedSplitter]. Defaults to "RandomSplitter".
        frac_train (float, optional): data split ratio of train/test. Defaults to 0.9.
        rnd (int, optional): random number. Defaults to 42.

    Returns:
        (np.array, np.array, np.array, np.array): (X_train, X_test, y_train, y_test)
    """

    # Descriptor & value
    # -------------------
    X = np.array([desc_set_df[feat_type][smi]
                 for smi in data_df[smiles_col].values])
    X = np.nan_to_num(X)
    y = data_df[task_col].values

    # data spliting
    # -------------
    # set SMILES as ids
    X_smiles = data_df[smiles_col].values
    
    # 1. RandomSplitter
    if split_mode == "RandomSplitter":
        splitter = dc.splits.RandomSplitter()
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
    # 2. ScaffoldSplitter
    elif split_mode == "ScaffoldSplitter":
        splitter = dc.splits.ScaffoldSplitter()
        dataset = dc.data.NumpyDataset(X=X, y=y, ids=X_smiles)
        
    # 3. SingletaskStratifiedSplitter
    elif split_mode == "SingletaskStratifiedSplitter":
        splitter = dc.splits.SingletaskStratifiedSplitter(task_number=0)
        y = np.expand_dims(y, axis=1)
        print(f"y.shape: {y.shape}")
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, frac_train=frac_train, seed=rnd)

    X_train = train_dataset.X
    y_train = train_dataset.y.ravel()
    X_test = test_dataset.X
    y_test = test_dataset.y.ravel()

    # Feature Scaling
    # -------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def SPOC_descriptor(data_df, desc_set_df, smiles_col, task_col, feat_type_S, feat_type_POC, split_mode="RandomSplitter", frac_train=0.9, rnd=42):
    """Data processing and splitting--Generate S+POC descriptor
        
    Args:
        data_df (pd.DataFrame): data containing SMILES, molecular properties and other essential information.
        desc_set_df (pd.DataFrame): descriptor dictionary, contianing descriptor of specific SMILES and specific descriptor types.
        smiles_col (str): SMILES column name
        task_col (str): target training/prediction property
        feat_type_S (str): molecular type type of S-descriptor
        feat_type_POC (str): molecular type type of POC-descriptor
        split_mode (str, optional): [RandomSplitter, ScaffoldSplitter, SingletaskStratifiedSplitter]. Defaults to "RandomSplitter".
        frac_train (float, optional): data split ratio of train/test. Defaults to 0.9.
        rnd (int, optional): random number. Defaults to 42.

    Returns:
        (np.array, np.array, np.array, np.array): (X_train, X_test, y_train, y_test)
    """
    # descriptor & value
    # -------------------
    X_S = np.array([desc_set_df[feat_type_S][smi]
                   for smi in data_df[smiles_col].values])
    X_POC = np.array([desc_set_df[feat_type_POC][smi]
                     for smi in data_df[smiles_col].values])
    X = np.hstack((X_S, X_POC))
    X = np.nan_to_num(X)
    y = data_df[task_col].values

    # data spliting
    # -------------
    # set SMILES as ids
    X_smiles = data_df[smiles_col].values
    
    # 1. RandomSplitter
    if split_mode == "RandomSplitter":
        splitter = dc.splits.RandomSplitter()
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
    # 2. ScaffoldSplitter
    elif split_mode == "ScaffoldSplitter":
        splitter = dc.splits.ScaffoldSplitter()
        dataset = dc.data.NumpyDataset(X=X, y=y, ids=X_smiles)
        
    # 3. SingletaskStratifiedSplitter
    elif split_mode == "SingletaskStratifiedSplitter":
        splitter = dc.splits.SingletaskStratifiedSplitter(task_number=0)
        y = np.expand_dims(y, axis=1)
        print(f"y.shape: {y.shape}")
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, frac_train=frac_train, seed=rnd)

    X_train = train_dataset.X
    y_train = train_dataset.y.ravel()
    X_test = test_dataset.X
    y_test = test_dataset.y.ravel()

    # Feature Scaling
    # -------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
