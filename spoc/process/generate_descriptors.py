# coding: utf-8

import os
import time
import pandas as pd
from urllib.request import urlretrieve
from spoc.descriptor import all_descriptor_generator as adg


def load_data(task_name, url, data_file, feature_file, smiles_col, test_mode):

    t1 = time.time()

    print('start')

    if test_mode == "test":
        start = -2
    elif test_mode == "production":
        start = 0

    if not os.path.exists(data_file):
        urlretrieve(url, data_file)

    # load data
    data = pd.read_csv(data_file)
    smiles = data[smiles_col].values
    smiles_set = list(set(smiles))[start:]
    print(f"data size: {len(smiles_set)}")

    # descriptor generation
    feature_dict = adg.feature_dic_generation(smiles_set)

    # save as *.pkl.zip
    df = pd.DataFrame(feature_dict)
    df.to_pickle(feature_file, compression="zip")

    print(f"Descriptor generation of task {task_name}: done!")

    t2 = time.time()
    print(f"t2-t1: {round(t2-t1,2)} s")

