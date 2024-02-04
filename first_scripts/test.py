import pandas as pd
import numpy as np
import tqdm
import os
import logging
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


# Logging
def get_log_file_name():
    counter = 1
    file_name = f"logs/train.log"
    while os.path.exists(file_name):
        counter += 1
        file_name = f"logs/train_{counter}.log"
    return file_name


logging.basicConfig(filename=get_log_file_name(), encoding="utf-8", level=logging.DEBUG)

# Hyperparameters
hours = 8
segments = int(166 / hours)
epochs = 500
batch_size = 16
lr = 0.001
weight_decay = 0.01
num_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    X = torch.load(f"X{segments}.t")
    y = torch.load(f"y{segments}.t")
except:
    train_df = pd.read_csv("datasets/outputs/train_df_nt_after_norm.csv")
    train_df = train_df[
        ~train_df["ID"].isin(
            ["D1", "A1", "A4", "C4", "E3", "F1", "F4", "G1", "G3", "G4", "I2"]
        )
    ]
    logging.info(train_df.shape)

    # Create an empty DataFrame with the same column structure as train_df
    expanded_df = pd.DataFrame(columns=train_df.columns)

    # Iterating over each unique ID
    for unique_id in tqdm.tqdm(train_df["ID"].unique()):

        id_rows = train_df.loc[train_df["ID"] == unique_id]

        # Dividing the current ID into the desired number of segments
        segment_length = len(id_rows) // segments

        for i in range(segments):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            new_id = f"{unique_id}_{i+1}"
            new_rows = id_rows.iloc[segment_start:segment_end].copy()
            new_rows["ID"] = new_id
            expanded_df = pd.concat([expanded_df, new_rows])

    expanded_df.reset_index(drop=True, inplace=True)
    train_df = expanded_df.astype(train_df.dtypes)

    # Preprocess the data
    continuous_features = ["sequence"]

    min_count = train_df.groupby("ID").size().min()
    print(f"Duree de chaque sequence = {min_count/36000} heures")

    sequences = []
    current_sequence = []

    for id in tqdm.tqdm(train_df["ID"].unique()):
        current_sequence = train_df[train_df["ID"] == id][continuous_features].values
        sequences.append(current_sequence[:min_count])

    X = torch.tensor(np.array(sequences), dtype=torch.float64)
    y = torch.tensor(train_df.groupby("ID")["label"].first().values, dtype=torch.int64)

    torch.save(X, f"X{segments}.t")
    torch.save(y, f"y{segments}.t")

    X = torch.load(f"X{segments}.t")
    y = torch.load(f"y{segments}.t")


input_size = X.shape
output_size = 1
print(f"input_size = {input_size}, output_size = {output_size}")
