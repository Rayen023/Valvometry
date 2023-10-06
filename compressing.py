import pandas as pd
import numpy as np

# Define the data types for each column
dtype_dict = {
    'sequence': np.float32,
    'label': 'category',
    'ID': 'category',
}

train_df = pd.read_csv("datasets/outputs/train_df_nt_after_norm.csv" , dtype=dtype_dict)
#train_df = train_df[~train_df['ID'].isin(['D1', 'A1', 'A4', 'C4', 'E3', 'F1', 'F4', 'G1', 'G3', 'G4', 'I2'])]


train_df.to_csv("datasets/outputs/train_df_nt_after2_norm.csv", index = False)